import copy
import os

import numpy as np
import time
import gym

from datetime import datetime
from tensorboardX import SummaryWriter
import torch
from torch import nn
import torch.distributed as dist

from timechamber.learning.lib.core import common_losses
from timechamber.learning.lib.core.dignostics import DefaultDiagnostics, PpoDiagnostics
from timechamber.learning.lib.core.interval_summary_writer import IntervalSummaryWriter
from timechamber.learning.lib.core.experience import ExperienceBuffer
from timechamber.learning.lib.core import schedulers
from timechamber.learning.lib.core import torch_ext
from timechamber.learning.lib.core.moving_mean_std import GeneralizedMovingStats
from rl_games.common import vecenv

from abc import ABC
from abc import abstractmethod, abstractproperty

def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

class BaseAlgorithm(ABC):
    def __init__(self, base_name, config):
        pass

    @abstractproperty
    def device(self):
        pass

    @abstractmethod
    def clear_stats(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def get_full_state_weights(self):
        pass

    @abstractmethod
    def set_full_state_weights(self, weights, set_epoch):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def set_weights(self, weights):
        pass

    # Get algo training parameters
    @abstractmethod
    def get_param(self, param_name):
        pass

    # Set algo training parameters
    @abstractmethod
    def set_param(self, param_name, param_value):
        pass

class A2CBase(BaseAlgorithm):

    def __init__(self, base_name, params):
        self.network_config = params['network']
        self.config = config = params['config']
        pbt_str = ''

        # generating a new name with the timestamp every time.
        full_experiment_name = config.get('full_experiment_name', None)
        if full_experiment_name:
            print(f'Exact experiment name requested from command line: {full_experiment_name}')
            self.experiment_name = full_experiment_name
        else:
            self.experiment_name = config['name'] + pbt_str + datetime.now().strftime("_%d-%H-%M-%S")

        self.algo_observer1 = config['features']['observer']
        self.algo_observer1.before_init(base_name, config, self.experiment_name)
        self.algo_observer2 = config['features']['observer']
        self.algo_observer2.before_init(base_name, config, self.experiment_name)
        # self.load_networks(params) 

        self.multi_gpu = config.get('multi_gpu', False)

        # multi-gpu/multi-node data
        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

        self.curr_frames = 0

        if self.multi_gpu:
            # local rank of the GPU in a node
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            self.global_rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))

            dist.init_process_group("nccl", rank=self.global_rank, world_size=self.world_size)

            self.device_name = 'cuda:' + str(self.local_rank)
            config['device'] = self.device_name
            if self.global_rank != 0:
                config['print_stats'] = False
                config['lr_schedule'] = None

        self.use_diagnostics = config.get('use_diagnostics', False)

        if self.use_diagnostics and self.global_rank == 0:
            self.diagnostics1 = PpoDiagnostics()
            self.diagnostics2 = PpoDiagnostics()
        else:
            self.diagnostics1 = DefaultDiagnostics()
            self.diagnostics2 = DefaultDiagnostics()

        self.network_path = config.get('network_path', "./nn/")
        self.log_path = config.get('log_path', "runs/")
        self.env_config = config.get('env_config', {})
        self.num_actors = config['num_actors']                      # ?
        self.env_name = config['env_name']

        # self.env_info is environment config class cfg.env
        self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
        self.env_info = self.vec_env.get_env_info()
        self.num_agents = self.vec_env.num_agents
        self.num_agents1 = self.env_info['num_agents'][0]
        self.num_agents2 = self.env_info['num_agents'][1]

        self.ppo_device = config.get('device', 'cuda:0')
        self.value_size = self.env_info.get('value_size',1)
        self.observation_space = self.env_info['observation_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)

        self.truncate_grads = self.config.get('truncate_grads', False)

        self.self_play_config = self.config.get('self_play_config', None)
        self.has_self_play_config = self.self_play_config is not None

        self.self_play = config.get('self_play', False)
        self.save_freq = config.get('save_frequency', 0)
        self.save_best_after = config.get('save_best_after', 100)
        self.print_stats = config.get('print_stats', True)
        self.name = base_name

        self.ppo = config.get('ppo', True)
        self.max_epochs = self.config.get('max_epochs', -1)
        self.max_frames = self.config.get('max_frames', -1)

        self.is_adaptive_lr = config['lr_schedule'] == 'adaptive'
        self.linear_lr = config['lr_schedule'] == 'linear'
        self.schedule_type = config.get('schedule_type', 'legacy')

        # Setting learning rate scheduler
        if self.is_adaptive_lr:
            self.kl_threshold = config['kl_threshold']
            self.scheduler1 = schedulers.AdaptiveScheduler(self.kl_threshold)
            self.scheduler2 = schedulers.AdaptiveScheduler(self.kl_threshold)

        elif self.linear_lr:
            if self.max_epochs == -1 and self.max_frames == -1:
                print("Max epochs and max frames are not set. Linear learning rate schedule can't be used, switching to the contstant (identity) one.")
                self.scheduler1 = schedulers.IdentityScheduler()
                self.scheduler2 = schedulers.IdentityScheduler()
            else:
                use_epochs = True
                max_steps = self.max_epochs

                if self.max_epochs == -1:
                    use_epochs = False
                    max_steps = self.max_frames

                self.scheduler1 = schedulers.LinearScheduler(float(config['learning_rate']), 
                    max_steps = max_steps,
                    use_epochs = use_epochs, 
                    apply_to_entropy = config.get('schedule_entropy', False),
                    start_entropy_coef = config.get('entropy_coef'))
                self.scheduler2 = schedulers.LinearScheduler(float(config['learning_rate']), 
                    max_steps = max_steps,
                    use_epochs = use_epochs, 
                    apply_to_entropy = config.get('schedule_entropy', False),
                    start_entropy_coef = config.get('entropy_coef'))
        else:
            self.scheduler1 = schedulers.IdentityScheduler()
            self.scheduler2 = schedulers.IdentityScheduler()

        self.e_clip = config['e_clip']
        self.clip_value = config['clip_value']
        self.rewards_shaper = config['reward_shaper']
        self.horizon_length = config['horizon_length']

        self.normalize_advantage = config['normalize_advantage']
        self.normalize_rms_advantage = config.get('normalize_rms_advantage', False)
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)
        self.truncate_grads = self.config.get('truncate_grads', False)

        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k,v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape1 = self.observation_space[0].shape
            self.obs_shape2 = self.observation_space[1].shape
 
        self.critic_coef1 = config['critic_coef']
        self.critic_coef2 = config['critic_coef']
        self.grad_norm1 = config['grad_norm']
        self.grad_norm2 = config['grad_norm']
        self.gamma1 = self.config['gamma']
        self.gamma2 = self.config['gamma']
        self.tau1 = self.config['tau']
        self.tau2 = self.config['tau']

        self.games_to_track1 = self.config.get('games_to_track', 100)
        self.games_to_track2 = self.config.get('games_to_track', 100)
        print('current training device:', self.ppo_device)
        self.game_rewards1 = torch_ext.AverageMeter(self.value_size, self.games_to_track1).to(self.ppo_device)
        self.game_shaped_rewards1 = torch_ext.AverageMeter(self.value_size, self.games_to_track1).to(self.ppo_device)
        self.game_lengths1 = torch_ext.AverageMeter(1, self.games_to_track1).to(self.ppo_device)
        self.game_rewards2 = torch_ext.AverageMeter(self.value_size, self.games_to_track2).to(self.ppo_device)
        self.game_shaped_rewards2 = torch_ext.AverageMeter(self.value_size, self.games_to_track2).to(self.ppo_device)
        self.game_lengths2 = torch_ext.AverageMeter(1, self.games_to_track2).to(self.ppo_device)
        self.obs1 = None
        self.obs2 = None

        self.batch_size1 = self.horizon_length * self.num_actors * self.num_agents1  # ?
        self.batch_size2 = self.horizon_length * self.num_actors * self.num_agents2  # ?
        self.batch_size = self.batch_size1 + self.batch_size2
        self.batch_size_envs = self.horizon_length * self.num_actors    # ?

        assert(('minibatch_size_per_env' in self.config) or ('minibatch_size' in self.config))
        self.minibatch_size_per_env = self.config.get('minibatch_size_per_env', 0)
        self.minibatch_size = self.config.get('minibatch_size', self.num_actors * self.minibatch_size_per_env)

        self.num_minibatches1 = self.batch_size1 // self.minibatch_size
        assert(self.batch_size1 % self.minibatch_size == 0)
        self.num_minibatches2 = self.batch_size2 // self.minibatch_size
        assert(self.batch_size2 % self.minibatch_size == 0)

        self.mini_epochs_num = self.config['mini_epochs']

        self.mixed_precision = self.config.get('mixed_precision', False)
        self.scaler1 = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.scaler2 = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.last_lr1 = self.config['learning_rate']
        self.last_lr2 = self.config['learning_rate']
        self.frame1 = 0
        self.frame2 = 0
        self.frame = 0
        self.update_time1 = 0
        self.update_time2 = 0
        self.mean_rewards1 = self.last_mean_rewards1 = -1000000000
        self.mean_rewards2 = self.last_mean_rewards2 = -1000000000
        self.play_time1 = 0
        self.play_time2 = 0
        self.epoch_num = 0
        self.curr_frames1 = 0
        self.curr_frames2 = 0
        # allows us to specify a folder where all experiments will reside
        self.train_dir = config.get('train_dir', 'runs')

        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir1 = os.path.join(self.experiment_dir, 'summaries1')
        self.summaries_dir2 = os.path.join(self.experiment_dir, 'summaries2')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir1, exist_ok=True)
        os.makedirs(self.summaries_dir2, exist_ok=True)

        self.entropy_coef1 = self.config['entropy_coef']
        self.entropy_coef2 = self.config['entropy_coef']

        if self.global_rank == 0:
            writer1 = SummaryWriter(self.summaries_dir1)
            writer2 = SummaryWriter(self.summaries_dir2)
            self.writer1 = writer1
            self.writer2 = writer2
        else:
            self.writer1 = self.writer2 = None

        self.value_bootstrap = self.config.get('value_bootstrap')
        self.use_smooth_clamp = self.config.get('use_smooth_clamp', False)

        if self.use_smooth_clamp:
            self.actor_loss_func1 = common_losses.smoothed_actor_loss
            self.actor_loss_func2 = common_losses.smoothed_actor_loss
        else:
            self.actor_loss_func1 = common_losses.actor_loss
            self.actor_loss_func2 = common_losses.actor_loss

        if self.normalize_advantage and self.normalize_rms_advantage:
            momentum1 = self.config.get('adv_rms_momentum', 0.5)
            momentum2 = self.config.get('adv_rms_momentum', 0.5)
            self.advantage_mean_std1 = GeneralizedMovingStats((1,), momentum=momentum1).to(self.ppo_device)
            self.advantage_mean_std2 = GeneralizedMovingStats((1,), momentum=momentum2).to(self.ppo_device)

        self.is_tensor_obses = False

        self.last_state_indices = None

        # features
        self.algo_observer1 = config['features']['observer']
        self.algo_observer2 = config['features']['observer']

        self.soft_aug = config['features'].get('soft_augmentation', None)
        self.has_soft_aug = self.soft_aug is not None
        # soft augmentation not yet supported
        assert not self.has_soft_aug

    def trancate_gradients_and_step(self):
        # if self.multi_gpu:
        #     # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
        #     all_grads_list = []
        #     for param in self.model.parameters():
        #         if param.grad is not None:
        #             all_grads_list.append(param.grad.view(-1))

        #     all_grads = torch.cat(all_grads_list)
        #     dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
        #     offset = 0
        #     for param in self.model.parameters():
        #         if param.grad is not None:
        #             param.grad.data.copy_(
        #                 all_grads[offset : offset + param.numel()].view_as(param.grad.data) / self.world_size
        #             )
        #             offset += param.numel()

        if self.truncate_grads:
            self.scaler1.unscale_(self.optimizer1)
            self.scaler2.unscale_(self.optimizer2)
            nn.utils.clip_grad_norm_(self.model1.parameters(), self.grad_norm1)
            nn.utils.clip_grad_norm_(self.model2.parameters(), self.grad_norm2)

        self.scaler1.step(self.optimizer1)
        self.scaler2.step(self.optimizer2)
        self.scaler1.update()
        self.scaler2.update()

    def write_stats1(self, total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames):
        # do we need scaled time?
        self.diagnostics1.send_info(self.writer1)
        self.writer1.add_scalar('performance/step_inference_rl_update_fps', curr_frames / scaled_time, frame)
        self.writer1.add_scalar('performance/step_inference_fps', curr_frames / scaled_play_time, frame)
        self.writer1.add_scalar('performance/step_fps', curr_frames / step_time, frame)
        self.writer1.add_scalar('performance/rl_update_time', update_time, frame)
        self.writer1.add_scalar('performance/step_inference_time', play_time, frame)
        self.writer1.add_scalar('performance/step_time', step_time, frame)
        self.writer1.add_scalar('losses/a_loss', torch_ext.mean_list(a_losses).item(), frame)
        self.writer1.add_scalar('losses/c_loss', torch_ext.mean_list(c_losses).item(), frame)

        self.writer1.add_scalar('losses/entropy', torch_ext.mean_list(entropies).item(), frame)
        self.writer1.add_scalar('info/last_lr', last_lr * lr_mul, frame)
        self.writer1.add_scalar('info/lr_mul', lr_mul, frame)
        self.writer1.add_scalar('info/e_clip', self.e_clip * lr_mul, frame)
        self.writer1.add_scalar('info/kl', torch_ext.mean_list(kls).item(), frame)
        self.writer1.add_scalar('info/epochs', epoch_num, frame)
        self.algo_observer1.after_print_stats(frame, epoch_num, total_time)
        
    def write_stats2(self, total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames):
        # do we need scaled time?
        self.diagnostics2.send_info(self.writer2)
        self.writer2.add_scalar('performance/step_inference_rl_update_fps', curr_frames / scaled_time, frame)
        self.writer2.add_scalar('performance/step_inference_fps', curr_frames / scaled_play_time, frame)
        self.writer2.add_scalar('performance/step_fps', curr_frames / step_time, frame)
        self.writer2.add_scalar('performance/rl_update_time', update_time, frame)
        self.writer2.add_scalar('performance/step_inference_time', play_time, frame)
        self.writer2.add_scalar('performance/step_time', step_time, frame)
        self.writer2.add_scalar('losses/a_loss', torch_ext.mean_list(a_losses).item(), frame)
        self.writer2.add_scalar('losses/c_loss', torch_ext.mean_list(c_losses).item(), frame)

        self.writer2.add_scalar('losses/entropy', torch_ext.mean_list(entropies).item(), frame)
        self.writer2.add_scalar('info/last_lr', last_lr * lr_mul, frame)
        self.writer2.add_scalar('info/lr_mul', lr_mul, frame)
        self.writer2.add_scalar('info/e_clip', self.e_clip * lr_mul, frame)
        self.writer2.add_scalar('info/kl', torch_ext.mean_list(kls).item(), frame)
        self.writer2.add_scalar('info/epochs', epoch_num, frame)
        self.algo_observer2.after_print_stats(frame, epoch_num, total_time)

    def set_eval(self):
        self.model1.eval()
        self.model2.eval()
        if self.normalize_rms_advantage:
            self.advantage_mean_std1.eval()
            self.advantage_mean_std2.eval()

    def set_train(self):
        self.model1.train()
        self.model2.train()
        if self.normalize_rms_advantage:
            self.advantage_mean_std1.train()
            self.advantage_mean_std2.train()

    def update_lr1(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.optimizer1.param_groups:
            param_group['lr'] = lr
            
    def update_lr2(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.optimizer2.param_groups:
            param_group['lr'] = lr

    def get_action_values1(self, obs):
        processed_obs = self._preproc_obs(obs['obs1'])
        self.model1.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs
        }

        with torch.no_grad():
            res_dict = self.model1(input_dict)

        return res_dict
    
    def get_action_values2(self, obs):
        processed_obs = self._preproc_obs(obs['obs2'])
        self.model2.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs
        }

        with torch.no_grad():
            res_dict = self.model2(input_dict)
        return res_dict

    def get_values1(self, obs):
        with torch.no_grad():
            self.model1.eval()
            processed_obs = self._preproc_obs(obs['obs1'])
            input_dict = {
                'is_train': False,
                'prev_actions': None, 
                'obs' : processed_obs,
            }
            result = self.model1(input_dict)
            value = result['values']
            return value
        
    def get_values2(self, obs):
        with torch.no_grad():
            self.model2.eval()
            processed_obs = self._preproc_obs(obs['obs2'])
            input_dict = {
                'is_train': False,
                'prev_actions': None, 
                'obs' : processed_obs,
            }
            result = self.model2(input_dict)
            value = result['values']
            return value

    @property
    def device(self):
        return self.ppo_device

    def reset_envs(self):
        self.obs = self.env_reset()
        self.obs1 = self.obs['obs1']
        self.obs2 = self.obs['obs2']

    def init_tensors(self):
        batch_size1 = self.num_agents1 * self.num_actors
        batch_size2 = self.num_agents2 * self.num_actors
        batch_size = batch_size1 + batch_size2
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'use_action_masks' : self.use_action_masks
        }

        self.experience_buffer1 = ExperienceBuffer(self.env_info, algo_info, self.ppo_device, 0)
        self.experience_buffer2 = ExperienceBuffer(self.env_info, algo_info, self.ppo_device, 1)

        val_shape1 = (self.horizon_length, batch_size1, self.value_size)
        val_shape2 = (self.horizon_length, batch_size2, self.value_size)
        current_rewards_shape1 = (batch_size1, self.value_size)
        current_rewards_shape2 = (batch_size2, self.value_size)
        self.current_rewards1 = torch.zeros(current_rewards_shape1, dtype=torch.float32, device=self.ppo_device)
        self.current_rewards2 = torch.zeros(current_rewards_shape2, dtype=torch.float32, device=self.ppo_device)
        self.current_shaped_rewards1 = torch.zeros(current_rewards_shape1, dtype=torch.float32, device=self.ppo_device)
        self.current_shaped_rewards2 = torch.zeros(current_rewards_shape2, dtype=torch.float32, device=self.ppo_device)
        self.current_lengths1 = torch.zeros(batch_size1, dtype=torch.float32, device=self.ppo_device)
        self.current_lengths2 = torch.zeros(batch_size2, dtype=torch.float32, device=self.ppo_device)
        self.dones = torch.ones((self.num_actors), dtype=torch.uint8, device=self.ppo_device)

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert(obs.dtype != np.int8)
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.ppo_device)
            else:
                obs = torch.FloatTensor(obs).to(self.ppo_device)
        return obs

    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or ('obs1' not in obs and 'obs2' not in obs):    
            print("Error!=============")
        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_step(self, action1, action2):
        action1 = self.preprocess_actions1(action1)
        action2 = self.preprocess_actions2(action2)
        obs, rewards, dones, infos = self.vec_env.step(action1, action2)

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(2)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=2)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos

    def env_reset(self):
        obs = self.vec_env.reset()
        obs = {'obs1': obs["obs1"], 'obs2': obs["obs2"]}
        obs = self.obs_to_tensors(obs)
        return obs

    def discount_values1(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma1 * nextvalues * nextnonterminal - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma1 * self.tau1 * nextnonterminal * lastgaelam
        return mb_advs
    
    def discount_values2(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma2 * nextvalues * nextnonterminal - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma2 * self.tau2 * nextnonterminal * lastgaelam
        return mb_advs

    def discount_values_masks1(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards, mb_masks):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)
        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)
            masks_t = mb_masks[t].unsqueeze(1)
            delta = (mb_rewards[t] + self.gamma1 * nextvalues * nextnonterminal  - mb_extrinsic_values[t])
            mb_advs[t] = lastgaelam = (delta + self.gamma1 * self.tau1 * nextnonterminal * lastgaelam) * masks_t
        return mb_advs
    
    def discount_values_masks2(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards, mb_masks):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)
        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)
            masks_t = mb_masks[t].unsqueeze(1)
            delta = (mb_rewards[t] + self.gamma2 * nextvalues * nextnonterminal  - mb_extrinsic_values[t])
            mb_advs[t] = lastgaelam = (delta + self.gamma2 * self.tau2 * nextnonterminal * lastgaelam) * masks_t
        return mb_advs



    def clear_stats(self):
        # batch_size = self.num_agents * self.num_actors
        self.game_rewards1.clear()
        self.game_rewards2.clear()
        self.game_shaped_rewards1.clear()
        self.game_shaped_rewards2.clear()
        self.game_lengths1.clear()
        self.game_lengths2.clear()
        self.mean_rewards1 = self.last_mean_rewards1 = -100500
        self.mean_rewards2 = self.last_mean_rewards2 = -100500
        self.algo_observer1.after_clear_stats()
        self.algo_observer2.after_clear_stats()

    def update_epoch(self):
        pass

    def train(self):
        pass

    def prepare_dataset(self, batch_dict):
        pass

    def train_epoch(self):
        self.vec_env.set_train_info(self.frame, self)

    def train_actor_critic(self, obs_dict, opt_step=True):
        pass

    def calc_gradients(self):
        pass

    def get_central_value(self, obs_dict):
        return self.central_value_net.get_value(obs_dict)

    def train_central_value(self):
        return self.central_value_net.train_net()

    def get_full_state_weights(self):
        state = self.get_weights()
        state['epoch'] = self.epoch_num
        state['frame1'] = self.frame1
        state['frame2'] = self.frame2
        state['frame'] = self.frame
        state['optimizer1'] = self.optimizer1.state_dict()
        state['optimizer2'] = self.optimizer2.state_dict()

        # This is actually the best reward ever achieved. last_mean_rewards is perhaps not the best variable name
        # We save it to the checkpoint to prevent overriding the "best ever" checkpoint upon experiment restart
        state['last_mean_rewards1'] = self.last_mean_rewards1
        state['last_mean_rewards2'] = self.last_mean_rewards2

        if self.vec_env is not None:                        # ?
            env_state = self.vec_env.get_env_state()
            state['env_state'] = env_state

        return state

    def set_full_state_weights(self, weights, set_epoch=True):

        self.set_weights(weights)
        if set_epoch:
            self.epoch_num = weights['epoch']
            self.frame1 = weights['frame1']
            self.frame2 = weights['frame2']
            self.frame = weights['frame']

        self.optimizer1.load_state_dict(weights['optimizer1'])
        self.optimizer2.load_state_dict(weights['optimizer2'])

        self.last_mean_rewards1 = weights.get('last_mean_rewards1', -1000000000)
        self.last_mean_rewards2 = weights.get('last_mean_rewards2', -1000000000)

        if self.vec_env is not None:
            env_state = weights.get('env_state', None)
            self.vec_env.set_env_state(env_state)

    def get_weights(self):
        state = self.get_stats_weights()
        state['model1'] = self.model1.state_dict()
        state['model2'] = self.model2.state_dict()
        return state

    def get_stats_weights(self, model_stats=False):
        state = {}
        if self.mixed_precision:
            state['scaler1'] = self.scaler1.state_dict()
            state['scaler2'] = self.scaler2.state_dict()
        if model_stats:
            if self.normalize_input:
                state['running_mean_std1'] = self.model1.running_mean_std.state_dict()
                state['running_mean_std2'] = self.model2.running_mean_std.state_dict()
            if self.normalize_value:
                state['reward_mean_std1'] = self.model1.value_mean_std.state_dict()
                state['reward_mean_std2'] = self.model2.value_mean_std.state_dict()
        return state

    def set_stats_weights(self, weights):
        if self.normalize_rms_advantage:
            self.advantage_mean_std1.load_state_dic(weights['advantage_mean_std1'])
            self.advantage_mean_std2.load_state_dict(weights['advantage_mean_std2'])
        if self.normalize_input and 'running_mean_std1' in weights and 'running_mean_std2' in weights:
            self.model1.running_mean_std.load_state_dict(weights['running_mean_std1'])
            self.model2.running_mean_std.load_state_dict(weights['running_mean_std2'])
        if self.normalize_value and 'normalize_value1' in weights and 'normalize_value2' in weights:
            self.model1.value_mean_std.load_state_dict(weights['reward_mean_std1'])
            self.model2.value_mean_std.load_state_dict(weights['reward_mean_std2'])
        if self.mixed_precision and 'scaler1' in weights and 'scaler2' in weights:
            self.scaler1.load_state_dict(weights['scaler1'])
            self.scaler2.load_state_dict(weights['scaler2'])

    def set_weights(self, weights):
        self.mode1.load_state_dict(weights['model1'])
        self.mode2.load_state_dict(weights['model2'])
        self.set_stats_weights(weights)

    def get_param(self, param_name):                                                        # deprecated?
        if param_name in [
            "grad_norm",
            "critic_coef", 
            "bounds_loss_coef",
            "entropy_coef",
            "kl_threshold",
            "gamma",
            "tau",
            "mini_epochs_num",
            "e_clip",
            ]:
            return getattr(self, param_name)
        elif param_name == "learning_rate":
            return self.last_lr
        else:
            raise NotImplementedError(f"Can't get param {param_name}")       

    def set_param(self, param_name, param_value):                                               # deprecated?
        if param_name in [
            "grad_norm",
            "critic_coef", 
            "bounds_loss_coef",
            "entropy_coef",
            "gamma",
            "tau",
            "mini_epochs_num",
            "e_clip",
            ]:
            setattr(self, param_name, param_value)
        elif param_name == "learning_rate":
            if self.global_rank == 0:
                if self.is_adaptive_lr:
                    raise NotImplementedError("Can't directly mutate LR on this schedule")
                else:
                    self.learning_rate = param_value

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
        elif param_name == "kl_threshold":
            if self.global_rank == 0:
                if self.is_adaptive_lr:
                    self.kl_threshold = param_value
                    self.scheduler.kl_threshold = param_value
                else:
                    raise NotImplementedError("Can't directly mutate kl threshold")
        else:
            raise NotImplementedError(f"No param found for {param_value}")

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k, v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def play_steps(self):
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length):
            # print(f"step{n}")
            if self.use_action_masks:                                                   # unimplemented
                masks = self.vec_env.get_action_masks()
                res_dict1 = self.get_masked_action_values1(self.obs, masks)
            else:
                res_dict1 = self.get_action_values1(self.obs)
                res_dict2 = self.get_action_values2(self.obs)
            self.experience_buffer1.update_data('obses', n, self.obs['obs1'])
            self.experience_buffer1.update_data('dones', n, self.dones.unsqueeze(1).repeat(1, self.num_agents1).view(-1))
            self.experience_buffer2.update_data('obses', n, self.obs['obs2'])
            self.experience_buffer2.update_data('dones', n, self.dones.unsqueeze(1).repeat(1, self.num_agents2).view(-1))

            # for k in update_list1:
            #     res_dict1[k] = res_dict1[k].view(self.batch_size1, self.num_agents1, -1)
            #     res_dict2[k] = res_dict2[k].view(self.batch_size2, self.num_agents2, -1)
            #     self.experience_buffer1.update_data(k, n, res_dict1[k])
            #     self.experience_buffer2.update_data(k, n, res_dict2[k])
            
            # res_dict1['actions'] = res_dict1['actions'].view(self.batch_size1 * self.num_agents1, -1)
            # res_dict2['actions'] = res_dict2['actions'].view(self.batch_size2 * self.num_agents2, -1)
            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict1['actions'], res_dict2['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)
            # print("rewards.shape", rewards.shape)
            shaped_rewards1 = self.rewards_shaper(rewards[:, :self.num_agents1].transpose(0,1).reshape(-1).unsqueeze(1))
            shaped_rewards2 = self.rewards_shaper(rewards[:, self.num_agents1:].transpose(0,1).reshape(-1).unsqueeze(1))
            # print("shaped_rewards1.shape", shaped_rewards1.shape)
            # print("shaped_rewards2.shape", shaped_rewards2.shape)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards1 += self.gamma1 * res_dict1['values'] * self.cast_obs(infos['time_outs'].unsqueeze(1).repeat(1, self.num_agents1).transpose(0,1).reshape(-1)).unsqueeze(1).float()
                shaped_rewards2 += self.gamma2 * res_dict2['values'] * self.cast_obs(infos['time_outs'].unsqueeze(1).repeat(1, self.num_agents2).transpose(0,1).reshape(-1)).unsqueeze(1).float()

            self.experience_buffer1.update_data('rewards', n, shaped_rewards1)
            self.experience_buffer2.update_data('rewards', n, shaped_rewards2)
            
            self.current_rewards1 += rewards[:, :self.num_agents1].transpose(0,1).reshape(-1).unsqueeze(1)
            self.current_rewards2 += rewards[:, self.num_agents1:].transpose(0,1).reshape(-1).unsqueeze(1)
            self.current_shaped_rewards1 += shaped_rewards1
            self.current_shaped_rewards2 += shaped_rewards2
            self.current_lengths1 += 1
            self.current_lengths2 += 1
            all_done_indices1 = self.dones.unsqueeze(1).repeat(1, self.num_agents1).transpose(0,1).reshape(-1).nonzero(as_tuple=False)
            all_done_indices2 = self.dones.unsqueeze(1).repeat(1, self.num_agents2).transpose(0,1).reshape(-1).nonzero(as_tuple=False)
            env_done_indices1 = all_done_indices1
            env_done_indices2 = all_done_indices2
     
            self.game_rewards1.update(self.current_rewards1[env_done_indices1])
            self.game_rewards2.update(self.current_rewards2[env_done_indices2])
            self.game_shaped_rewards1.update(self.current_shaped_rewards1[env_done_indices1])
            self.game_shaped_rewards2.update(self.current_shaped_rewards2[env_done_indices2])
     
            self.game_lengths1.update(self.current_lengths1[env_done_indices1])
            self.game_lengths2.update(self.current_lengths2[env_done_indices2])
            self.algo_observer1.process_infos(infos, env_done_indices1)
            self.algo_observer2.process_infos(infos, env_done_indices2)

            not_dones1 = 1.0 - self.dones.unsqueeze(1).repeat(1, self.num_agents1).transpose(0,1).reshape(-1).float()
            not_dones2 = 1.0 - self.dones.unsqueeze(1).repeat(1, self.num_agents2).transpose(0,1).reshape(-1).float()

            self.current_rewards1 = self.current_rewards1 * not_dones1.unsqueeze(1)
            self.current_shaped_rewards1 = self.current_shaped_rewards1 * not_dones1.unsqueeze(1)
            self.current_lengths1 = self.current_lengths1 * not_dones1
            self.current_rewards2 = self.current_rewards2 * not_dones2.unsqueeze(1)
            self.current_shaped_rewards2 = self.current_shaped_rewards2 * not_dones2.unsqueeze(1)
            self.current_lengths2 = self.current_lengths2 * not_dones2

        last_values1 = self.get_values1(self.obs)
        last_values2 = self.get_values2(self.obs)

        fdones1 = self.dones.unsqueeze(1).repeat(1, self.num_agents1).transpose(0,1).reshape(-1).float()
        fdones2 = self.dones.unsqueeze(1).repeat(1, self.num_agents2).transpose(0,1).reshape(-1).float()
        mb_fdones1 = self.experience_buffer1.tensor_dict['dones'].float()
        mb_values1 = self.experience_buffer1.tensor_dict['values']
        mb_rewards1 = self.experience_buffer1.tensor_dict['rewards']
        mb_advs1 = self.discount_values1(fdones1, last_values1, mb_fdones1, mb_values1, mb_rewards1)
        mb_returns1 = mb_advs1 + mb_values1
        mb_fdones2 = self.experience_buffer2.tensor_dict['dones'].float()
        mb_values2 = self.experience_buffer2.tensor_dict['values']
        mb_rewards2 = self.experience_buffer2.tensor_dict['rewards']
        mb_advs2 = self.discount_values2(fdones2, last_values2, mb_fdones2, mb_values2, mb_rewards2)
        mb_returns2 = mb_advs2 + mb_values2

        batch_dict1 = self.experience_buffer1.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict1['returns'] = swap_and_flatten01(mb_returns1)
        batch_dict1['played_frames'] = self.batch_size1
        batch_dict1['step_time'] = step_time
        batch_dict2 = self.experience_buffer2.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict2['returns'] = swap_and_flatten01(mb_returns2)
        batch_dict2['played_frames'] = self.batch_size2
        batch_dict2['step_time'] = step_time

        return batch_dict1, batch_dict2
