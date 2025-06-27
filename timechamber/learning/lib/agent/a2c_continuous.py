from timechamber.learning.lib.core import torch_ext
from timechamber.learning.lib.core import common_losses
from timechamber.learning.lib.core import datasets
from timechamber.learning.lib.agent.a2c_base import A2CBase

import os
import torch
import numpy as np
import time
from torch import optim
import torch.distributed as dist

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action

def print_statistics(print_stats, curr_frames, step_time, step_inference_time, total_time, epoch_num, max_epochs, frame, max_frames):
    if print_stats:
        step_time = max(step_time, 1e-9)
        fps_step = curr_frames / step_time
        fps_step_inference = curr_frames / step_inference_time
        fps_total = curr_frames / total_time

        if max_epochs == -1 and max_frames == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}')
        elif max_epochs == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}/{max_frames:.0f}')
        elif max_frames == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}')
        else:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}/{max_frames:.0f}')


class ContinuousA2CBase(A2CBase):

    def __init__(self, base_name, params):
        A2CBase.__init__(self, base_name, params)
        self.is_discrete = False
        action_space = self.env_info['action_space']
        self.actions_num1 = action_space[0].shape[0]
        self.actions_num2 = action_space[1].shape[0]
        self.bounds_loss_coef1 = self.config.get('bounds_loss_coef', None)
        self.bounds_loss_coef2 = self.config.get('bounds_loss_coef', None)
        self.clip_actions = self.config.get('clip_actions', True)

        # todo introduce device instead of cuda()
        self.actions_low1 = torch.from_numpy(action_space[0].low.copy()).float().to(self.ppo_device)
        self.actions_high1 = torch.from_numpy(action_space[0].high.copy()).float().to(self.ppo_device)
        self.actions_low2 = torch.from_numpy(action_space[1].low.copy()).float().to(self.ppo_device)
        self.actions_high2 = torch.from_numpy(action_space[1].high.copy()).float().to(self.ppo_device)

        from timechamber.learning.lib.model.a2c_continuous_logstd_model import ModelA2CContinuousLogStd
        from timechamber.learning.lib.mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
        keys1 = {
            'actions_num' : self.actions_num1,
            'input_shape' : self.obs_shape1,
            'num_seqs' : self.num_actors * self.num_agents1,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        # self.model1 = ModelA2CContinuousLogStd(params, keys1)
        keys2 = {
            'actions_num' : self.actions_num2,
            'input_shape' : self.obs_shape2,
            'num_seqs' : self.num_actors * self.num_agents2,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        # self.model2 = ModelA2CContinuousLogStd(params, keys2)
        # 使用mat替换原mlp
        self.model1 = MultiAgentTransformer(self.obs_shape1[0], self.obs_shape1[0], self.actions_num1, self.num_agents1, 1, 64, 1, self.obs_shape1, False, torch.device("cuda:0"), 'Continuous', False, False, True, True) # 第一个参数state_dim无所谓，后续mat没用到
        self.model2 = MultiAgentTransformer(self.obs_shape2[0], self.obs_shape2[0], self.actions_num2, self.num_agents2, 1, 64, 1, self.obs_shape2, False, torch.device("cuda:0"), 'Continuous', False, False, True, True)
        print("obs_shape1", self.obs_shape1)
        print("obs_shape2", self.obs_shape2)

    def preprocess_actions1(self, actions):
        if self.clip_actions:
            clamped_actions = torch.clamp(actions, -1.0, 1.0)
            rescaled_actions = rescale_actions(self.actions_low1, self.actions_high1, clamped_actions)
        else:
            rescaled_actions = actions

        if not self.is_tensor_obses:
            rescaled_actions = rescaled_actions.cpu().numpy()

        return rescaled_actions
    
    def preprocess_actions2(self, actions):
        if self.clip_actions:
            clamped_actions = torch.clamp(actions, -1.0, 1.0)
            rescaled_actions = rescale_actions(self.actions_low2, self.actions_high2, clamped_actions)
        else:
            rescaled_actions = actions

        if not self.is_tensor_obses:
            rescaled_actions = rescaled_actions.cpu().numpy()

        return rescaled_actions
    
    def init_tensors(self):
        A2CBase.init_tensors(self)
        self.update_list = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']
        self.tensor_list = self.update_list + ['obses', 'states', 'dones', 'outs']

    def train_epoch(self):
        # super().train_epoch()   # useless?

        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            batch_dict1, batch_dict2 = self.play_steps()
        play_time_end = time.time()
        update_time_start = time.time()

        self.set_train()
        self.curr_frames1 = batch_dict1.pop('played_frames')
        self.curr_frames2 = batch_dict2.pop('played_frames')
        self.curr_frames = self.curr_frames1 + self.curr_frames2
        # print("batch_dict1['actions'].shape", batch_dict1['actions'].shape)
        self.prepare_dataset1(batch_dict1)
        self.prepare_dataset2(batch_dict2)
        self.algo_observer1.after_steps()
        self.algo_observer2.after_steps()

        a_losses1 = []
        c_losses1 = []
        b_losses1 = []
        entropies1 = []
        kls1 = []
        a_losses2 = []
        c_losses2 = []
        b_losses2 = []
        entropies2 = []
        kls2 = []
        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls1 = []
            for i in range(len(self.dataset1)):
                # for key in self.dataset1.values_dict.keys():
                #     print(f"key {key}:", self.dataset1.values_dict[key][1])
                a_loss1, c_loss1, entropy1, kl1, last_lr1, lr_mul1, cmu1, csigma1, b_loss1 = self.train_actor_critic1(self.dataset1[i])
                a_losses1.append(a_loss1)
                c_losses1.append(c_loss1)
                ep_kls1.append(kl1)
                entropies1.append(entropy1)
                if self.bounds_loss_coef1 is not None:
                    b_losses1.append(b_loss1)

                self.dataset1.update_mu_sigma(cmu1, csigma1)
                if self.schedule_type == 'legacy':
                    av_kls1 = kl1
                    if self.multi_gpu:
                        dist.all_reduce(kl1, op=dist.ReduceOp.SUM)
                        av_kls1 /= self.world_size
                    self.last_lr1, self.entropy_coef1 = self.scheduler1.update(self.last_lr1, self.entropy_coef1, self.epoch_num, 0, av_kls1.item())
                    self.update_lr1(self.last_lr1)

            av_kls1 = torch_ext.mean_list(ep_kls1)
            if self.multi_gpu:
                dist.all_reduce(av_kls1, op=dist.ReduceOp.SUM)
                av_kls1 /= self.world_size
            if self.schedule_type == 'standard':
                self.last_lr1, self.entropy_coef1 = self.scheduler1.update(self.last_lr1, self.entropy_coef1, self.epoch_num, 0, av_kls1.item())
                self.update_lr1(self.last_lr1)

            kls1.append(av_kls1)
            self.diagnostics1.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model1.running_mean_std.eval() # don't need to update statstics more than one miniepoch
                
        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls2 = []
            for i in range(len(self.dataset2)):
                a_loss2, c_loss2, entropy2, kl2, last_lr2, lr_mul2, cmu2, csigma2, b_loss2 = self.train_actor_critic2(self.dataset2[i])
                a_losses2.append(a_loss2)
                c_losses2.append(c_loss2)
                ep_kls2.append(kl2)
                entropies2.append(entropy2)
                if self.bounds_loss_coef2 is not None:
                    b_losses2.append(b_loss2)

                self.dataset2.update_mu_sigma(cmu2, csigma2)
                if self.schedule_type == 'legacy':
                    av_kls2 = kl2
                    if self.multi_gpu:
                        dist.all_reduce(kl2, op=dist.ReduceOp.SUM)
                        av_kls2 /= self.world_size
                    self.last_lr2, self.entropy_coef2 = self.scheduler2.update(self.last_lr2, self.entropy_coef2, self.epoch_num, 0, av_kls2.item())
                    self.update_lr2(self.last_lr2)

            av_kls2 = torch_ext.mean_list(ep_kls2)
            if self.multi_gpu:
                dist.all_reduce(av_kls2, op=dist.ReduceOp.SUM)
                av_kls2 /= self.world_size
            if self.schedule_type == 'standard':
                self.last_lr2, self.entropy_coef2 = self.scheduler2.update(self.last_lr2, self.entropy_coef2, self.epoch_num, 0, av_kls2.item())
                self.update_lr2(self.last_lr2)

            kls2.append(av_kls2)
            self.diagnostics2.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model2.running_mean_std.eval() # don't need to update statstics more than one miniepoch
                
        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        # breakpoint()

        return batch_dict1['step_time'], play_time, update_time, total_time, a_losses1, c_losses1, b_losses1, entropies1, kls1, last_lr1, lr_mul1, a_losses2, c_losses2, b_losses2, entropies2, kls2, last_lr2, lr_mul2 # step_time in two batch_dict is the same

    def prepare_dataset1(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        outs = batch_dict['outs']

        advantages = returns - values

        if self.normalize_value:
            self.value_mean_std1.train()
            values = self.value_mean_std1(values)
            returns = self.value_mean_std1(returns)
            self.value_mean_std1.eval()

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.normalize_rms_advantage:
                advantages = self.advantage_mean_std1(advantages)
            else:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['dones'] = dones
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas
        dataset_dict['outs'] = outs

        self.dataset1.update_values_dict(dataset_dict)
        self.dataset1.clear_out_infos()
        
    def prepare_dataset2(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        outs = batch_dict['outs']

        advantages = returns - values

        if self.normalize_value:
            self.value_mean_std2.train()
            values = self.value_mean_std2(values)
            returns = self.value_mean_std2(returns)
            self.value_mean_std2.eval()

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.normalize_rms_advantage:
                advantages = self.advantage_mean_std2(advantages)
            else:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['dones'] = dones
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas
        dataset_dict['outs'] = outs

        self.dataset2.update_values_dict(dataset_dict)
        self.dataset2.clear_out_infos()

    def train(self):
        self.init_tensors()
        self.last_mean_rewards1 = -100500
        self.last_mean_rewards2 = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:                                                      # useless
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses1, c_losses1, b_losses1, entropies1, kls1, last_lr1, lr_mul1, a_losses2, c_losses2, b_losses2, entropies2, kls2, last_lr2, lr_mul2 = self.train_epoch()
            total_time += sum_time
            frame1 = self.frame1 // self.num_agents1
            frame2 = self.frame2 // self.num_agents2
            frame = frame1 + frame2

            # cleaning memory to optimize space
            self.dataset1.update_values_dict(None)
            self.dataset2.update_values_dict(None)
            should_exit = False

            if self.global_rank == 0:
                self.diagnostics1.epoch(self, current_epoch = epoch_num)
                self.diagnostics2.epoch(self, current_epoch = epoch_num)
                # do we need scaled_time?
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = self.curr_frames * self.world_size if self.multi_gpu else self.curr_frames
                curr_frames1 = self.curr_frames1 * self.world_size if self.multi_gpu else self.curr_frames1
                curr_frames2 = self.curr_frames2 * self.world_size if self.multi_gpu else self.curr_frames2
                self.frame += curr_frames
                self.frame1 += self.curr_frames1 * self.world_size if self.multi_gpu else self.curr_frames1
                self.frame2 += self.curr_frames2 * self.world_size if self.multi_gpu else self.curr_frames2

                print_statistics(self.print_stats, curr_frames, step_time, scaled_play_time, scaled_time, 
                                epoch_num, self.max_epochs, frame, self.max_frames)

                self.write_stats1(total_time, epoch_num, step_time, play_time, update_time,
                                a_losses1, c_losses1, entropies1, kls1, last_lr1, lr_mul1, frame1,
                                scaled_time, scaled_play_time, curr_frames1)
                self.write_stats2(total_time, epoch_num, step_time, play_time, update_time,
                                a_losses2, c_losses2, entropies2, kls2, last_lr2, lr_mul2, frame2,
                                scaled_time, scaled_play_time, curr_frames2)

                if len(b_losses1) > 0:
                    self.writer1.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses1).item(), frame1)

                if self.has_soft_aug:
                    self.writer1.add_scalar('losses/aug_loss', np.mean(aug_losses1), frame1)
                    
                if len(b_losses2) > 0:
                    self.writer2.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses2).item(), frame2)

                if self.has_soft_aug:
                    self.writer2.add_scalar('losses/aug_loss', np.mean(aug_losses2), frame2)

                if self.game_rewards1.current_size > 0:
                    mean_rewards1 = self.game_rewards1.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards1.get_mean()
                    mean_lengths = self.game_lengths1.get_mean()
                    self.mean_rewards1 = mean_rewards1[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer1.add_scalar(rewards_name + '/step'.format(i), mean_rewards1[i], frame1)
                        self.writer1.add_scalar(rewards_name + '/iter'.format(i), mean_rewards1[i], epoch_num)
                        self.writer1.add_scalar(rewards_name + '/time'.format(i), mean_rewards1[i], total_time)
                        self.writer1.add_scalar('shaped_' + rewards_name + '/step'.format(i), mean_shaped_rewards[i], frame1)
                        self.writer1.add_scalar('shaped_' + rewards_name + '/iter'.format(i), mean_shaped_rewards[i], epoch_num)
                        self.writer1.add_scalar('shaped_' + rewards_name + '/time'.format(i), mean_shaped_rewards[i], total_time)

                    self.writer1.add_scalar('episode_lengths/step', mean_lengths, frame1)
                    self.writer1.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer1.add_scalar('episode_lengths/time', mean_lengths, total_time)

                if self.game_rewards2.current_size > 0:
                    mean_rewards2 = self.game_rewards2.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards2.get_mean()
                    mean_lengths = self.game_lengths2.get_mean()
                    self.mean_rewards2 = mean_rewards2[0]
                    
                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer2.add_scalar(rewards_name + '/step'.format(i), mean_rewards2[i], frame2)
                        self.writer2.add_scalar(rewards_name + '/iter'.format(i), mean_rewards2[i], epoch_num)
                        self.writer2.add_scalar(rewards_name + '/time'.format(i), mean_rewards2[i], total_time)
                        self.writer2.add_scalar('shaped_' + rewards_name + '/step'.format(i), mean_shaped_rewards[i], frame2)
                        self.writer2.add_scalar('shaped_' + rewards_name + '/iter'.format(i), mean_shaped_rewards[i], epoch_num)
                        self.writer2.add_scalar('shaped_' + rewards_name + '/time'.format(i), mean_shaped_rewards[i], total_time)

                    self.writer2.add_scalar('episode_lengths/step', mean_lengths, frame2)
                    self.writer2.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer2.add_scalar('episode_lengths/time', mean_lengths, total_time)
                    

                    print('test rewards: ', mean_rewards1, mean_rewards2)
                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards1[0] + mean_rewards2)

                    if self.save_freq > 0:
                        if epoch_num % self.save_freq == 0:
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_rewards1[0] + mean_rewards2[0] > self.last_mean_rewards1 + self.last_mean_rewards2 and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards1, mean_rewards2)
                        self.last_mean_rewards1 = mean_rewards1[0]
                        self.last_mean_rewards2 = mean_rewards2[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))

                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards1 + self.last_mean_rewards2 > self.config['score_to_win']:
                                print('Maximum reward achieved. Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True

                if epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards1.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(epoch_num) \
                        + '_rew1_' + str(mean_rewards1[0]) + '_rew2_' + str(mean_rewards2[0]).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                if self.frame >= self.max_frames and self.max_frames != -1:
                    if self.game_rewards1.current_size == 0:
                        print('WARNING: Max frames reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                        + '_rew1_' + str(mean_rewards1[0]) + '_rew2_' + str(mean_rewards2[0]).replace('[', '_').replace(']', '_')))
                    print('MAX FRAMES NUM!')
                    should_exit = True

                update_time = 0

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards1, self.last_mean_rewards2, epoch_num
    
    def get_masked_action_values(self, obs, action_masks):
        assert False

    def calc_gradients1(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }

        # print all keys in batch_dict1
        # for key in batch_dict1:
        #     if key != 'is_train':
        #         print(f"key {key}:", batch_dict1[key].shape)
        
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            
            # res_dict = self.model1(batch_dict)                  #
            
            reshaped_actions_batch = actions_batch.reshape(-1, self.num_agents1, self.actions_num1)
            reshaped_obs = obs_batch.reshape(-1, self.num_agents1, self.obs_shape1[0]) # 将obs变成mat使用的shape
            res_dict = self.model1(reshaped_obs, reshaped_obs, reshaped_actions_batch, None)  # 使用新模型计算相同动作在新策略下的概率

            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func1(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)              #

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(self.model1, value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)     #
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss , entropy.unsqueeze(1), b_loss.unsqueeze(1)])
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = a_loss + 0.5 * c_loss * self.critic_coef1 - entropy * self.entropy_coef1 + b_loss * self.bounds_loss_coef1
            
            if self.multi_gpu:
                self.optimizer1.zero_grad()
            else:
                for param in self.model1.parameters():
                    param.grad = None

        self.scaler1.scale(loss).backward()                                                     #
        #TODO: Refactor this ugliest code of they year
        self.trancate_gradients_and_step1()                                                         #

        with torch.no_grad():
            reduce_kl = True
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)

        self.diagnostics1.mini_batch(self,                      #
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : None,
        }, curr_e_clip, 0)
        

        self.train_result1 = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr1, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)
        
    def calc_gradients2(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }
        
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            
            # res_dict = self.model2(batch_dict)
            reshaped_actions_batch = actions_batch.reshape(-1, self.num_agents2, self.actions_num2)
            reshaped_obs = obs_batch.reshape(-1, self.num_agents2, self.obs_shape2[0]) # 将obs变成mat使用的shape
            res_dict = self.model2(reshaped_obs, reshaped_obs, reshaped_actions_batch, None) # 使用新模型计算相同动作在新策略下的概率

            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func2(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(self.model2, value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss , entropy.unsqueeze(1), b_loss.unsqueeze(1)])
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = a_loss + 0.5 * c_loss * self.critic_coef2 - entropy * self.entropy_coef2 + b_loss * self.bounds_loss_coef2
            
            if self.multi_gpu:
                self.optimizer2.zero_grad()
            else:
                for param in self.model2.parameters():
                    param.grad = None

        self.scaler2.scale(loss).backward()                                                     #
        #TODO: Refactor this ugliest code of they year
        self.trancate_gradients_and_step2()                                                         #

        with torch.no_grad():
            reduce_kl = True
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)

        self.diagnostics2.mini_batch(self,                      #
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : None,
        }, curr_e_clip, 0)
        

        self.train_result2 = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr2, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)        
        
    def train_actor_critic1(self, input_dict):
        self.calc_gradients1(input_dict)
        return self.train_result1
    
    def train_actor_critic2(self, input_dict):
        self.calc_gradients2(input_dict)
        return self.train_result2

    def reg_loss(self, mu):
        if self.bounds_loss_coef1 is not None and self.bounds_loss_coef2 is not None:
            reg_loss = (mu*mu).sum(axis=-1)
        else:
            reg_loss = 0
        return reg_loss

    def bound_loss(self, mu):
        if self.bounds_loss_coef1 is not None and self.bounds_loss_coef2 is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss


class A2CAgent(ContinuousA2CBase):

    def __init__(self, base_name, params):
        # breakpoint()
        ContinuousA2CBase.__init__(self, base_name, params)        
        self.model1.to(self.ppo_device)
        self.model2.to(self.ppo_device)
        self.states = None
        self.last_lr1 = float(self.last_lr1)
        self.last_lr2 = float(self.last_lr2)
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound') # 'regularisation' or 'bound'
        self.optimizer1 = optim.Adam(self.model1.parameters(), float(self.last_lr1), eps=1e-08, weight_decay=self.weight_decay)
        self.optimizer2 = optim.Adam(self.model2.parameters(), float(self.last_lr2), eps=1e-08, weight_decay=self.weight_decay)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset1 = datasets.PPODataset(self.batch_size1, self.minibatch_size, self.is_discrete, self.ppo_device)
        self.dataset2 = datasets.PPODataset(self.batch_size2, self.minibatch_size, self.is_discrete, self.ppo_device)
        if self.normalize_value:
            self.value_mean_std1 = self.model1.value_mean_std
            self.value_mean_std2 = self.model2.value_mean_std

        self.has_value_loss = self.use_experimental_cv
        self.algo_observer1.after_init(self, 0)
        self.algo_observer2.after_init(self, 1)

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num
        
    def save(self, fn):
        # state = self.get_full_state_weights()
        # torch_ext.save_checkpoint(fn, state)
        import os
        os.makedirs(fn, exist_ok=True)
        # ant
        ant_state = self.get_full_state_weights()
        # 只保留ant相关内容
        ant_only = {k: v for k, v in ant_state.items() if not k.startswith('model2') and not k.startswith('optimizer2') and not k.startswith('last_mean_rewards2')}
        ant_only['model1'] = ant_state['model1']
        ant_only['optimizer1'] = ant_state['optimizer1']
        ant_only['last_mean_rewards1'] = ant_state['last_mean_rewards1']
        ant_only['epoch'] = ant_state['epoch']
        ant_only['frame1'] = ant_state['frame1']
        ant_only['frame'] = ant_state['frame']
        if 'env_state' in ant_state:
            ant_only['env_state'] = ant_state['env_state']
        from timechamber.learning.lib.core import torch_ext
        torch_ext.save_checkpoint(os.path.join(fn, 'ant'), ant_only)
        # bug
        bug_only = {k: v for k, v in ant_state.items() if not k.startswith('model1') and not k.startswith('optimizer1') and not k.startswith('last_mean_rewards1')}
        bug_only['model2'] = ant_state['model2']
        bug_only['optimizer2'] = ant_state['optimizer2']
        bug_only['last_mean_rewards2'] = ant_state['last_mean_rewards2']
        bug_only['epoch'] = ant_state['epoch']
        bug_only['frame2'] = ant_state['frame2']
        bug_only['frame'] = ant_state['frame']
        if 'env_state' in ant_state:
            bug_only['env_state'] = ant_state['env_state']
        torch_ext.save_checkpoint(os.path.join(fn, 'bug'), bug_only)

    def restore(self, fn, set_epoch=True):                                             # TODO
        # checkpoint = torch_ext.load_checkpoint(fn)
        # self.set_full_state_weights(checkpoint, set_epoch=set_epoch)
        """
        分别从指定文件夹下的ant.pth和bug.pth加载ant和bug的参数。
        """
        import os
        from timechamber.learning.lib.core import torch_ext
        ant_ckpt = torch_ext.load_checkpoint(os.path.join(fn, 'ant.pth'))
        bug_ckpt = torch_ext.load_checkpoint(os.path.join(fn, 'bug.pth'))
        # 合并为一个dict，兼容set_full_state_weights
        merged = {}
        merged['model1'] = ant_ckpt['model1']
        merged['optimizer1'] = ant_ckpt['optimizer1']
        merged['last_mean_rewards1'] = ant_ckpt.get('last_mean_rewards1', -1000000000)
        merged['frame1'] = ant_ckpt.get('frame1', 0)
        merged['model2'] = bug_ckpt['model2']
        merged['optimizer2'] = bug_ckpt['optimizer2']
        merged['last_mean_rewards2'] = bug_ckpt.get('last_mean_rewards2', -1000000000)
        merged['frame2'] = bug_ckpt.get('frame2', 0)
        # epoch/frame/env_state 取ant的
        merged['epoch'] = ant_ckpt.get('epoch', 0)
        merged['frame'] = ant_ckpt.get('frame', 0)
        if 'env_state' in ant_ckpt:
            merged['env_state'] = ant_ckpt['env_state']
        elif 'env_state' in bug_ckpt:
            merged['env_state'] = bug_ckpt['env_state']
        self.set_full_state_weights(merged, set_epoch=set_epoch)

    def set_full_state_weights(self, checkpoint, set_epoch=True):
        weights = checkpoint
        print(weights['model1'].keys)
        print(weights['model2'].keys)
        try:
            self.model1.load_state_dict(weights['model1'])
            self.model2.load_state_dict(weights['model2'])
        except:
            """
            Load pretrained mlp.
            ['logstd', 
            'value_mean_std.running_mean', 
            'value_mean_std.running_var', 
            'value_mean_std.count', 
            'running_mean_std.running_mean', 
            'running_mean_std.running_var', 
            'running_mean_std.count', 
            'actor_mlp.layers.0.weight', 
            'actor_mlp.layers.0.bias', 
            'actor_mlp.layers.1.weight', 
            'actor_mlp.layers.1.bias', 
            'actor_mlp.layers.2.weight', 
            'actor_mlp.layers.2.bias', 
            'mu.weight', 
            'mu.bias', 
            'value_head.weight', 
            'value_head.bias']
            """
            print("Missing CNN part. Loading Pretrained MLP Model......")
            with torch.no_grad():
                self.model1.logstd.copy_(weights['model1']['logstd'])

            running_mean_std_state_dict1 = {'running_mean': weights['model1']['running_mean_std.running_mean'], 
                                           'running_var': weights['model1']['running_mean_std.running_var'], 
                                           'count': weights['model1']['running_mean_std.count']}
            self.model1.running_mean_std.running_mean_std["observation"].load_state_dict(running_mean_std_state_dict1)

            value_mean_std_state_dict1 = {'running_mean': weights['model1']['value_mean_std.running_mean'], 
                                         'running_var': weights['model1']['value_mean_std.running_var'], 
                                         'count': weights['model1']['value_mean_std.count']}
            self.model1.value_mean_std.load_state_dict(value_mean_std_state_dict1)

            mlp_keys1 = [key for key in weights['model1'].keys() if 'actor_mlp' in key]
            mlp_state_dict1 = {key: weights['model1'][key] for key in mlp_keys1}
            self.model1.actor_mlp.load_state_dict(mlp_state_dict1, strict=False)

            mu_state_dict1 = {'weight': weights['model1']['mu.weight'], 'bias': weights['model1']['mu.bias']}
            self.model1.mu.load_state_dict(mu_state_dict1)
            
            value_state_dict1 = {'weight': weights['model1']['value_head.weight'], 'bias': weights['model1']['value_head.bias']}
            self.model1.value_head.load_state_dict(value_state_dict1)

            # Model 2
            with torch.no_grad():
                self.model2.logstd.copy_(weights['model2']['logstd'])
            
            running_mean_std_state_dict2 = {'running_mean': weights['model2']['running_mean_std.running_mean'], 
                                           'running_var': weights['model2']['running_mean_std.running_var'], 
                                           'count': weights['model2']['running_mean_std.count']}
            self.model2.running_mean_std.running_mean_std["observation"].load_state_dict(running_mean_std_state_dict2)

            value_mean_std_state_dict2 = {'running_mean': weights['model2']['value_mean_std.running_mean'], 
                                         'running_var': weights['model2']['value_mean_std.running_var'], 
                                         'count': weights['model2']['value_mean_std.count']}
            self.model2.value_mean_std.load_state_dict(value_mean_std_state_dict2)

            mlp_keys2 = [key for key in weights['model2'].keys() if 'actor_mlp' in key]
            mlp_state_dict2 = {key: weights['model2'][key] for key in mlp_keys2}
            self.model2.actor_mlp.load_state_dict(mlp_state_dict2, strict=False)

            mu_state_dict2 = {'weight': weights['model2']['mu.weight'], 'bias': weights['model2']['mu.bias']}
            self.model2.mu.load_state_dict(mu_state_dict2)
            
            value_state_dict2 = {'weight': weights['model2']['value_head.weight'], 'bias': weights['model2']['value_head.bias']}
            self.model2.value_head.load_state_dict(value_state_dict2)


    

