import gym
import numpy as np
import torch
import copy
from os.path import basename
from typing import Optional
import os
import shutil
import threading
import time

from rl_games.common import vecenv
from rl_games.common import env_configurations
from timechamber.learning.lib.core import torch_ext
from timechamber.learning.lib.utils.tr_helpers import unsqueeze_obs

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action


class BasePlayer(object):

    def __init__(self, params):
        self.config = config = params['config']
        self.env_name = self.config['env_name']
        self.player_config = self.config.get('player', {})
        self.env_config = self.config.get('env_config', {})
        self.env_config = self.player_config.get('env_config', self.env_config)
        self.env_info = self.config.get('env_info')
        self.clip_actions = config.get('clip_actions', True)
        self.seed = self.env_config.pop('seed', None)
        self.env = vecenv.create_vec_env(self.env_name, self.config['num_actors'], **self.env_config)
        self.env_info = self.env.get_env_info()
        
        self.num_agents = self.env.num_agents
        self.num_agents1 = self.env_info['num_agents'][0]
        self.num_agents2 = self.env_info['num_agents'][1]
        self.value_size = self.env_info.get('value_size', 1)
        self.action_space1 = self.env_info['action_space'][0]
        self.action_space2 = self.env_info['action_space'][1]

        self.observation_space = self.env_info['observation_space']
        self.observation_space1 = self.env_info['observation_space'][0]
        self.observation_space2 = self.env_info['observation_space'][1]
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k, v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape1 = self.observation_space1.shape
            self.obs_shape2 = self.observation_space2.shape
        self.is_tensor_obses = False

        self.states = None
        self.player_config = self.config.get('player', {})
        self.use_cuda = True
        self.batch_size = 1
        self.has_batch_dimension = False
        self.device_name = self.config.get('device_name', 'cuda')
        self.render_env = self.player_config.get('render', False)
        self.games_num = self.player_config.get('games_num', 2000)

        if 'deterministic' in self.player_config:
            self.is_deterministic = self.player_config['deterministic']
        else:
            self.is_deterministic = self.player_config.get('deterministic', True)

        self.n_game_life = self.player_config.get('n_game_life', 1)
        self.print_stats = self.player_config.get('print_stats', True)
        self.render_sleep = self.player_config.get('render_sleep', 0.002)
        self.max_steps = 108000 // 4
        self.device = torch.device(self.device_name)

    # def wait_for_checkpoint(self):
    #     attempt = 0
    #     while True:
    #         attempt += 1
    #         with self.checkpoint_mutex:
    #             if self.checkpoint_to_load is not None:
    #                 if attempt % 10 == 0:
    #                     print(f"Evaluation: waiting for new checkpoint in {self.dir_to_monitor}...")
    #                 break
    #         time.sleep(1.0)

    #     print(f"Checkpoint {self.checkpoint_to_load} is available!")

    # def process_new_eval_checkpoint(self, path):
    #     with self.checkpoint_mutex:
    #         # print(f"New checkpoint {path} available for evaluation")
    #         # copy file to eval_checkpoints dir using shutil
    #         # since we're running the evaluation worker in a separate process,
    #         # there is a chance that the file is changed/corrupted while we're copying it
    #         # not sure what we can do about this. In practice it never happened so far though
    #         try:
    #             eval_checkpoint_path = os.path.join(self.eval_checkpoint_dir, basename(path))
    #             shutil.copyfile(path, eval_checkpoint_path)
    #         except Exception as e:
    #             print(f"Failed to copy {path} to {eval_checkpoint_path}: {e}")
    #             return

    #         self.checkpoint_to_load = eval_checkpoint_path

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

    def env_step(self, env, action1, action2):
        if not self.is_tensor_obses:
            action1 = action1.cpu().numpy()
            action2 = action2.cpu().numpy()
        obs, rewards, dones, infos = env.step(action1, action2)
        # print(rewards)
        # rewards维度是二维，第一维度维数是num_envs,第二维度维数是num_agents，元素是这个step该环境该智能体获得的reward
        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return self.obs_to_torch(obs), rewards.cpu(), dones.cpu(), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def obs_to_torch(self, obs):
        if isinstance(obs, dict):
            if 'obs' in obs:
                obs = obs['obs']
            if isinstance(obs, dict):
                upd_obs = {}
                for key, value in obs.items():
                    upd_obs[key] = self._obs_to_tensors_internal(value, False)
            else:
                upd_obs = self.cast_obs(obs)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def _obs_to_tensors_internal(self, obs, cast_to_dict=True):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value, False)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert (obs.dtype != np.int8)
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.device)
            else:
                obs = torch.FloatTensor(obs).to(self.device)
        elif np.isscalar(obs):
            obs = torch.FloatTensor([obs]).to(self.device)
        return obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_reset(self, env):
        obs = env.reset()
        obs = {'obs1': obs["obs1"], 'obs2': obs["obs2"]}
        return self.obs_to_torch(obs)

    def restore(self, fn):
        raise NotImplementedError('restore')

    def get_weights(self):
        weights = {}
        weights['model1'] = self.model1.state_dict()
        weights['model2'] = self.model2.state_dict()
        return weights

    def set_weights(self, weights):
        self.model1.load_state_dict(weights['model1'])
        self.model2.load_state_dict(weights['model2'])

    def create_env(self):
        return env_configurations.configurations[self.env_name]['env_creator'](**self.env_config)

    def get_action1(self, obs, is_deterministic=False):
        raise NotImplementedError('step')
    
    def get_action2(self, obs, is_deterministic=False):
        raise NotImplementedError('step')

    def get_masked_action1(self, obs, mask, is_deterministic=False):
        raise NotImplementedError('step')
    
    def get_masked_action2(self, obs, mask, is_deterministic=False):
        raise NotImplementedError('step')

    def reset(self):
        raise NotImplementedError('raise')

    def run(self):
        print("==============看看你的？？？？？？？？？===========")
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards1 = 0
        sum_rewards2 = 0
        sum_steps1 = 0
        sum_steps2 = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            batch_size1 = 1
            batch_size2 = 1
            batch_size1, batch_size2 = self.get_batch_size(obses, batch_size1, batch_size2)
            # print(f"batch_size1:{batch_size1}, batch_size2:{batch_size2}")

            cr1 = torch.zeros(batch_size1, dtype=torch.float32)
            cr2 = torch.zeros(batch_size2, dtype=torch.float32)
            steps1 = torch.zeros(batch_size1, dtype=torch.float32)
            steps2 = torch.zeros(batch_size2, dtype=torch.float32)

            print_game_res = False

            for n in range(self.max_steps):
                if has_masks:                                                     # useless?
                    masks = self.env.get_action_mask()
                    action1 = self.get_masked_action1(
                        obses, masks, is_deterministic)
                    action2 = self.get_masked_action2(
                        obses, masks, is_deterministic)
                else:
                    action1 = self.get_action1(obses, is_deterministic)
                    action2 = self.get_action2(obses, is_deterministic)

                obses, r, done, info = self.env_step(self.env, action1, action2)
                r1 = r[:, :self.num_agents1].reshape(-1)
                # print(r1)
                r2 = r[:, self.num_agents1:].reshape(-1)
                # print("r1.shape",r1.shape)
                cr1 += r1
                cr2 += r2
                steps1 += 1
                steps2 += 1
                
                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                done1 = done.unsqueeze(1).repeat(1, self.num_agents1).view(-1)
                done2 = done.unsqueeze(1).repeat(1, self.num_agents2).view(-1)
                all_done_indices1 = done1.nonzero(as_tuple=False)
                all_done_indices2 = done2.nonzero(as_tuple=False)
                done_indices1 = all_done_indices1
                done_indices2 = all_done_indices2
                done_count1 = len(done_indices1)
                done_count2 = len(done_indices2)
                games_played += len(done.nonzero(as_tuple=False))

                if done_count1 > 0 or done_count2 > 0:
                    # print(done_indices1)
                    cur_rewards1 = cr1[done_indices1].sum().item()
                    cur_rewards2 = cr2[done_indices2].sum().item()
                    cur_steps1 = steps1[done_indices1].sum().item()
                    cur_steps2 = steps2[done_indices2].sum().item()

                    cr1 = cr1 * (1.0 - done1.float())
                    cr2 = cr2 * (1.0 - done2.float())
                    steps1 = steps1 * (1.0 - done1.float())
                    steps2 = steps2 * (1.0 - done2.float())
                    sum_rewards1 += cur_rewards1
                    sum_rewards2 += cur_rewards2
                    sum_steps1 += cur_steps1
                    sum_steps2 += cur_steps2

                    game_res = 0.0
                    if isinstance(info, dict):                          # useless?
                        if 'battle_won' in info:             
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if self.print_stats:
                        cur_rewards_done1 = cur_rewards1/done_count1
                        cur_rewards_done2 = cur_rewards2/done_count2
                        cur_steps_done1 = cur_steps1/done_count1
                        cur_steps_done2 = cur_steps2/done_count2
                        if print_game_res:
                            print(f'reward1: {cur_rewards_done1:.2f} steps1: {cur_steps_done1:.1f}, reward2: {cur_rewards_done2:.2f} steps2: {cur_steps_done2:.1f}, w: {game_res:.2f}')
                        else:
                            print(f'reward1: {cur_rewards_done1:.2f} steps1: {cur_steps_done1:.1f}, reward2: {cur_rewards_done2:.2f} steps2: {cur_steps_done2:.1f}')

                    sum_game_res += game_res
                    if batch_size1 //self.num_agents1 == 1 or games_played >= n_games:
                        break

        if print_game_res:
            print('av reward1:', sum_rewards1 / (games_played * self.num_agents1), 'av reward2:', sum_rewards2 / (games_played * self.num_agents2), 'av steps1:', sum_steps1 /
                  (games_played * self.num_agents1), 'av steps2:', sum_steps2 / (games_played * self.num_agents2), 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward1:', sum_rewards1 / (games_played * self.num_agents1), 'av reward2:', sum_rewards2 / (games_played * self.num_agents2), 'av steps1:', sum_steps1 /
                  (games_played * self.num_agents1), 'av steps2:', sum_steps2 / (games_played * self.num_agents2))
            
    def get_batch_size(self, obses, batch_size1, batch_size2):
        # obs_shape1 = self.obs_shape1
        # obs_shape2 = self.obs_shape2
        # if type(self.obs_shape) is dict:
        #     if 'obs' in obses:
        #         obses = obses['obs']
        #     keys_view = self.obs_shape.keys()
        #     keys_iterator = iter(keys_view)
        #     if 'observation' in obses:
        #         first_key = 'observation'
        #     else:
        #         first_key = next(keys_iterator)
        #     obs_shape = self.obs_shape[first_key]
        #     obses = obses[first_key]

        batch_size1 = obses['obs1'].size()[0]
        batch_size2 = obses['obs2'].size()[0]
        self.has_batch_dimension = True

        self.batch_size1 = batch_size1
        self.batch_size2 = batch_size2

        return batch_size1, batch_size2


class A2CPlayer(BasePlayer):

    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.actions_num1 = self.action_space1.shape[0]
        self.actions_num2 = self.action_space2.shape[0]
        self.actions_low1 = torch.from_numpy(self.action_space1.low.copy()).float().to(self.device)
        self.actions_high1 = torch.from_numpy(self.action_space1.high.copy()).float().to(self.device)
        self.actions_low2 = torch.from_numpy(self.action_space2.low.copy()).float().to(self.device)
        self.actions_high2 = torch.from_numpy(self.action_space2.high.copy()).float().to(self.device)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        from timechamber.learning.lib.model.a2c_continuous_logstd_model import ModelA2CContinuousLogStd
        keys1 = {
            'actions_num' : self.actions_num1,
            'input_shape' : self.obs_shape1,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.model1 = ModelA2CContinuousLogStd(params, keys1)
        keys2 = {
            'actions_num' : self.actions_num2,
            'input_shape' : self.obs_shape2,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.model2 = ModelA2CContinuousLogStd(params, keys2)
        self.model1.to(self.device)
        self.model1.eval()
        self.model2.to(self.device)
        self.model2.eval()

    def get_action1(self, obs, is_deterministic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs["obs1"],
        }
        with torch.no_grad():
            res_dict = self.model1(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())
        
        # print("mu value:", mu.detach().cpu().numpy())
        # print("selected action:", current_action.detach().cpu().numpy())

        if self.clip_actions:
            return rescale_actions(self.actions_low1, self.actions_high1, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action
        
    def get_action2(self, obs, is_deterministic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs["obs2"],
        }
        with torch.no_grad():
            res_dict = self.model2(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())
        
        # print("mu value:", mu.detach().cpu().numpy())
        # print("selected action:", current_action.detach().cpu().numpy())
        # print("||||||||||||||||||")

        # print("action space bounds - low:", self.actions_low2.cpu().numpy())
        # print("action space bounds - high:", self.actions_high2.cpu().numpy())
        # current_action = torch.tensor([0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1], device=self.device)
        
        
        if self.clip_actions:
            return rescale_actions(self.actions_low2, self.actions_high2, torch.clamp(current_action, -1.0, 1.0))
            # return rescale_actions(self.actions_low2, self.actions_high2, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    # def restore(self, fn):
    #     checkpoint = torch_ext.load_checkpoint(fn)
    #     print(checkpoint['model'].keys())
    #     # print(self.model)
    #     self.model.load_state_dict(checkpoint['model'])
    #     if self.normalize_input and 'running_mean_std' in checkpoint:
    #         self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    #     env_state = checkpoint.get('env_state', None)
    #     if self.env is not None and env_state is not None:
    #         self.env.set_env_state(env_state)

    def restore(self, fn):
        # checkpoint = torch_ext.load_checkpoint(fn)
        # self.set_full_state_weights(checkpoint)
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
        merged['model2'] = bug_ckpt['model2']
        self.set_full_state_weights(merged)

    def set_full_state_weights(self, checkpoint):
        weights = checkpoint
        print(weights['model1'].keys)
        print(weights['model2'].keys)
        try:
            self.model1.load_state_dict(weights['model1'])
            self.model2.load_state_dict(weights['model2'])
            print("||||||||||||||sigma值||||||||||||||||")
            print("Model 1 logstd after loading:", self.model1.logstd.detach().cpu().numpy())
            print("Model 1 sigma after loading:", torch.exp(self.model1.logstd).detach().cpu().numpy())
            print("Model 2 sigma after loading:", torch.exp(self.model2.logstd).detach().cpu().numpy())
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
            

    def reset(self):
        pass

