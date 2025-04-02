from typing import Tuple
import os

import torch
from isaacgym import gymtorch
from isaacgym.gymtorch import *
import traceback

from timechamber.utils.torch_jit_utils import *
from .base.va_vec_task import VA_VecTask


class MA_Ant_Bug_Battle(VA_VecTask):

    def __init__(self, cfg, sim_device, rl_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        
        self.extras = None
        self.cfg = cfg
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.termination_height = self.cfg["env"]["terminationHeight"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.ant_agents_state = []
        self.bug_agents_state = []
        self.win_reward_scale = 2000
        self.move_to_op_reward_scale = 1.
        self.stay_in_center_reward_scale = 0.2
        self.action_cost_scale = -0.000025
        self.push_scale = 1.
        self.dense_reward_scale = 1.0
        self.hp_decay_scale = 1.
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        self.cfg["env"]["numObservations1"] = 32 + 27 * (self.cfg["env"].get("numAgents1", 1) - 1) + 35 * (self.cfg["env"].get("numAgents2", 1))  # 1 for ant, 2 for bug
        self.cfg["env"]["numObservations2"] = 40 + 27 * (self.cfg["env"].get("numAgents1", 1)) + 35 * (self.cfg["env"].get("numAgents2", 1) - 1)  # 1 for ant, 2 for bug
        self.cfg["env"]["numActions1"] = 8
        self.cfg["env"]["numActions2"] = 12
        self.ant_dof_cnt = 8
        self.bug_dof_cnt = 12
        self.num_dof = self.ant_dof_cnt + self.bug_dof_cnt
        self.borderline_space = cfg["env"]["borderlineSpace"]
        self.borderline_space_unit = self.borderline_space / self.max_episode_length
        self.ant_body_colors = [gymapi.Vec3(*rgb_arr) for rgb_arr in self.cfg["env"]["color"]]
        self.bug_body_colors = [gymapi.Vec3(*rgb_arr) for rgb_arr in self.cfg["env"]["color"]]
        super().__init__(config=self.cfg, sim_device=sim_device, rl_device=rl_device,
                         graphics_device_id=graphics_device_id,
                         headless=headless, virtual_screen_capture=virtual_screen_capture,
                         force_render=force_render)

        self.use_central_value = False
        self.obs_idxs = torch.eye(4, dtype=torch.float32, device=self.device)
        if self.viewer is not None:
            for i, env in enumerate(self.envs):
                self._add_circle_borderline(env, self.borderline_space)
            cam_pos = gymapi.Vec3(15.0, 0.0, 3.4)
            cam_target = gymapi.Vec3(10.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        print(f'actor_root_state:{actor_root_state.shape}')
        print(f'dof_state_tensor:{dof_state_tensor.shape}')
        print(f'sensor_tensor:{sensor_tensor.shape}')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        ant_sensors_per_env = 4
        bug_sensors_per_env = 6
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs,
                                                                          (ant_sensors_per_env * self.num_agents1 + bug_sensors_per_env * self.num_agents2) * 6)
        print(f'vec_sensor_tensor:{self.vec_sensor_tensor.shape}')

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        print(f'root_states:{self.root_states.shape}')
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0  # set lin_vel and ang_vel to 0

        # create some wrapper tensors for different slices
        total_dof_cnt = self.num_agents1 * 8 + self.num_agents2 * 12
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, total_dof_cnt, 2)
        print(f'dof:{self.dof_state.shape}')
        dof_state_shaped = self.dof_state.view(self.num_envs, -1, 2)
        for idx in range(self.num_agents1):
            ant_root_state = self.root_states[idx::self.num_agents]
            ant_dof_pos = dof_state_shaped[:, idx * self.ant_dof_cnt:(idx + 1) * self.ant_dof_cnt, 0]
            ant_dof_vel = dof_state_shaped[:, idx * self.ant_dof_cnt:(idx + 1) * self.ant_dof_cnt, 1]
            self.ant_agents_state.append((ant_root_state, ant_dof_pos, ant_dof_vel))
        for idx in range(self.num_agents1, self.num_agents):
            bug_root_state = self.root_states[idx::self.num_agents]
            bug_dof_pos = dof_state_shaped[:, (self.num_agents1 * self.ant_dof_cnt + (idx - self.num_agents1) * self.bug_dof_cnt):(self.num_agents1 * self.ant_dof_cnt + (idx - self.num_agents1 + 1) * self.bug_dof_cnt), 0]
            bug_dof_vel = dof_state_shaped[:, (self.num_agents1 * self.ant_dof_cnt + (idx - self.num_agents1) * self.bug_dof_cnt):(self.num_agents1 * self.ant_dof_cnt + (idx - self.num_agents1 + 1) * self.bug_dof_cnt), 1]
            self.bug_agents_state.append((bug_root_state, bug_dof_pos, bug_dof_vel))
            
        self.ant_initial_dof_pos = torch.zeros_like(self.ant_agents_state[0][1], device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.ant_initial_dof_pos = torch.where(self.ant_dof_limits_lower > zero_tensor, self.ant_dof_limits_lower,
                                           torch.where(self.ant_dof_limits_upper < zero_tensor, self.ant_dof_limits_upper,
                                                       self.ant_initial_dof_pos))
        self.ant_initial_dof_vel = torch.zeros_like(self.ant_agents_state[0][2], device=self.device, dtype=torch.float)
        self.bug_initial_dof_pos = torch.zeros_like(self.bug_agents_state[0][1], device=self.device, dtype=torch.float)
        self.bug_initial_dof_pos = torch.where(self.bug_dof_limits_lower > zero_tensor, self.bug_dof_limits_lower,
                                           torch.where(self.bug_dof_limits_upper < zero_tensor, self.bug_dof_limits_upper,
                                                       self.bug_initial_dof_pos))
        self.bug_initial_dof_vel = torch.zeros_like(self.bug_agents_state[0][2], device=self.device, dtype=torch.float)
        self.dt = self.cfg["sim"]["dt"]

        torques = self.gym.acquire_dof_force_tensor(self.sim)
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_agents1 * self.ant_dof_cnt + self.num_agents2 * self.bug_dof_cnt)

        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat(
            (self.num_agents * self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat(
            (self.num_agents * self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat(
            (self.num_agents * self.num_envs, 1))

    def allocate_buffers(self):
        self.obs_buf1 = torch.zeros((self.num_agents1 * self.num_envs, self.num_observations1), device=self.device,
                                   dtype=torch.float)
        self.obs_buf2 = torch.zeros((self.num_agents2 * self.num_envs, self.num_observations2), device=self.device,
                                   dtype=torch.float)                               
        self.rew_buf = torch.zeros(
            (self.num_envs, self.num_agents), device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs * self.num_agents, device=self.device, dtype=torch.long)
        self.extras = {'ranks': torch.zeros((self.num_envs, self.num_agents), device=self.device, dtype=torch.long),
                       'win': torch.zeros((self.num_envs * self.num_agents,), device=self.device,
                                          dtype=torch.bool),
                       'lose': torch.zeros((self.num_envs * self.num_agents,), device=self.device,
                                           dtype=torch.bool),
                       'draw': torch.zeros((self.num_envs * self.num_agents,), device=self.device,
                                           dtype=torch.bool)}

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        lines = []
        borderline_height = 0.01
        for height in range(20):
            for angle in range(360):
                begin_point = [np.cos(np.radians(angle)), np.sin(np.radians(angle)), borderline_height * height]
                end_point = [np.cos(np.radians(angle + 1)), np.sin(np.radians(angle + 1)), borderline_height * height]
                lines.append(begin_point)
                lines.append(end_point)
        self.lines = np.array(lines, dtype=np.float32)
        self._create_ground_plane()
        print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _add_circle_borderline(self, env, radius):
        lines = self.lines * radius
        colors = np.array([[1, 0, 0]] * (len(lines) // 2), dtype=np.float32)
        self.gym.add_lines(self.viewer, env, len(lines) // 2, lines, colors)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        ant_asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        ant_asset_file = "mjcf/nv_ant.xml"


        ant_asset_path = os.path.join(ant_asset_root, ant_asset_file)
        ant_asset_root = os.path.dirname(ant_asset_path)
        ant_asset_file = os.path.basename(ant_asset_path)
        
        print("ant_asset_path:", ant_asset_path)

        asset_options = gymapi.AssetOptions()
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.angular_damping = 0.0
        ant_assets = []
        for _ in range(self.num_agents1):
            ant_asset = self.gym.load_asset(self.sim, ant_asset_root, ant_asset_file, asset_options)
            ant_assets.append(ant_asset)
        ant_dof_props = self.gym.get_asset_dof_properties(ant_assets[0])

        # self.num_dof = self.gym.get_asset_dof_count(ant_assets[0])
        # print("=====num_dof=====", self.num_dof)
        # self.num_bodies = self.gym.get_asset_rigid_body_count(ant_assets[0])
        # print("=====num_bodies=====", self.num_bodies)
        
        for i in range(self.ant_dof_cnt):
            ant_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            ant_dof_props['stiffness'][i] = self.Kp
            ant_dof_props['damping'][i] = self.Kd
            
        bug_asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        bug_asset_file = "mjcf/temp.xml"
        bug_asset_path = os.path.join(bug_asset_root, bug_asset_file)
        bug_asset_root = os.path.dirname(bug_asset_path)
        bug_asset_file = os.path.basename(bug_asset_path)
        print("bug_asset_path:", bug_asset_path)
        bug_assets = []
        for _ in range(self.num_agents2):
            bug_asset = self.gym.load_asset(self.sim, bug_asset_root, bug_asset_file, asset_options)
            bug_assets.append(bug_asset)
        bug_dof_props = self.gym.get_asset_dof_properties(bug_assets[0])
        for i in range(self.bug_dof_cnt):
            bug_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            bug_dof_props['stiffness'][i] = self.Kp
            bug_dof_props['damping'][i] = self.Kd

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(-self.borderline_space + 1, -self.borderline_space + 1, 1.)
        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w],
                                           device=self.device)

        self.torso_index = 0
        self.num_bodies_ant = self.gym.get_asset_rigid_body_count(ant_assets[0])
        ant_body_names = [self.gym.get_asset_rigid_body_name(ant_assets[0], i) for i in range(self.num_bodies_ant)]

        ant_extremity_names = [s for s in ant_body_names if "foot" in s]
        self.ant_extremities_index = torch.zeros(len(ant_extremity_names), dtype=torch.long, device=self.device)

        print("Ant Robot Body Names:", ant_body_names)
        print("Ant Robot Extremity Names:", ant_extremity_names)

        ant_extremity_indices = [self.gym.find_asset_rigid_body_index(ant_assets[0], name) for name in ant_extremity_names]
        sensor_pose = gymapi.Transform()
        for body_idx in ant_extremity_indices:
            for agent_idx in range(self.num_agents1):
                self.gym.create_asset_force_sensor(ant_assets[agent_idx], body_idx, sensor_pose)

        self.num_bodies_bug = self.gym.get_asset_rigid_body_count(bug_assets[0])
        bug_body_names = [self.gym.get_asset_rigid_body_name(bug_assets[0], i) for i in range(self.num_bodies_bug)]

        bug_extremity_names = [s for s in bug_body_names if "foot" in s]
        self.bug_extremities_index = torch.zeros(len(bug_extremity_names), dtype=torch.long, device=self.device)

        print("Bug Robot Body Names:", bug_body_names)
        print("Bug Robot Extremity Names:", bug_extremity_names)

        bug_extremity_indices = [self.gym.find_asset_rigid_body_index(bug_assets[0], name) for name in bug_extremity_names]
        for body_idx in bug_extremity_indices:
            for agent_index in range(self.num_agents2):
                self.gym.create_asset_force_sensor(bug_assets[0], body_idx, sensor_pose)


        # self.ant_handles = []
        # self.actor_indices = []
        # self.envs = []
        # self.dof_limits_lower = []
        # self.dof_limits_upper = []

        # for i in range(self.num_envs):
        #     # create env instance
        #     env_ptr = self.gym.create_env(
        #         self.sim, lower, upper, num_per_row
        #     )
        #     # create actor instance
        #     for j in range(self.num_agents):
        #         ant_handle = self.gym.create_actor(env_ptr, ant_assets[j], start_pose, "ant_" + str(j), i, -1, 0)
        #         actor_index = self.gym.get_actor_index(env_ptr, ant_handle, gymapi.DOMAIN_SIM)
        #         self.gym.set_actor_dof_properties(env_ptr, ant_handle, dof_props)
        #         self.actor_indices.append(actor_index)
        #         self.gym.enable_actor_dof_force_sensors(env_ptr, ant_handle)
        #         self.ant_handles.append(ant_handle)self.num_dof
        #     if dof_prop['lower'][j] > dof_prop['upper'][j]:
        #         self.dof_limits_lower.append(dof_prop['upper'][j])
        #         self.dof_limits_upper.append(dof_prop['lower'][j])
        #     else:
        #         self.dof_limits_lower.append(dof_prop['lower'][j])
        #         self.dof_limits_upper.append(dof_prop['upper'][j])

        # self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        # self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
        # self.actor_indices = to_torch(self.actor_indices, device=self.device).to(dtype=torch.int32)

        # for i in range(len(extremity_names)):
        #     self.extremities_index[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ant_handles[0],
        #                                                                       extremity_names[i])
        
        self.ant_handles = []
        self.bug_handles = []
        self.actor_indices = []
        self.envs = []
        self.ant_dof_limits_lower = []
        self.ant_dof_limits_upper = []
        self.bug_dof_limits_lower = []
        self.bug_dof_limits_upper = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            for j in range(self.num_agents1):
                ant_handle = self.gym.create_actor(env_ptr, ant_assets[j], start_pose, "ant_" + str(j), i, -1, 0)
                actor_index = self.gym.get_actor_index(env_ptr, ant_handle, gymapi.DOMAIN_SIM)
                self.gym.set_actor_dof_properties(env_ptr, ant_handle, ant_dof_props)
                self.actor_indices.append(actor_index)
                self.gym.enable_actor_dof_force_sensors(env_ptr, ant_handle)
                self.ant_handles.append(ant_handle)
                for k in range(self.num_bodies_ant):
                    self.gym.set_rigid_body_color(env_ptr, ant_handle, k, gymapi.MESH_VISUAL, self.ant_body_colors[0])

            for j in range(self.num_agents2):
                bug_handle = self.gym.create_actor(env_ptr, bug_assets[j], start_pose, "bug_" + str(j), i, -1, 0)
                actor_index = self.gym.get_actor_index(env_ptr, bug_handle, gymapi.DOMAIN_SIM)
                self.gym.set_actor_dof_properties(env_ptr, bug_handle, bug_dof_props)
                self.actor_indices.append(actor_index)
                self.gym.enable_actor_dof_force_sensors(env_ptr, bug_handle)
                self.bug_handles.append(bug_handle)

                for k in range(self.num_bodies_bug):
                    self.gym.set_rigid_body_color(env_ptr, bug_handle, k, gymapi.MESH_VISUAL, self.bug_body_colors[1])

            self.envs.append(env_ptr)

        ant_dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.ant_handles[0])
        print("ant dof_prop:", ant_dof_prop['upper'])
        bug_dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.bug_handles[0])
        print("bug dof_prop:", bug_dof_prop['upper'])

        for j in range(self.ant_dof_cnt):
            if ant_dof_prop['lower'][j] > ant_dof_prop['upper'][j]:
                self.ant_dof_limits_lower.append(ant_dof_prop['upper'][j])
                self.ant_dof_limits_upper.append(ant_dof_prop['lower'][j])
            else:
                self.ant_dof_limits_lower.append(ant_dof_prop['lower'][j])
                self.ant_dof_limits_upper.append(ant_dof_prop['upper'][j])
                
        for j in range(self.bug_dof_cnt):
            if bug_dof_prop['lower'][j] > bug_dof_prop['upper'][j]:
                self.bug_dof_limits_lower.append(bug_dof_prop['upper'][j])
                self.bug_dof_limits_upper.append(bug_dof_prop['lower'][j])
            else:
                self.bug_dof_limits_lower.append(bug_dof_prop['lower'][j])
                self.bug_dof_limits_upper.append(bug_dof_prop['upper'][j])

        self.ant_dof_limits_lower = to_torch(self.ant_dof_limits_lower, device=self.device)
        self.ant_dof_limits_upper = to_torch(self.ant_dof_limits_upper, device=self.device)
        self.bug_dof_limits_lower = to_torch(self.bug_dof_limits_lower, device=self.device)
        self.bug_dof_limits_upper = to_torch(self.bug_dof_limits_upper, device=self.device)
        self.actor_indices = to_torch(self.actor_indices, device=self.device).to(dtype=torch.int32)

        for i in range(len(ant_extremity_names)):
            self.ant_extremities_index[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.ant_handles[0], ant_extremity_names[i]
            )

        for i in range(len(bug_extremity_names)):
            self.bug_extremities_index[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.bug_handles[0], bug_extremity_names[i]
            )


    def compute_reward(self, actions1, actions2):

        self.rew_buf[:], self.reset_buf[:], self.extras['ranks'][:], self.extras['win'], self.extras['lose'], \
        self.extras[
            'draw'] = compute_agent_reward(
            self.obs_buf1,
            self.obs_buf2,
            self.reset_buf,
            self.progress_buf,
            self.torques,
            self.extras['ranks'],
            self.termination_height,
            self.max_episode_length,
            self.borderline_space,
            self.borderline_space_unit,
            self.win_reward_scale,
            self.stay_in_center_reward_scale,
            self.action_cost_scale,
            self.push_scale,
            self.joints_at_limit_cost_scale,
            self.dense_reward_scale,
            self.dt,
            self.num_agents1,
            self.num_agents2
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        for agent_idx in range(self.num_agents1 + self.num_agents2):
            if agent_idx < self.num_agents1:
                self.obs_buf1[agent_idx * self.num_envs:(agent_idx + 1) * self.num_envs, :] = compute_agent_observations(
                    self.ant_agents_state,
                    self.bug_agents_state,
                    self.progress_buf,
                    self.ant_dof_limits_lower,
                    self.ant_dof_limits_upper,
                    self.bug_dof_limits_lower,
                    self.bug_dof_limits_upper,
                    self.dof_vel_scale,
                    self.termination_height,
                    self.borderline_space_unit,
                    self.borderline_space,
                    self.num_agents1,
                    self.num_agents2,
                    agent_idx,
                )
            else:
                self.obs_buf2[(agent_idx - self.num_agents1) * self.num_envs:(agent_idx - self.num_agents1 + 1) * self.num_envs, :] = compute_agent_observations(
                    self.ant_agents_state,
                    self.bug_agents_state,
                    self.progress_buf,
                    self.ant_dof_limits_lower,
                    self.ant_dof_limits_upper,
                    self.bug_dof_limits_lower,
                    self.bug_dof_limits_upper,
                    self.dof_vel_scale,
                    self.termination_height,
                    self.borderline_space_unit,
                    self.borderline_space,
                    self.num_agents1,
                    self.num_agents2,
                    agent_idx,
                )

    def reset_idx(self, env_ids):
        # print('reset.....', env_ids)
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        ant_positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.ant_dof_cnt), device=self.device)
        bug_positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.bug_dof_cnt), device=self.device)
        ant_velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.ant_dof_cnt), device=self.device)
        bug_velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.bug_dof_cnt), device=self.device)

        for agent_idx in range(self.num_agents1):
            ant_root_state, ant_dof_pos, ant_dof_vel = self.ant_agents_state[agent_idx]
            ant_dof_pos[env_ids] = tensor_clamp(self.ant_initial_dof_pos[env_ids] + ant_positions, self.ant_dof_limits_lower,
                                            self.ant_dof_limits_upper)
            ant_dof_vel[env_ids] = ant_velocities
        for agent_idx in range(self.num_agents2):
            bug_root_state, bug_dof_pos, bug_dof_vel = self.bug_agents_state[agent_idx]
            bug_dof_pos[env_ids] = tensor_clamp(self.bug_initial_dof_pos[env_ids] + bug_positions, self.bug_dof_limits_lower,
                                            self.bug_dof_limits_upper)
            bug_dof_vel[env_ids] = bug_velocities
        agent_env_ids = expand_env_ids(env_ids, self.num_agents)
        env_ids_int32 = self.actor_indices[agent_env_ids]
        rand_angle = torch.rand((len(env_ids),), device=self.device) * torch.pi * 2  # generate angle in 0-360

        rand_pos = (self.borderline_space * torch.ones((len(agent_env_ids), 2), device=self.device) -
                    torch.rand((len(agent_env_ids), 2), device=self.device))

        unit_angle = 2 * torch.pi / self.num_agents
        for agent_idx in range(self.num_agents):
            rand_pos[agent_idx::self.num_agents, 0] *= torch.cos(rand_angle + agent_idx * unit_angle)
            rand_pos[agent_idx::self.num_agents, 1] *= torch.sin(rand_angle + agent_idx * unit_angle)
        rand_floats = torch_rand_float(-1.0, 1.0, (len(agent_env_ids), 1), device=self.device)
        rand_rotation = quat_from_angle_axis(rand_floats[:, 0] * np.pi, self.z_unit_tensor[agent_env_ids])
        self.root_states[agent_env_ids] = self.initial_root_states[agent_env_ids]
        self.root_states[agent_env_ids, :2] = rand_pos
        self.root_states[agent_env_ids, 3:7] = rand_rotation
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.extras['ranks'][env_ids] = 0

    def pre_physics_step(self, actions1, actions2):                                                                    # ??
        # actions.shape = [num_envs * num_agents, num_actions], stacked as followed:
        # {[(agent1_act_1, agent1_act2)|(agent2_act1, agent2_act2)|...]_(env0),
        #  [(agent1_act_1, agent1_act2)|(agent2_act1, agent2_act2)|...]_(env1),
        #  ... }

        self.actions1 = torch.tensor([], device=self.device)
        self.actions2 = torch.tensor([], device=self.device)
        for agent_idx in range(self.num_agents1):
            self.actions1 = torch.cat((self.actions1, actions1[agent_idx * self.num_envs:(agent_idx + 1) * self.num_envs]),
                                     dim=-1)
        for agent_idx in range(self.num_agents2):
            self.actions2 = torch.cat((self.actions2, actions2[agent_idx * self.num_envs:(agent_idx + 1) * self.num_envs]),
                                     dim=-1)
        tmp_actions1 = self.extras['ranks'][:, :self.num_agents1].unsqueeze(-1).repeat_interleave(self.num_actions1, dim=-1).view(self.num_envs,
                                                                                                          self.num_actions1 * self.num_agents1)
        tmp_actions2 = self.extras['ranks'][:, self.num_agents1:].unsqueeze(-1).repeat_interleave(self.num_actions2, dim=-1).view(self.num_envs,
                                                                                                          self.num_actions2 * self.num_agents2)
        zero_actions1 = torch.zeros_like(tmp_actions1, dtype=torch.float)
        zero_actions2 = torch.zeros_like(tmp_actions2, dtype=torch.float)
        self.actions1 = torch.where(tmp_actions1 > 0, zero_actions1, self.actions1)
        self.actions2 = torch.where(tmp_actions2 > 0, zero_actions2, self.actions2)

        # reshape [num_envs * num_agents, num_actions] to [num_envs, num_agents * num_actions] print(f'action_size{
            
        self.actions = torch.cat((self.actions1, self.actions2), dim=-1)

        targets = self.actions

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        resets = self.reset_buf.reshape(self.num_envs, 1).sum(dim=1)
        # print(resets)
        env_ids = (resets == 1).nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions1, self.actions2)

        if self.viewer is not None:
            self.gym.clear_lines(self.viewer)
            for i, env in enumerate(self.envs):
                self._add_circle_borderline(env, self.borderline_space - self.borderline_space_unit * self.progress_buf[
                    i].item())

    def get_number_of_agents(self):
        # only train 2 agent
        return 2

    def zero_actions1(self) -> torch.Tensor:
        """Returns a buffer with zero actions.

        Returns:
            A buffer of zero torch actions
        """
        actions = torch.zeros([self.num_envs * self.num_agents, self.num_actions1], dtype=torch.float32,
                              device=self.rl_device)
        self.extras['win'] = self.extras['lose'] = self.extras['draw'] = 0
        return actions
    
    def zero_actions2(self) -> torch.Tensor:
        """Returns a buffer with zero actions.

        Returns:
            A buffer of zero torch actions
        """
        actions = torch.zeros([self.num_envs * self.num_agents, self.num_actions2], dtype=torch.float32,
                              device=self.rl_device)
        self.extras['win'] = self.extras['lose'] = self.extras['draw'] = 0
        return actions

    def clear_count(self):
        self.dense_reward_scale *= 0.9
        self.extras['ranks'] = torch.zeros((self.num_agents, self.num_agents), device=self.device, dtype=torch.float)


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def expand_env_ids(env_ids, n_agents):
    # type: (Tensor, int) -> Tensor
    device = env_ids.device
    # print(f'nanget:{n_agents}')
    agent_env_ids = torch.zeros((n_agents * len(env_ids)), device=device, dtype=torch.long)
    for idx in range(n_agents):
        agent_env_ids[idx::n_agents] = env_ids * n_agents + idx
    return agent_env_ids


@torch.jit.script
def compute_agent_reward(
        obs_buf1,
        obs_buf2,
        reset_buf,
        progress_buf,
        torques,
        now_rank,
        termination_height,
        max_episode_length,
        borderline_space,
        borderline_space_unit,
        win_reward_scale,
        stay_in_center_reward_scale,
        action_cost_scale,
        push_scale,
        joints_at_limit_cost_scale,
        dense_reward_scale,
        dt,
        num_agents1,
        num_agents2
):
    # type: (Tensor, Tensor, Tensor, Tensor,Tensor,Tensor,float,float,float,float,float,float,float,float,float,float,float,int,int) -> Tuple[Tensor, Tensor,Tensor,Tensor,Tensor,Tensor]
    # print("input list:", obs_buf1.shape, obs_buf2.shape, reset_buf.shape, progress_buf.shape, torques.shape, now_rank.shape, termination_height, max_episode_length, borderline_space, borderline_space_unit, win_reward_scale, stay_in_center_reward_scale, action_cost_scale, push_scale, joints_at_limit_cost_scale, dense_reward_scale, dt, num_agents1, num_agents2)
    obs1 = obs_buf1.view(num_agents1, -1, obs_buf1.shape[1])
    obs2 = obs_buf2.view(num_agents2, -1, obs_buf2.shape[1])
    nxt_rank_val = num_agents1 + num_agents2 - torch.count_nonzero(now_rank, dim=-1).view(-1, 1).repeat_interleave(num_agents1 + num_agents2, dim=-1)
    is_out = torch.sum(torch.square(torch.cat((obs1[:, :, 0:2], obs2[:, :, 0:2]), dim=0)), dim=-1) >= \
             (borderline_space - progress_buf * borderline_space_unit).square()
    nxt_rank = torch.where((torch.transpose(is_out, 0, 1) > 0) & (now_rank == 0), nxt_rank_val, now_rank)
    # reset agents
    tmp_ones = torch.ones_like(reset_buf)
    reset = torch.where(torch.min(is_out[:num_agents1], dim=0).values, tmp_ones, reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, tmp_ones, reset)
    reset = torch.where(torch.min(is_out[num_agents1:], dim=0).values, tmp_ones, reset)
    tmp_reset = reset.view(-1, 1).repeat_interleave(num_agents1 + num_agents2, dim=-1)
    nxt_rank = torch.where((tmp_reset == 1) & (nxt_rank == 0),
                           nxt_rank_val - 1,
                           nxt_rank)
    # compute metric logic
    tmp_reset = reset.view(1, -1).repeat_interleave(num_agents1 + num_agents2, dim=0)
    tmp_zeros = torch.zeros_like(is_out, dtype=torch.bool)
    wins = torch.ones_like(is_out, dtype=torch.bool)
    loses = torch.ones_like(wins, dtype=torch.bool)
    draws = (progress_buf >= max_episode_length - 1).view(1, -1).repeat_interleave(num_agents1 + num_agents2, dim=0)
    wins = torch.where(is_out, wins & (tmp_reset == 1), tmp_zeros) # (num_agents, num_envs)
    draws = torch.where(is_out == 0, draws & (tmp_reset == 1), tmp_zeros)
    loses = torch.where(is_out == 0, loses & (tmp_reset == 1) & (draws == 0), tmp_zeros)

    sparse_reward = 1.0 * reset.unsqueeze(-1)
    reward_per_rank = 2 * win_reward_scale / (num_agents1 + num_agents2)
    sparse_reward = sparse_reward * (win_reward_scale - (nxt_rank - 1) * reward_per_rank)
    ant_stay_in_center_reward = stay_in_center_reward_scale * torch.exp(-torch.linalg.norm(obs1[:, :, :2], dim=-1))
    bug_stay_in_center_reward = stay_in_center_reward_scale * torch.exp(-torch.linalg.norm(obs2[:, :, :2], dim=-1))
    ant_dof_at_limit_cost = torch.sum(obs1[:, :, 13:21] > 0.99, dim=-1) * joints_at_limit_cost_scale
    bug_dof_at_limit_cost = torch.sum(obs2[:, :, 13:21] > 0.99, dim=-1) * joints_at_limit_cost_scale
    ant_action_cost_penalty = torch.sum(torch.square(torques[:, :num_agents1 * 8]).view(-1, num_agents1, 8), dim=-1) * action_cost_scale
    bug_action_cost_penalty = torch.sum(torch.square(torques[:, num_agents1 * 8:]).view(-1, num_agents2, 12), dim=-1) * action_cost_scale
    # print("torques:", torques[0, 2])
    ant_not_move_penalty = torch.exp(-torch.sum(torch.abs(torques[:, :num_agents1 * 8]).view(-1, num_agents1, 8), dim=-1))
    bug_not_move_penalty = torch.exp(-torch.sum(torch.abs(torques[:, num_agents1 * 8:]).view(-1, num_agents2, 12), dim=-1))
    # print("shape used in the below two lines:", ant_dof_at_limit_cost.shape, ant_action_cost_penalty.shape, ant_not_move_penalty.shape, ant_stay_in_center_reward.shape)
    # print(f'action:...{action_cost_penalty.shape}')
    ant_dense_reward = ant_dof_at_limit_cost.transpose(0,1) + ant_action_cost_penalty + ant_not_move_penalty + ant_stay_in_center_reward.transpose(0, 1)
    bug_dense_reward = bug_dof_at_limit_cost.transpose(0,1) + bug_action_cost_penalty + bug_not_move_penalty + bug_stay_in_center_reward.transpose(0, 1)
    total_reward = sparse_reward + torch.cat([ant_dense_reward * dense_reward_scale, bug_dense_reward * dense_reward_scale], dim=1)
    # print('total_reward.shape:', total_reward.shape)

    return total_reward, reset, nxt_rank, wins.flatten(), loses.flatten(), draws.flatten()


@torch.jit.script
def compute_agent_observations(
        ant_agents_state,
        bug_agents_state,
        progress_buf,
        ant_dof_limits_lower,
        ant_dof_limits_upper,
        bug_dof_limits_lower,
        bug_dof_limits_upper,
        dof_vel_scale,
        termination_height,
        borderline_space_unit,
        borderline_space,
        num_agents1,
        num_agents2,
        agent_idx,
):
    # type: (List[Tuple[Tensor,Tensor,Tensor]], List[Tuple[Tensor, Tensor, Tensor]],Tensor,Tensor,Tensor,Tensor,Tensor,float,float,float,float,int,int, int)->Tensor
    # tot length:13+12+12+1+1+1+(num_agents-1)*(7+2+12+12+1+1)
    if agent_idx < num_agents1:
        be_ant = True
        self_root_state, self_dof_pos, self_dof_vel = ant_agents_state[agent_idx]
    else:
        be_ant = False
        self_root_state, self_dof_pos, self_dof_vel = bug_agents_state[agent_idx - num_agents1]
    dof_pos_scaled = unscale(self_dof_pos, ant_dof_limits_lower if be_ant else bug_dof_limits_lower, ant_dof_limits_upper if be_ant else bug_dof_limits_upper)
    now_border_space = (borderline_space - progress_buf * borderline_space_unit).unsqueeze(-1)
    
    # print all tensors' shape
    # print(f'self_root_state:{self_root_state.shape}, self_dof_pos:{self_dof_pos.shape}, self_dof_vel:{self_dof_vel.shape}, dof_pos_scaled:{dof_pos_scaled.shape}, now_border_space:{now_border_space.shape}')
    obs = torch.cat((self_root_state[:, :13], dof_pos_scaled, self_dof_vel * dof_vel_scale,
                     now_border_space - torch.sqrt(torch.sum(self_root_state[:, :2].square(), dim=-1)).unsqueeze(-1),
                     # dis to border
                     now_border_space,
                     torch.unsqueeze(self_root_state[:, 2] < termination_height, -1)), dim=-1) 
    for op_idx in range(num_agents1 + num_agents2):
        if op_idx == agent_idx:
            continue
        if op_idx < num_agents1:
            be_ant = True
            op_root_state, op_dof_pos, op_dof_vel = ant_agents_state[op_idx]
        else:
            be_ant = False
            op_root_state, op_dof_pos, op_dof_vel = bug_agents_state[op_idx - num_agents1]
        dof_pos_scaled = unscale(op_dof_pos, ant_dof_limits_lower if be_ant else bug_dof_limits_lower, ant_dof_limits_upper if be_ant else bug_dof_limits_upper)

        obs = torch.cat((obs, op_root_state[:, :7], self_root_state[:, :2] - op_root_state[:, :2],
                         dof_pos_scaled, op_dof_vel * dof_vel_scale,
                         now_border_space - torch.sqrt(torch.sum(op_root_state[:, :2].square(), dim=-1)).unsqueeze(-1),
                         torch.unsqueeze(op_root_state[:, 2] < termination_height, -1)), dim=-1)
    # print(obs.shape)
    return obs


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))
