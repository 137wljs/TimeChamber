# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Dict, Any, Tuple

import gym
from gym import spaces

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch
from isaacgym.gymutil import get_property_setter_map, get_property_getter_map, get_default_setter_args, apply_random_samples, check_buckets, generate_random_samples

import torch
import numpy as np
import operator, random
from copy import deepcopy
import sys

import abc
from abc import ABC

EXISTING_SIM = None
SCREEN_CAPTURE_RESOLUTION = (1027, 768)

def _create_sim_once(gym, *args, **kwargs):
    global EXISTING_SIM
    if EXISTING_SIM is not None:
        return EXISTING_SIM
    else:
        EXISTING_SIM = gym.create_sim(*args, **kwargs)
        return EXISTING_SIM


class VA_Env(ABC):
    def __init__(self, config: Dict[str, Any], rl_device: str, sim_device: str, graphics_device_id: int, headless: bool):
        """Initialise the env.

        Args:
            config: the configuration dictionary.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
        """

        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0

        self.device = "cpu"
        if config["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
                self.device = "cuda" + ":" + str(self.device_id)
            else:
                print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
                config["sim"]["use_gpu_pipeline"] = False

        self.rl_device = rl_device

        # Rendering
        # if training in a headless mode
        self.headless = headless

        enable_camera_sensors = config.get("enableCameraSensors", False)
        self.graphics_device_id = graphics_device_id
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1

        self.num_environments = config["env"]["numEnvs"]
        self.num_agents1 = config["env"].get("numAgents1", 1)  # used for various-agent environments
        self.num_agents2 = config["env"].get("numAgents2", 1)  # used for various-agent environments
        self.num_agents = self.num_agents1 + self.num_agents2
        self.num_observations1 = config["env"]["numObservations1"]
        self.num_observations2 = config["env"]["numObservations2"]
        self.num_states1 = config["env"].get("numStates1", 0)    # seem useless
        self.num_states2 = config["env"].get("numStates2", 0)    # seem useless
        self.num_actions1 = config["env"]["numActions1"]
        self.num_actions2 = config["env"]["numActions2"]
        

        self.control_freq_inv = config["env"].get("controlFrequencyInv", 1)

        self.obs_space = (spaces.Box(np.ones(self.num_observations1) * -np.Inf, np.ones(self.num_observations1) * np.Inf), spaces.Box(np.ones(self.num_observations2) * -np.Inf, np.ones(self.num_observations2) * np.Inf))
        self.state_space = (spaces.Box(np.ones(self.num_states1) * -np.Inf, np.ones(self.num_states1) * np.Inf), spaces.Box(np.ones(self.num_states2) * -np.Inf, np.ones(self.num_states2) * np.Inf))

        self.act_space = (spaces.Box(np.ones(self.num_actions1) * -1., np.ones(self.num_actions1) * 1.), spaces.Box(np.ones(self.num_actions2) * -1., np.ones(self.num_actions2) * 1.))

        self.clip_obs = config["env"].get("clipObservations", np.Inf)
        self.clip_actions = config["env"].get("clipActions", np.Inf)

    @abc.abstractmethod 
    def allocate_buffers(self):
        """Create torch buffers for observations, rewards, actions dones and any additional data."""

    @abc.abstractmethod
    def step(self, actions1: torch.Tensor, actions2: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

    @abc.abstractmethod
    def reset(self)-> Dict[str, torch.Tensor]:
        """Reset the environment.
        Returns:
            Observation dictionary
        """

    @abc.abstractmethod
    def reset_idx(self, env_ids: torch.Tensor):
        """Reset environments having the provided indices.
        Args:
            env_ids: environments to reset
        """

    @property
    def observation_space(self) -> gym.Space:
        """Get the environment's observation space."""
        return self.obs_space

    @property
    def action_space(self) -> gym.Space:
        """Get the environment's action space."""
        return self.act_space

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.num_environments

    @property
    def num_acts(self) -> int:
        """Get the number of actions in the environment."""
        return (self.num_actions1, self.num_actions2)

    @property
    def num_obs(self) -> int:
        """Get the number of observations in the environment."""
        return (self.num_observations1, self.num_observations2)