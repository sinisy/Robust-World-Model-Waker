
#!/usr/bin/env python

import gym
import gym.spaces
import numpy as np
import common
from PIL import Image
from copy import deepcopy
from collections import OrderedDict
import mujoco_py
from mujoco_py import MjViewer, MujocoException, const, MjRenderContextOffscreen

from envs.safety_gym.envs.engine import Engine

import sys


class ReplayEngine(Engine):

    '''
    ReplayEngine: a class to enable resetting environments to earlier configurations.

    '''
    def __init__(self, config={}, param_list=None, eval_cases=None):
        super().__init__(config=config)
        self._param_list = param_list
        self.eval_cases = eval_cases

    @property
    def max_num_objects(self):
        return self.buttons_num + self.hazards_num + self.vases_num + self.pillars_num + self.gremlins_num
    
    def env_params_from_layout(self, layout):
        ''' Stores the number of each type of object in a numpy array'''
        
        num_config_params = len(self.configurable_types)
        env_params = np.zeros(num_config_params + 1)
        env_params[0] = float(self.arena)
        for i, object_type in enumerate(self.configurable_types):
            obj_num = self.num_objects_in_layout(object_type, layout)
            env_params[i + 1] = float(obj_num)
        return env_params
    