import atexit
import os
import sys
import threading
import traceback

import cloudpickle
import gym
import numpy as np
import common

class SafetyGymWrapper:

  def __init__(self, name, env, action_repeat=1, obs_key='state', act_key='action', size=(64, 64)):
    if "all" in name:
      self._dict_reward = True
      self._tasks = common.DOMAIN_TASK_IDS[name]
    self.name = name

    self._env = env
    self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_is_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key
    self._action_repeat = action_repeat
    self._size = size

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError