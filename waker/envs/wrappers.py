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
    except AttributeError:
      raise ValueError(name)

  @property
  def obs_space(self):
    if self._obs_is_dict:
      spaces = self._env.observation_space.spaces.copy()
    else:
      spaces = {self._obs_key: self._env.observation_space}
    return {
        **spaces,
        'image': gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=bool),
    }

  @property
  def act_space(self):
    return {self._act_key: self._env.action_space}
  
  def eval_cases(self, task):
    return self._env.eval_cases

  def step(self, action):
    if not self._act_is_dict:
      action = action[self._act_key]

    if self._dict_reward:
        reward = []
    else:
        reward = 0.0
    for _ in range(self._action_repeat):
      obs, rew, done, info = self._env.step(action)
      if self._dict_reward:
        curr_reward = list(rew.values())
        if len(reward) == 0:
          reward = curr_reward
        else:
          reward = [sum(x) for x in zip(reward, curr_reward)]
      else:
          reward += rew or 0.0
      if done:
        break

    if not isinstance(obs, dict):
      obs = {self._obs_key: obs}
    obs['reward'] = reward
    obs['is_first'] = False
    obs['is_last'] = done
    obs['is_terminal'] = info.get('is_terminal', done)
    obs['image'] = self.render()

    if "task_completion" in info.keys():
      obs["task_completion"] = info["task_completion"]
    return obs
  
  def render(self):
    img = self._env.render(camera_id=0,
                          mode="rgb_array",
                          height=self._size[0],
                          width=self._size[