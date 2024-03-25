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
                          width=self._size[1])
    return img

  def reset(self, env_params=None, task=None):
    obs, info = self._env.reset(env_params=env_params)
    if not isinstance(obs, dict):
      obs = {self._obs_key: obs}
    if self._dict_reward:
      reward = [0.0 for _ in self._tasks]
    else:
      reward = 0.0
    obs['reward'] = reward
    obs['is_first'] = True
    obs['is_last'] = False
    obs['is_terminal'] = False
    obs['image'] = self.render()
    if "task_completion" in info.keys():
      obs["task_completion"] = info["task_completion"]
    return obs
  
class CombinedEnvWrapper:

  def __init__(self, safety_gym_env, dmc_env, dmc_env_params=3, safety_gym_env_params=6):
    self.safety_gym_env = safety_gym_env
    self.dmc_env = dmc_env
    self.dmc_env_params = dmc_env_params
    self.safety_gym_env_params = safety_gym_env_params
    self.aug_env_params = max(self.dmc_env_params, self.safety_gym_env_params) + 1

    self.safety_gym_actions = int(self.safety_gym_env.act_space["action"].shape[0])
    self.dmc_actions = int(self.dmc_env.act_space["action"].shape[0])
    self.num_actions = max(self.safety_gym_actions, self.dmc_actions)

    self.dmc_tasks = self.dmc_env._tasks
    self.safety_gym_tasks = self.safety_gym_env._tasks
    self.name = self.dmc_env.name + "-" + self.safety_gym_env.name
    self.combined_tasks = common.DOMAIN_TASK_IDS[self.name]

    self.act_space = {"action": gym.spaces.Box(-1, 1, (self.num_actions,), dtype=np.float32)}
    self.obs_space = self.safety_gym_env.obs_space

    self.obs_keys = ["env_params", "image", "reward", "is_first", "is_last", "is_terminal", "task_completion"]
    self.domain_id_map = {"dmc": 0.0, "safety_gym": 1.0}

    self.dmc_eval_cases = dict()
    for key, cases in self.dmc_env.eval_cases(None).items():
      new_key = "dmc_" + key
      new_cases = [self.to_aug_env_params(case, self.domain_id_map["dmc"]) for case in cases]
      self.dmc_eval_cases[new_key] = new_cases

    self.safety_gym_eval_cases = dict()
    for key, cases in self.safety_gym_env.eval_cases(None).items():
      new_key = "safetygym_" + key
      new_cases = [self.to_aug_env_params(case, self.domain_id_map["safety_gym"]) for case in cases]
      self.safety_gym_eval_cases[new_key] = new_cases

  def eval_cases(self, task):
    if task in self.dmc_tasks:
      return self.dmc_eval_cases
    elif task in self.safety_gym_tasks:
      return self.safety_gym_eval_cases
    else:
      raise ValueError("Task not recognized.")

  def reset(self, env_params=None, task=None):
    # if the task is specified set domain appropriately
    if task is not None:
      if task in self.dmc_tasks:
        self.domain_id = self.domain_id_map["dmc"]
      elif task in self.safety_gym_tasks:
        self.domain_id = self.domain_id_map["safety_gym"]
      else:
        raise ValueError("Task not recognized.")

    # else if env params set use that to env the domain
    elif env_params is not None:
      self.domain_id = env_params[0]

    # otherwise choose randomly
    else:
      if np.random.uniform() < 0.5:
        self.domain_id = self.domain_id_map["dmc"]
      else:
        self.domain_id = self.domain_id_map["safety_gym"]

    # set the current env
    if np.isclose(self.domain_id, self.domain_id_map["safety_gym"]):
      self.current_env = self.safety_gym_env
    elif np.isclose(self.domain_id, self.domain_id_map["dmc"]):
      self.current_env = self.dmc_env
    else:
      raise ValueError("Domain ID not recognized.")

    obs = self.current_env.reset(env_params=self.to_original_env_params(env_params))

    if env_params is not None:
      self.current_env_params = env_params
    else:
      self.current_env_params = self.to_aug_env_params(obs["env_params"])
    
    if "task_completion" not in obs.keys():
      obs["task_completion"] = self.to_combined_task_completion()
    else:
      obs["task_completion"] = self.to_combined_task_completion(obs["task_completion"])

    obs["env_params"] = self.current_env_params.copy()
    if "state" in obs.keys():
      del obs["state"]
    
    obs["reward"] = np.zeros(len(self.combined_tasks))
    obs_fin = {key: obs[key] for key in self.obs_keys}
    return obs_fin
  
  def to_aug_env_params(self, original_env_params=None, domain_id=None):
    if original_env_params is None:
      return None
    original_params_size = len(original_env_params)
    new_env_params = np.zeros(self.aug_env_params)
    if domain_id is None:
      new_env_params[0] = self.domain_id
    else:
      new_env_params[0] = domain_id
    new_env_params[1:(1 + original_params_size)] = original_env_params
    return new_env_params
  
  def to_original_env_params(self, aug_env_params=None):
    if aug_env_params is None:
      return None
    original_env_params = aug_env_params[1:]
    if np.isclose(self.domain_id, self.domain_id_map["safety_gym"]):
      original_env_params = original_env_params[:self.safety_gym_env_params]
    else:
      original_env_params = original_env_params[:self.dmc_env_params]
    return original_env_params
  
  def step(self, action):
    action = action.copy()
    if self.current_env == self.safety_gym_env:
      action["action"] = action["action"][:self.safety_gym_actions]
    else:
      action["action"] = action["action"][:self.dmc_actions]

    obs = self.current_env.step(action)

    if "task_completion" not in obs.keys():
      obs["task_completion"] = self.to_combined_task_completion()
    else:
      obs["task_completion"] = self.to_combined_task_completion(obs["task_completion"])

    obs["env_params"] = self.current_env_params.copy()

    if "state" in obs.keys():
      del obs["state"]
    obs["reward"] = self.to_combined_reward(obs["reward"])
    obs_fin = {key: obs[key] for key in self.obs_keys}
    return obs_fin
  
  def to_combined_reward(self, env_reward):
    combined_reward = np.zeros(len(self.combined_tasks))

    for task in self.combined_tasks:
      if task in self.dmc_tasks and self.current_env == self.dmc_env:
        combined_reward[self.combined_tasks.index(task)] = env_reward[self.dmc_tasks.index(task)]
      elif task in self.safety_gym_tasks and self.current_env == self.safety_gym_env:
        combined_reward[self.combined_tasks.index(task)] = env_reward[self.safety_gym_tasks.index(task)]
    return combined_reward
  
  def to_combined_task_completion(self, task_completion=None):
    combined_task_completion = np.zeros(len(self.combined_tasks))
    if task_completion is None:
      return combined_task_completion
    
    for task in self.combined_tasks:
      if task in self.safety_gym_tasks and self.current_env == self.safety_gym_env:
        combined_task_completion[self.combined_tasks.index(task)] = task_completion[self.safety_gym_tasks.index(task)]
    return combined_task_completion

class DMC:

  def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
    domain, task = name.split('_', 1)
    if task == 'all':
        self._dict_reward = True
        self._tasks = common.DOMAIN_TASK_IDS[name]
    else:
        self._dict_reward = False
    self.name = name
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if domain == 'manip':
      from dm_control import manipulation
      self._env = manipulation.load(task + '_vision')
    elif domain == 'locom':
      from dm_control.locomotion.examples import basic_rodent_2020
      self._env = getattr(basic_rodent_2020, task)()
    else:
      from envs.dm_control import sui