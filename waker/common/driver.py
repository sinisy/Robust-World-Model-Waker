import numpy as np


class Driver:

  def __init__(self, envs, **kwargs):
    self._envs = envs
    self._kwargs = kwargs
    self._on_steps = []
    self._on_resets = []
    self._on_episodes = []
    self._on_calls = []
    self._act_spaces = [env.act_space for env in envs]
    self._act_is_discrete = [hasattr(s['action'], 'n') for s in self._act_spaces]
    self.total_episodes = 0
    self.reset()

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_reset(self, callback):
    self._on_resets.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def on_call(self, callback):
    self._on_calls.append(callback)

  def reset(self):
    self._obs = [None] * len(self._envs)
    self._eps = [None] * len(self._envs)
    self._state = None

  def __call__(self, policy, steps=0, episodes=0, env_sampler=None, task=None):
    step, episode = 0, 0
    eps = []
    while step < steps or episode < episodes:
      if env_sampler is None: