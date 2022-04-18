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
 