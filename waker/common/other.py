import collections
import contextlib
import re
import time

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from . import dists
from . import tfutils


class RandomAgent:

  def __init__(self, act_space, logprob=False):
    self.act_space = act_space['action']
    self.logprob = logprob
    if hasattr(self.act_space, 'n'):
      self._dist = dists.OneHotDist(tf.zeros(self.act_space.n))
    else:
      dist = tfd.Uniform(self.act_space.low, self.act_space.high)
      self._dist = tfd.Independent(dist, 1)

  def __call__(self, obs, state=None, mode=None):
    action = self._dist.sample(len(obs['is_first']))
    output = {'action': action}
    if self.logprob:
      output['logprob'] = self._dist.log_prob(action)
    return output, None


def static_scan(fn, inputs, start, reverse=False):
  last = start
  outputs = [[] for _ in tf.nest.flatten(start)]
  indices = range(tf.nest.flatten(inputs)[0].shape[0])
  if reverse:
    indices = reversed(indices)
  for index in indices:
    inp = tf.nest.map_s