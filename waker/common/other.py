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
    inp = tf.nest.map_structure(lambda x: x[index], inputs)
    last = fn(last, inp)
    [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
  if reverse:
    outputs = [list(reversed(x)) for x in outputs]
  outputs = [tf.stack(x, 0) for x in outputs]
  return tf.nest.pack_sequence_as(start, outputs)


def schedule(string, step):
  try:
    return float(string)
  except ValueError:
    step = tf.cast(step, tf.float32)
    match = re.match(r'linear\((.+),(.+),(.+)\)', string)
    if match:
      initial, final, duration = [float(group) for group in match.groups()]
      mix = tf.clip_by_value(step / duration, 0, 1)
      return (1 - mix) * initial + mix * final
    match = re.match(r'warmup\((.+),(.+)\)', string)
    if match:
      warmup, value = [float(group) for group in match.groups()]
      scale = tf.clip_by_value(step / warmup, 0, 1)
      return scale * value
    match = re.match(r'exp\((.+),(.+),(.+)\)', string)
    if match:
      initial, final, halflife = [float(group) for group in match.groups()]
      return (initial - final) * 0.5 ** (step / halflife) + final
    match = re.match(r'horizon\((.+),(.+),(.+)\)', string)
    if match:
      initial, final, duration = [float(group) for group in match.groups()]
      mix = tf.clip_by_value(step / duration, 0, 1)
      horizon = (1 - mix) * initial + mix * final
      return 1 - 1 / horizon
    raise NotImplementedError(string)


def lambda_return(
    reward, value, pcont, bootstrap, lambda_, axis):
  # Setting lambda=1 gives a discounted Monte Carlo return.
  # Setting lambda=0 gives a fixed 1-step return.
  assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
  if isinstance(pcont, (int, float)):
    pcont = pcont * tf.ones_like(reward)
  dims = l