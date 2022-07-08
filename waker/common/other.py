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
      dist = tfd.Uniform(self.act_space.low, self.act_spa