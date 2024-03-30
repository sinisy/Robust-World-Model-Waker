import tensorflow as tf
from tensorflow_probability import distributions as tfd
from collections import deque

import dreamerv2
import common
import numpy as np


class RandomExplore(common.Module):

  def __init__(self, config, act_space, wm, tfstep, reward):
    self.act_space = act_space
    self.config = config
    self.reward = reward
    self.wm = wm
    self.actor = self.random_actor
    stoch_size = config.rssm.stoch
    if config.