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
    if config.rssm.discrete:
      stoch_size *= config.rssm.discrete
    size = {
        'embed': 32 * config.encoder.cnn_depth,
        'stoch': stoch_size,
        'deter': config.rssm.deter,
        'feat': stoch_size + config.rssm.deter,
    }[self.config.disag_target]
    self._networks = [
        common.MLP(size, **config.expl_head)
        for _ in range(config.disag_models)]
    self.opt = common.Optimizer('expl', **config.expl_opt)
    self.extr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)
    self.intr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)
    s