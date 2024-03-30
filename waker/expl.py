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
    self.rewnorm = common.StreamNorm(**self.config.reward_norm)

  def random_actor(self, feat):
    shape = feat.shape[:-1] + self.act_space.shape
    if self.config.actor.dist == 'onehot':
      return common.OneHotDist(tf.zeros(shape))
    else:
      dist = tfd.Uniform(-tf.ones(shape), tf.ones(shape))
      return tfd.Independent(dist, 1)
    
  def train(self, start, context, data):
    metrics = {}
    stoch = start['stoch']
    if self.config.rssm.discrete:
      stoch = tf.reshape(
          stoch, stoch.shape[:-2] + (stoch.shape[-2] * stoch.shape[-1]))
    target = {
        'embed': context['embed'],
        'stoch': stoch,
        'deter': start['deter'],
        'feat': context['feat'],
    }[self.config.disag_target]
    inputs = context['feat']
    if self.config.disag_action_cond:
      action = tf.cast(data['action'], inputs.dtype)
      inputs = tf.concat([inputs, action], -1)
    expl_metrics, training_seq = self.wm_sequence(
        self.wm, start, data['is_terminal'], self._intr_reward)
    ens_mets = self._train_ensemble(inputs, target)
    metrics.update(ens_mets)
    metrics.update(expl_metrics)
    return None, metrics, training_seq
  
  @tf.function
  def wm_s