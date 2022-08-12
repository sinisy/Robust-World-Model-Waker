import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import common
import expl


class Agent(common.Module):

  def __init__(self, config, obs_space, act_space, step, domain):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.tfstep = tf.Variable(int(self.step), tf.int64)
    self.wm =