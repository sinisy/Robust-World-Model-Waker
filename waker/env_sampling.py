
import numpy as np
import tensorflow as tf
import collections
import common


class EnvSampler(object):
  """ Base class for setting the environment parameters.

  Args:
    interval: number of steps between which to estimate the change in uncertainty.
    gamma_uncert_reduct: the discount factor for smoothing the change in uncertainty
  """
  def __init__(self, task, interval=10000, gamma_uncert_reduct=0.95):
    self.interval = interval
    self.gamma_uncert_reduct = gamma_uncert_reduct
    self.gamma_uncert = 1 - 1 / interval # for smoothing the uncertainty estimate
    self.episode_num = 0
    self.task = task

    # initialise buffer of environment params and uncertainty estimates
    self.ens_uncert = collections.OrderedDict()
    self.episodes = collections.OrderedDict()
    self.prev_ens_uncert = collections.OrderedDict()
    self.ens_uncert_change = collections.OrderedDict()
    self.initialised = False
  
  def get_env_params(self):
    """ Set the environment parameters.
    """
    self.episode_num += 1
    env_params = self.sample_env_params()

    # record number of times context has been selected
    if env_params is not None:
      dict_key = common.get_dict_key(env_params)
      self.episodes[dict_key] = 1 + self.episodes.get(dict_key, 0)
    return env_params
  
  def sample_env_params(self):
    """ Sample the environment parameters according to sampling strategy.
    """
    raise NotImplementedError
  
  def sample_env_params_dr(self):
    """ Sample the environment parameters according to the domain randomisation distribution 
    over the environment parameters.
    """
    if "terrain" in self.task and "clean" not in self.task:
      amplitude_range = [0.0, 1.0]
      length_scale_range = [0.2, 2.0]