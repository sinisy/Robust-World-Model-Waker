import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


# Patch to ignore seed to avoid synchronization across GPUs.
_orig_random_categorical = tf.random.categorical
def random_categorical(*args, **kwargs):
  kwargs['seed'] = None
  return _orig_random_categorical(*args, **kwargs)
tf.random.categorical = random_categorical

# Patch to ignore seed to avoid synchronization across GPUs.
_orig_random_normal = tf.random.normal
def random_normal(*args, **kwargs):
  kwargs['seed'] = None
  return _orig_random_normal(*args, **kwargs)
tf.random.normal = random_normal


class SampleDist:

  def __init__(self, dist, samples=100):
    self._dist = dist
    self._samples = samples

  @pr