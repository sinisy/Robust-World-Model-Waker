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

  @property
  def name(self):
    return 'SampleDist'

  def __getattr__(self, name):
    return getattr(self._dist, name)

  def mean(self):
    samples = self._dist.sample(self._samples)
    return samples.mean(0)

  def mode(self):
    sample = self._dist.sample(self._samples)
    logprob = self._dist.log_prob(sample)
    return tf.gather(sample, tf.argmax(logprob))[0]

  def entropy(self):
    sample = self._dist.sample(self._samples)
    logprob = self.log_prob(sample)
    return -logprob.mean(0)


class OneHotDist(tfd.OneHotCategorical):

  def __init__(self, logits=None, probs=None, dtype=None):
    self._sample_dtype = dtype or tf.float32
    super().__init__(logits=logits, probs=probs)

  def mode(self):
    return tf.cast(super().mode(), self._sample_dtype)

  def sample(self, sample_shape=(), seed=None):
    # Straight through biased gradient estimator.
    sample = tf.cast(super().sample(sample_shape, seed), self._sample_dtype)
    probs = self._pad(super().probs_parameter(), sample.shape)
    sample += tf.cast(probs - tf.stop_gradient(probs), self._sample_dtype)
    return sample

  def _pad(self, tensor, shape):
    tensor = super().pr