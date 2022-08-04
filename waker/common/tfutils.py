import pathlib
import pickle
import re

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

try:
  from tensorflow.python.distribute import values
except Exception:
  from google3.third_party.tensorflow.python.distribute import values

tf.tensor = tf.convert_to_tensor
for base in (tf.Tensor, tf.Variable, values.PerReplica):
  base.mean = tf.math.reduce_mean
  base.std = tf.math.reduce_std
  base.var = tf.math.reduce_variance
  base.sum = tf.math.reduce_sum
  base.any = tf.math.reduce_any
  base.all = tf.math.reduce_all
  base.min = tf.math.reduce_min
  base.max = tf.math.reduce_max
  base.abs = tf.math.abs
  base.logsumexp = tf.math.reduce_logsumexp
  base.transpose = tf.transpose
  base.reshape = tf.reshape
  base.astype = tf.cast


# values.PerReplica.dtype = property(lambda self: self.values[0].dtype)

# tf.TensorHandle.__repr__ = lambda x: '<tensor>'
# tf.TensorHandle.__str__ = lambda x: '<tensor>'
# np.set_printoptions(threshold=5, edgeitems=0)


class Module(tf.Module):

  def save(self, filename):
    values = tf.nest.map_structure(lambda x: x.numpy(), sel