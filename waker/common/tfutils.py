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
  base.all = tf.math.re