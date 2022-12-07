
# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Tests for index."""

import collections

from absl.testing import absltest
from absl.testing import parameterized
from dm_control.mujoco import index
from dm_control.mujoco import wrapper
from dm_control.mujoco.testing import assets
from dm_control.mujoco.wrapper.mjbindings import sizes
import numpy as np

MODEL = assets.get_contents('cartpole.xml')
MODEL_NO_NAMES = assets.get_contents('cartpole_no_names.xml')
MODEL_3RD_ORDER_ACTUATORS = assets.get_contents(
    'model_with_third_order_actuators.xml')

FIELD_REPR = {
    'act': ('FieldIndexer(act):\n'
            '(empty)'),
    'qM': ('FieldIndexer(qM):\n'
           '0  [ 0       ]\n'
           '1  [ 1       ]\n'
           '2  [ 2       ]'),
    'sensordata': ('FieldIndexer(sensordata):\n'
                   '0 accelerometer [ 0       ]\n'
                   '1 accelerometer [ 1       ]\n'
                   '2 accelerometer [ 2       ]\n'
                   '3     collision [ 3       ]'),
    'xpos': ('FieldIndexer(xpos):\n'
             '           x         y         z         \n'
             '0  world [ 0         1         2       ]\n'
             '1   cart [ 3         4         5       ]\n'
             '2   pole [ 6         7         8       ]\n'
             '3 mocap1 [ 9         10        11      ]\n'
             '4 mocap2 [ 12        13        14      ]'),
}


class MujocoIndexTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._model = wrapper.MjModel.from_xml_string(MODEL)
    self._data = wrapper.MjData(self._model)

    self._size_to_axis_indexer = index.make_axis_indexers(self._model)

    self._model_indexers = index.struct_indexer(self._model, 'mjmodel',
                                                self._size_to_axis_indexer)

    self._data_indexers = index.struct_indexer(self._data, 'mjdata',
                                               self._size_to_axis_indexer)

  def assertIndexExpressionEqual(self, expected, actual):
    try:
      if isinstance(expected, tuple):
        self.assertLen(actual, len(expected))
        for expected_item, actual_item in zip(expected, actual):
          self.assertIndexExpressionEqual(expected_item, actual_item)
      elif isinstance(expected, (list, np.ndarray)):
        np.testing.assert_array_equal(expected, actual)
      else:
        self.assertEqual(expected, actual)
    except AssertionError:
      self.fail('Indexing expressions are not equal.\n'
                'expected: {!r}\nactual: {!r}'.format(expected, actual))

  @parameterized.parameters(
      # (field name, named index key, expected integer index key)
      ('actuator_gear', 'slide', 0),
      ('geom_rgba', ('mocap_sphere', 'g'), (6, 1)),
      ('dof_armature', 'slider', slice(0, 1, None)),
      ('dof_armature', ['slider', 'hinge'], [0, 1]),
      ('numeric_data', 'three_numbers', slice(1, 4, None)),
      ('numeric_data', ['three_numbers', 'control_timestep'], [1, 2, 3, 0]))
  def testModelNamedIndexing(self, field_name, key, numeric_key):

    indexer = getattr(self._model_indexers, field_name)
    field = getattr(self._model, field_name)

    converted_key = indexer._convert_key(key)

    # Explicit check that the converted key matches the numeric key.
    converted_key = indexer._convert_key(key)
    self.assertIndexExpressionEqual(numeric_key, converted_key)

    # This writes unique values to the underlying buffer to prevent false
    # negatives.
    field.flat[:] = np.arange(field.size)

    # Check that the result of named indexing matches the result of numeric
    # indexing.
    np.testing.assert_array_equal(field[numeric_key], indexer[key])

  @parameterized.parameters(
      # (field name, named index key, expected integer index key)
      ('xpos', 'pole', 2),
      ('xpos', ['pole', 'cart'], [2, 1]),
      ('sensordata', 'accelerometer', slice(0, 3, None)),
      ('sensordata', 'collision', slice(3, 4, None)),
      ('sensordata', ['accelerometer', 'collision'], [0, 1, 2, 3]),
      # Slices.
      ('xpos', (slice(None), 0), (slice(None), 0)),
      # Custom fixed-size columns.
      ('xpos', ('pole', 'y'), (2, 1)),
      ('xmat', ('cart', ['yy', 'zz']), (1, [4, 8])),
      # Custom indexers for mocap bodies.
      ('mocap_quat', 'mocap1', 0),
      ('mocap_pos', (['mocap2', 'mocap1'], 'z'), ([1, 0], 2)),
      # Two-dimensional named indexing.
      ('xpos', (['pole', 'cart'], ['x', 'z']), ([2, 1], [0, 2])),
      ('xpos', ([['pole'], ['cart']], ['x', 'z']), ([[2], [1]], [0, 2])))
  def testDataNamedIndexing(self, field_name, key, numeric_key):

    indexer = getattr(self._data_indexers, field_name)
    field = getattr(self._data, field_name)

    # Explicit check that the converted key matches the numeric key.
    converted_key = indexer._convert_key(key)
    self.assertIndexExpressionEqual(numeric_key, converted_key)

    # This writes unique values to the underlying buffer to prevent false
    # negatives.
    field.flat[:] = np.arange(field.size)
