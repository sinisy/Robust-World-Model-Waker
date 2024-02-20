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

"""Control Environment tests."""

from absl.testing import absltest
from absl.testing import parameterized
from envs.dm_control.rl import control
from dm_env import specs
import mock
import numpy as np

_CONSTANT_REWARD_VALUE = 1.0
_CONSTANT_OBSERVATION = {'observations': np.asarray(_CONSTANT_REWARD_VALUE)}

_ACTION_SPEC = specs.BoundedArray(
    shape=(1,), dtype=float, minimum=0.0, maximum=1.0)
_OBSERVATION_SPEC = {'observations': specs.Array(shape=(), dtype=float)}


class EnvironmentTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._task = mock.Mock(spec=control.Task)
    self._task.initialize_episode = mock.Mock()
    self._task.get_observation = mock.Mock(return_value=_CONSTANT_OBSERVATION)
    self._task.get_reward = mock.Mock(return_value=_CONSTANT_REWARD_VALUE)
    self._task.get_termination = mock.Mock(return_value=None)
    self._task.action_spec = mock.Mock(return_value=_ACTION_SPEC)
    self._task.observation_spec.side_effect = NotImplementedError()
