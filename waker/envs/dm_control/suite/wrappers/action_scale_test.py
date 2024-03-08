# Copyright 2019 The dm_control Authors.
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

"""Tests for the action scale wrapper."""

from absl.testing import absltest
from absl.testing import parameterized
from envs.dm_control.rl import control
from envs.dm_control.suite.wrappers import action_scale
from dm_env import specs
import mock
import numpy as np


def make_action_spec(lower=(-1.,), upper=(1.,)):
  lower, upper = np.broadcast_arrays(lower, upper)
  return specs.BoundedArray(
      shape=lower.shape, dtype=float, minimum=lower, maximum=upper)


def make_mock_env(action_spec):
  env = mock.Mock(spec=control.Environment)
  env.action_spec.return_value = action_spec
  return env


class ActionScaleTest(parameterized.TestCase):

  def assertStepCalledOnceWithCorrectAction(self, env, expected_action):
    # NB: `assert_called_once_with()` doesn't support numpy arrays.
    env.step.assert_called_once()
    actual_action = env.step.call_args_list[0][0][0]
    np.testing.assert_array_equal(expected_action, actual_action)

  @parameterized.parameters(
      {
          'minimum': np.r_[-1., -1.],
          'maximum': np.r_[1., 1.],
          'scaled_minimum': np.r_[-2., -2.],
          'scaled_maximum': np.r_[2., 2.],
      },
      {
          'minimum': np.r_[-2., -2.],
          'maximum': np.r_[2., 2.],
          'scaled_minimum': np.r_[-1., -1.],
          'scaled_maximum': np.r_[1., 1.],
      },
      {
          'minimum': np.r_[-1., -1.],
          'maximum': np.r_[1., 1.],
          'scaled_minimum': np.r_[-2., -2.],
          'scaled_maximum': np.r_[1., 1.],
      },
      {
          'minimum': np.r_[-1., -1.],
          'maximum': np.r_[1., 1.],
          'scaled_minimum': np.r_[-1., -1.],
          'scaled_maximum': np.r_[2., 2.],
      },
  )
  def test_step(self, minimum, maximum, scaled_minimum, scaled_maximum):
    action_spec = make_action_spec(lower=minimum, upper=maximum)
    env = make_mock_env(action_spec=action_spec)
    wrapped_env = action_scale.Wrapper(
        env, minimum=scaled_minimum, maximum=scaled_maximum)

    time_step = wrapped_env.step(scaled_minimum)
    self.assertStepCalledOnceWithCorrectAction(env, minimum)
    self.assertIs(time_step, env.step(minimum))

    env.reset_mock()

    time_step = wrapped_env.step(scaled_maximum)
    self.assertStepCalledOnceWithCorrectAction(env, maximum)
    self.assertIs(time_step, env.step(maximum))

  @parameterized.parameters(
      {
          'minimum': np.r_[-1., -1.],
          'maximum': np.r_[1., 1.],
      },
      {
          'minimum': np.r_[0, 1],
          'maximum': np.r_[2, 3],
      },
  )
  def test_correct_action_spec(self, minimum, maximum):
    original_action_spec = make_action_spec(
        lower=np.r_[-2., -2.], upper=np.r_[2., 2.])
    env = make_mock_env(action_spec=original_action_spec)
    wrapped_env = action_scale.Wrapper(env, minimum=minimum