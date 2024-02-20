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
_OBSERVATION_SPEC = {'observations': specs.Array(shape