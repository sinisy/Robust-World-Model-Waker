
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

"""Tests for the pixel wrapper."""

import collections

from absl.testing import absltest
from absl.testing import parameterized
from envs.dm_control.suite import cartpole
from envs.dm_control.suite.wrappers import pixels
import dm_env
from dm_env import specs
import numpy as np


class FakePhysics:

  def render(self, *args, **kwargs):