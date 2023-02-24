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

"""Tests for image_utils."""

import os

from absl.testing import absltest
from absl.testing import parameterized
from dm_control.mujoco.testing import image_utils
import mock
import numpy as np
from PIL import Image

SEED = 0


class ImageUtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(frame_index1=0, frame_index2=0, expected_rms=0.0),
      dict(frame_index1=0, frame_index2=1, expected_rms=23.214),
      dict(frame_index1=0, frame_index2=9, expected_rms=55.738))
  def test_compute_rms(self, frame_index1, frame_index2, expected_rms):
    # Force loading 