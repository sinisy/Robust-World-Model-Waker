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
    # Force loading of the software rendering reference images regardless of the
    # actual GL backend, since these should match the expected RMS values.
    with mock.patch.object(image_utils, 'BACKEND_STRING', new='software'):
      frames = list(image_utils.humanoid.iter_load())
    camera_idx = 0
    num_cameras = image_utils.humanoid.num_cameras
    image1 = frames[camera_idx + frame_index1 * num_cameras]
    image2 = frames[camera_idx + frame_index2 * num_cameras]
    rms = image_utils.compute_rms(image1, image2)
    self.assertAlmostEqual(rms, expected_rms, places=3)

  def test_assert_images_cl