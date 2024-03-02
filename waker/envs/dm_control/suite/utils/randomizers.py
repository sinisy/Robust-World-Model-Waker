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

"""Randomization functions."""


from dm_control.mujoco.wrapper import mjbindings
import numpy as np


def random_limited_quaternion(random, limit):
  """Generates a random quaternion limited to the specified rotations."""
  axis = random.randn(3)
  axis /= np.linalg.norm(axis)
  angle = random.rand() * limit

  quaternion = np.zeros(4)
  mjbindings.mjlib.mju_axisAngle2Quat(quaternion, axis, angle)

  return quaternion


def randomize_limited_and_rotational_joints(physics, random=None):
  """Randomizes the positions of joints defined in the physics body.

  The following randomization rules apply:
    - Bounded joints (hinges or sliders) are sampled uniformly in the bounds.
    - Unbounded hinges are samples uniformly in [-pi, pi]
    - Quaternions for unlimited free joints and ball joints are sampled
      uniformly on the unit 3-sphere.
    - Quaternions for limited ball joints are sampled uniformly on a sector
      of the unit 3-sphere.
    - The linear degrees of freedom of free joints are not randomized.

  Args:
    physics: Instance of 'Physics' class that holds a loaded model.
    random: Optional instance of 'np.random.RandomState'. Defaults to the global
      NumPy random state.
  """
  random = random or np.random

  hinge = mjb