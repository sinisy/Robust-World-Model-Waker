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

"""Tests for randomizers.py."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mujoco
from dm_control.mujoco.wrapper import mjbindings
from envs.dm_control.suite.utils import randomizers
import numpy as np

mjlib = mjbindings.mjlib


class RandomizeUnlimitedJointsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rand = np.random.RandomState(100)

  def test_single_joint_of_each_type(self):
    physics = mujoco.Physics.from_xml_string("""<mujoco>
          <default>
            <joint range="0 90" />
          </default>
          <worldbody>
            <body>
              <geom type="box" size="1 1 1"/>
              <joint name="free" type="free"/>
            </body>
            <body>
              <geom type="box" size="1 1 1"/>
              <joint name="limited_hinge" type="hinge" limited="true"/>
              <joint name="slide" type="slide" limited="false"/>
              <joint name="limited_slide" type="slide" limited="true"/>
              <joint name="hinge" type="hinge" limited="false"/>
            </body>
            <body>
              <geom type="box" size="1 1 1"/>
              <joint name="ball" type="ball" limited="false"/>
            </body>
            <body>
              <geom type="box" size="1 1 1"/>
              <joint name="limited_ball" type="ball" limited="true"/>
            </body>
          </worldbody>
        </mujoco>""")

    randomizers.randomize_limited_and_rotational_joints(physics, self.rand)
    self.assertNotEqual(0., physics.named.data.qpos['hinge'])
    self.assertNotEqual(0., physics.named.data.qpos['limited_hinge'])
    self.assertNotEqual(0., physics.named.data.qpos['limited_slide'])

    self.assertNotEqual(0., np.sum(physics.named.data.qpos['ball']))
    self.assertNotEqual(0., np.sum(physics.named.data.qpos['limited_ball']))

    self.assertNotEqual(0., np.sum(physics.named.data.qpos['free'][3:]))

    # Unlimited slide and the positional part of the free joint remains
    # uninitialized.
    self.assertEqual(0., physics.named.data.qpos['slide'])
    self.assertEqual(0., np.sum(physics.named.data.qpos['free'][:3]))

  def test_multiple_joints_of_same_type(self):
    physics = mujoco.Physics.from_xml_string("""<mujoco>
          <worldbody>
            <body>
              <geom type="box" size="1 1 1"/>
              <joint name="hinge_1" type="hinge"/>
              <joint name="hinge_2" type="hinge"/>
              <joint name="hinge_3" type="hinge"/>
            </body>
          </worldbody>
        </mujoco>""")

    randomizers.randomize_limited_and_rotational_joints(physics, self.rand)
    self.assertNotEqual(0