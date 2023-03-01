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

"""Tests to check whether methods of `mujoco.Physics` are threadsafe."""

import platform

from absl.testing import absltest
from dm_control import _render
from dm_control.mujoco import engine
from dm_control.mujoco.testing import assets
from dm_control.mujoco.testing import decorators


MODEL = assets.get_contents('cartpole.xml')
NUM_STEPS = 10

# Context creation with GLFW is not threadsafe.
if _render.BACKEND == 'glfw':
  # On Linux we are able to create a GLFW window in a single thread that is not
  # the main thread.
  # On Mac we are only allowed to create windows on the main thread, so we
  # disable the `run_threaded` wrapper entirely.
  NUM_THREADS = None if platform.system() == 'Darwin' else 1
else:
  NUM_THREADS = 4


class ThreadSafetyTest(absltest.TestCase):

  @decorators.run_threaded(num_threads=NUM_THREADS)
  def test_load_physics_from_string(self):
    engine.Physics.from_xml_string(MODEL)

  @decorators.run_threaded(num_threads=NUM_THREADS)
  def test_load_and_reload_physics_from_string(self):
    physics = engine.Physics.from_xml_string(MODEL)
    physics.reload_from_xml_string(MODEL)

  @decorators.run_threaded(