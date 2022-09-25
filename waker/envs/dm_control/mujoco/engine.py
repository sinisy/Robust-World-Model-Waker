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

"""Mujoco `Physics` implementation and helper classes.

The `Physics` class provides the main Python interface to MuJoCo.

MuJoCo models are defined using the MJCF XML format. The `Physics` class
can load a model from a path to an XML file, an XML string, or from a serialized
MJB binary format. See the named constructors for each of these cases.

Each `Physics` instance defines a simulated world. To step forward the
simulation, use the `step` method. To set a control or actuation signal, use the
`set_control` method, which will apply the provided signal to the actuators in
subsequent calls to `step`.

Use the `Camera` class to create RGB or depth images. A `Camera` can render its
viewport to an array using the `render` method, and can query for objects
visible at specific positions using the `select` method. The `Physics` class
also provides a `render` method that returns a pixel array directly.
"""
import collections
import contextlib
import threading
from typing import Callable, NamedTuple, Optional, Union

from absl import logging

from dm_control import _render
from dm_control.mujoco import index
from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper import util
from dm_control.rl import control as _control
from dm_env import specs
import mujoco
import numpy as np

_FONT_STYLES = {
    'normal': mujoco.mjtFont.mjFONT_NORMAL,
    'shadow': mujoco.mjtFont.mjFONT_SHADOW,
    'big': mujoco.mjtFont.mjFONT_BIG,
}
_GRID_POSITIONS = {
    'top left': mujoco.mjtGridPos.mjGRID_TOPLEFT,
    'top right': mujoco.mjtGridPos.mjGRID_TOPRIGHT,
    'bottom left': mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
    'bottom right': mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
}

Contexts = collections.namedtuple('Contexts', ['gl', 'mujoco'])
Selected = collections.namedtuple(
    'Selected', ['body', 'geom', 'skin', 'world_position'])
NamedIndexStructs = collections.namedtuple(
    'NamedIndexStructs', ['model', 'data'])
Pose = collections.namedtuple(
    'Pose', ['lookat', 'distance', 'azimuth', 'elevation'])

_BOTH_SEGMENTATION_AND_DEPTH_ENABLED = (
    '`segmentation` and `depth` cannot both be `True`.')
_INVALID_PHYSICS_STATE = (
    'Physics state is invalid. Warning(s) raised: {warning_names}')
_OVERLAYS_NOT_SUPPORTED_FOR_DEPTH_OR_SEGMENTATION = (
    'Overlays are not supported with depth or segmentation rendering.')
_RENDER_FLAG_OVERRIDES_NOT_SUPPORTED_FOR_DEPTH_OR_SEGMENTATION = (
    '`render_flag_overrides` are not supported for depth or segmentation '
    'rendering.')
_KEYFRAME_ID_OUT_OF_RANGE = (
    '`keyframe_id` must be between 0 and {max_valid} inclusive, got: {actual}.')


class Physics(_control.Physics):
  """Encapsulates a MuJoCo model.

  A MuJoCo model is typically defined by an MJCF XML file [0]

  ```python
  physics = Physics.from_xml_path('/path/to/model.xml')

  with physics.reset_context():
    physics.named.data.qpos['hinge'] = np.random.rand()

  # Apply controls and advance the simulation state.
  physics.set_control(np.random.random_sample(size=N_ACTUATORS))
  physics.step()

  # Render a camera defined in the XML file to a NumPy array.
  rgb = physics.render(height=240, width=320, id=0)
  ```

  [0] http://www.mujoco.org/book/modeling.html
  """

  _contexts = None

  def __new__(cls, *args, **kwargs):
    # TODO(b/174603485): Re-enable once lint stops spuriously firing here.
    obj = super(Physics, cls).__new__(cls)  # pylint: disable=no-value-for-parameter
    # The lock is created in `__new__` rather than `__init__` because there are
    # a number of existing subclasses that override `__init__` without calling
    # the `__init__` method of the  superclass.
    obj._contexts_lock = threading.Lock()  # pylint: disable=protected-access
    return obj

  def __init__(self, data):
    """Initializes a new `Physics` instance.

    Args:
      data: Instance of `wrapper.MjData`.
    """
    self._warnings_cause_exception = True
    self._reload_from_data(data)

  @contextlib.contextmanager
  def suppress_physics_errors(self):
    """Physics warnings will be logged rather than raise exceptions."""
    prev_state = self._warnings_cause_exception
 