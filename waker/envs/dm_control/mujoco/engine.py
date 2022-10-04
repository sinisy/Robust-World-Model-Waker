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
    self._warnings_cause_exception = False
    try:
      yield
    finally:
      self._warnings_cause_exception = prev_state

  def enable_profiling(self):
    """Enables Mujoco timing profiling."""
    wrapper.enable_timer(True)

  def set_control(self, control):
    """Sets the control signal for the actuators.

    Args:
      control: NumPy array or array-like actuation values.
    """
    np.copyto(self.data.ctrl, control)

  def step(self, nstep=1):
    """Advances physics with up-to-date position and velocity dependent fields.

    Args:
      nstep: Optional integer, number of steps to take.

    The actuation can be updated by calling the `set_control` function first.
    """
    # In the case of Euler integration we assume mj_step1 has already been
    # called for this state, finish the step with mj_step2 and then update all
    # position and velocity related fields with mj_step1. This ensures that
    # (most of) mjData is in sync with qpos and qvel. In the case of non-Euler
    # integrators (e.g. RK4) an additional mj_step1 must be called after the
    # last mj_step to ensure mjData syncing.
    with self.check_invalid_state():
      if self.model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4.value:
        mujoco.mj_step2(self.model.ptr, self.data.ptr)
        if nstep > 1:
          mujoco.mj_step(self.model.ptr, self.data.ptr, nstep-1)
      else:
        mujoco.mj_step(self.model.ptr, self.data.ptr, nstep)

      mujoco.mj_step1(self.model.ptr, self.data.ptr)

  def render(
      self,
      height=240,
      width=320,
      camera_id=-1,
      overlays=(),
      depth=False,
      segmentation=False,
      scene_option=None,
      render_flag_overrides=None,
      scene_callback: Optional[Callable[['Physics', mujoco.MjvScene],
                                        None]] = None,
  ):
    """Returns a camera view as a NumPy array of pixel values.

    Args:
      height: Viewport height (number of pixels). Optional, defaults to 240.
      width: Viewport width (number of pixels). Optional, defaults to 320.
      camera_id: Optional camera name or index. Defaults to -1, the free
        camera, which is always defined. A nonnegative integer or string
        corresponds to a fixed camera, which must be defined in the model XML.
        If `camera_id` is a string then the camera must also be named.
      overlays: An optional sequence of `TextOverlay` instances to draw. Only
        supported if `depth` is False.
      depth: If `True`, this method returns a NumPy float array of depth values
        (in meters). Defaults to `False`, which results in an RGB image.
      segmentation: If `True`, this method returns a 2-channel NumPy int32 array
        of label values where the pixels of each object are labeled with the
        pair (mjModel ID, mjtObj enum object type). Background pixels are
        labeled (-1, -1). Defaults to `False`, which returns an RGB image.
      scene_option: An optional `wrapper.MjvOption` instance that can be used to
        render the scene with custom visualization options. If None then the
        default options will be used.
      render_flag_overrides: Optional mapping specifying rendering flags to
        override. The keys can be either lowercase strings or `mjtRndFlag` enum
        values, and the values are the overridden flag values, e.g.
        `{'wireframe': True}` or `{mujoco.mjtRndFlag.mjRND_WIREFRAME: True}`.
        See `mujoco.mjtRndFlag` for the set of valid flags. Must be None if
        either `depth` or `segmentation` is True.
      scene_callback: Called after the scene has been created and before
        it is rendered. Can be used to add more geoms to the scene.

    Returns:
      The rendered RGB, depth or segmentation image.
    """
    camera = Camera(
        physics=self,
        height=height,
        width=width,
        camera_id=camera_id,
        scene_callback=scene_callback)
    image = camera.render(
        overlays=overlays, depth=depth, segmentation=segmentation,
        scene_option=scene_option, render_flag_overrides=render_flag_overrides)
    camera._scene.free()  # pylint: disable=protected-access
    return image

  def get_state(self):
    """Returns the physics state.

    Returns:
      NumPy array containing full physics simulation state.
    """
    return np.concatenate(self._physics_state_items())

  def set_state(self, physics_state):
    """Sets the physics state.

    Args:
      physics_state: NumPy array containing the full physics simulation state.

    Raises:
      ValueError: If `physics_state` has invalid size.
    """
    state_items = self._physics_state_items()

    expected_shape = (sum(item.size for item in state_items),)
    if expected_shape != physics_state.shape:
      raise ValueError('Input physics state has shape {}. Expected {}.'.format(
          physics_state.shape, expected_shape))

    start = 0
    for state_item in state_items:
      size = state_item.size
      np.copyto(state_item, physics_state[start:start + size])
      start += size

  def copy(self, share_model=False):
    """Creates a copy of this `Physics` instance.

    Args:
      share_model: If True, the copy and the original will share a common
        MjModel instance. By default, both model and data will both be copied.

    Returns:
      A `Physics` instance.
    """
    new_data = self.data._make_copy(share_model=share_model)  # pylint: disable=protected-access
    cls = self.__class__
    new_obj = cls.__new__(cls)
    # pylint: disable=protected-access
    new_obj._warnings_cause_exception = True
    new_obj._reload_from_data(new_data)
    # pylint: enable=protected-access
    return new_obj

  def reset(self, keyframe_id=None):
    """Resets internal variables of the simulation, possibly to a keyframe.

    Args:
      keyframe_id: Optional integer specifying the index of a keyframe defined
        in the model XML to which the simulation state should be initialized.
        Must be between 0 and `self.model.nkey - 1` (inclusive).

    Raises:
      ValueError: If `keyframe_id` is out of range.
    """
    if keyframe_id is None:
      mujoco.mj_resetData(self.model.ptr, self.data.ptr)
    else:
      if not 0 <= keyframe_id < self.model.nkey:
        raise ValueError(_KEYFRAME_ID_OUT_OF_RANGE.format(
            max_valid=self.model.nkey-1, actual=keyframe_id))
      mujoco.mj_resetDataKeyframe(self.model.ptr, self.data.ptr, keyframe_id)

    # Disable actuation since we don't yet have meaningful control inputs.
    with self.model.disable('actuation'):
      self.forward()

  def after_reset(self):
    """Runs after resetting internal variables of the physics simulation."""
    # Disable actuation since we don't yet have meaningful control inputs.
    with self.model.disable('actuation'):
      self.forward()

  def forward(self):
    """Recomputes the forward dynamics without advancing the simulation."""
    # Note: `mj_forward` differs from `mj_step1` in that it also recomputes
    # quantities that depend on acceleration (and therefore on the state of the
    # controls). For example `mj_forward` updates accelerometer and gyro
    # readings, whereas `mj_step1` does not.
    # http://www.mujoco.org/book/programming.html#siForward
    with self.check_invalid_state():
      mujoco.mj_forward(self.model.ptr, self.data.ptr)

  @contextlib.contextmanager
  def check_invalid_state(self):
    """Checks whether the physics state is invalid at exit.

    Yields:
      None

    Raises:
      PhysicsError: if the simulation state is invalid at exit, unless this
        context is nested inside a `suppress_physics_errors` context, in which
        case a warning will be logged instead.
    """
    np.copyto(self._warnings_before, self._warnings)
    yield
    np.greater(self._warnings, self._warnings_before, out=self._new_warnings)
    if any(self._new_warnings):
      warning_names = np.compress(self._new_warnings,
                                  list(mujoco.mjtWarning.__members__))
      message = _INVALID_PHYSICS_STATE.format(
          warning_names=', '.join(warning_names))
      if self._warnings_cause_exception:
        raise _control.PhysicsError(message)
      else:
        logging.warn(message)

  def __getstate__(self):
    return self.data  # All state is assumed to reside within `self.data`.

  def __setstate__(self, data):
    # Note: `_contexts_lock` is normally created in `__new__`, but `__new__` is
    #       not invoked during unpickling.
    self._contexts_lock = threading.Lock()
    self._warnings_cause_exception = True
    self._reload_from_data(data)

  def _reload_from_model(self, model):
    """Initializes a new or existing `Physics` from a `wrapper.MjModel`.

    Creates a new `wrapper.MjData` instance, then delegates to
    `_reload_from_data`.

    Args:
      model: Instance of `wrapper.MjModel`.
    """
    data = wrapper.MjData(model)
    self._reload_from_data(data)

  def _reload_from_data(self, data):
    """Initializes a new or existing `Physics` instance from a `wrapper.MjData`.

    Assigns all attributes, sets up named indexing, and creates rendering
    contexts if rendering is enabled.

    The default constructor as well as the other `reload_from` methods should
    delegate to this method.

    Args:
      data: Instance of `wrapper.MjData`.
    """
    if not isinstance(data, wrapper.MjData):
      raise TypeError(f'Expected wrapper.MjData. Got: {type(data)}.')
    self._data = data

    # Performance optimization: pre-allocate numpy arrays used when checking for
    # MuJoCo warnings on each step.
    self._warnings = self.data.warning.number
    self._warnings_before = np.empty_like(self._warnings)
    self._new_warnings = np.empty(dtype=bool, shape=(len(self._warnings),))

    # Forcibly free any previous GL context in order to avoid problems with GL
    # implementations that do not support multiple contexts on a given device.
    with self._contexts_lock:
      if self._contexts:
        self._free_rendering_contexts()

    # Call kinematics update to enable rendering.
    try:
      self.after_reset()
    except _control.PhysicsError as e:
      logging.warning(e)

    # Set up named indexing.
    axis_indexers = index.make_axis_indexers(self.model)
    self._named = NamedIndexStructs(
        model=index.struct_indexer(self.model, 'mjmodel', axis_indexers),
        data=index.struct_indexer(self.data, 'mjdata', axis_indexers),)

  def free(self):
    """Frees the native MuJoCo data structures held by this `Physics` instance.

    This is an advanced feature for use when manual memory management is
    necessary. This `Physics` object MUST NOT be used after this function has
    been called.
    """
    with self._contexts_lock:
      if self._contexts:
        self._free_rendering_contexts()
    del self._data

  @classmethod
  def from_model(cls, model):
    """A named constructor from a `wrapper.MjModel` instance."""
    data = wrapper.MjData(model)
    return cls(data)

  @classmethod
  def from_xml_string(cls, xml_string, assets=None):
    """A named constructor from a string containing an MJCF XML file.

    Args:
      xml_string: XML string containing an MJCF model description.
      assets: Optional dict containing external assets referenced by the model
        (such as additional XML files, textures, meshes etc.), in the form of
        `{filename: contents_string}` pairs. The keys should correspond to the
        filenames specified in the model XML.

    Returns:
      A new `Physics` instance.
    """
    model = wrapper.MjModel.from_xml_string(xml_string, assets=assets)
    return cls.from_model(model)

  @classmethod
  def from_byte_string(cls, byte_string):
    """A named constructor from a model binary as a byte string."""
    model = wrapper.MjModel.from_byte_string(byte_string)
    return cls.from_model(model)

  @classmethod
  def from_xml_path(cls, file_path):
    """A named constructor from a path to an MJCF XML file.

    Args:
      file_path: String containing path to model definition file.

    Returns:
      A new `Physics` instance.
    """
    model = wrapper.MjModel.from_xml_path(file_path)
    return cls.from_model(model)

  @classmethod
  def from_binary_path(cls, file_path):
    """A named constructor from a path to an MJB model binary file.

    Args:
      file_path: String containing path to model definition file.

    Returns:
      A new `Physics` instance.
    """
    model = wrapper.MjModel.from_binary_path(file_path)
    return cls.from_model(model)

  def reload_from_xml_string(self, xml_string, assets=None):
    """Reloads the `Physics` instance from a string containing an MJCF XML file.

    After calling this method, the state of the `Physics` instance is the same
  