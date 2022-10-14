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
    as a new `Physics` instance created with the `from_xml_string` named
    constructor.

    Args:
      xml_string: XML string containing an MJCF model description.
      assets: Optional dict containing external assets referenced by the model
        (such as additional XML files, textures, meshes etc.), in the form of
        `{filename: contents_string}` pairs. The keys should correspond to the
        filenames specified in the model XML.
    """
    new_model = wrapper.MjModel.from_xml_string(xml_string, assets=assets)
    self._reload_from_model(new_model)

  def reload_from_xml_path(self, file_path):
    """Reloads the `Physics` instance from a path to an MJCF XML file.

    After calling this method, the state of the `Physics` instance is the same
    as a new `Physics` instance created with the `from_xml_path`
    named constructor.

    Args:
      file_path: String containing path to model definition file.
    """
    self._reload_from_model(wrapper.MjModel.from_xml_path(file_path))

  @property
  def named(self):
    return self._named

  def _make_rendering_contexts(self):
    """Creates the OpenGL and MuJoCo rendering contexts."""
    # Get the offscreen framebuffer size, as specified in the model XML.
    max_width = self.model.vis.global_.offwidth
    max_height = self.model.vis.global_.offheight
    # Create the OpenGL context.
    render_context = _render.Renderer(
        max_width=max_width, max_height=max_height)
    # Create the MuJoCo context.
    mujoco_context = wrapper.MjrContext(self.model, render_context)
    self._contexts = Contexts(gl=render_context, mujoco=mujoco_context)

  def _free_rendering_contexts(self):
    """Frees existing OpenGL and MuJoCo rendering contexts."""
    self._contexts.mujoco.free()
    self._contexts.gl.free()
    self._contexts = None

  @property
  def contexts(self):
    """Returns a `Contexts` namedtuple, used in `Camera`s and rendering code."""
    with self._contexts_lock:
      if not self._contexts:
        self._make_rendering_contexts()
    return self._contexts

  @property
  def model(self):
    return self._data.model

  @property
  def data(self):
    return self._data

  def _physics_state_items(self):
    """Returns list of arrays making up internal physics simulation state.

    The physics state consists of the state variables, their derivatives and
    actuation activations. If the model contains plugins, then the state will
    also contain any plugin state.

    Returns:
      List of NumPy arrays containing full physics simulation state.
    """
    if self.model.nplugin > 0:
      return [
          self.data.qpos,
          self.data.qvel,
          self.data.act,
          self.data.plugin_state,
      ]
    else:
      return [self.data.qpos, self.data.qvel, self.data.act]

  # Named views of simulation data.

  def control(self):
    """Returns a copy of the control signals for the actuators."""
    return self.data.ctrl.copy()

  def activation(self):
    """Returns a copy of the internal states of actuators.

    For details, please refer to
    http://www.mujoco.org/book/computation.html#geActuation

    Returns:
      Activations in a numpy array.
    """
    return self.data.act.copy()

  def state(self):
    """Returns the full physics state. Alias for `get_physics_state`."""
    return np.concatenate(self._physics_state_items())

  def position(self):
    """Returns a copy of the generalized positions (system configuration)."""
    return self.data.qpos.copy()

  def velocity(self):
    """Returns a copy of the generalized velocities."""
    return self.data.qvel.copy()

  def timestep(self):
    """Returns the simulation timestep."""
    return self.model.opt.timestep

  def time(self):
    """Returns episode time in seconds."""
    return self.data.time


class CameraMatrices(NamedTuple):
  """Component matrices used to construct the camera matrix.

  The matrix product over these components yields the camera matrix.

  Attributes:
    image: (3, 3) image matrix.
    focal: (3, 4) focal matrix.
    rotation: (4, 4) rotation matrix.
    translation: (4, 4) translation matrix.
  """
  image: np.ndarray
  focal: np.ndarray
  rotation: np.ndarray
  translation: np.ndarray


class Camera:
  """Mujoco scene camera.

  Holds rendering properties such as the width and height of the viewport. The
  camera position and rotation is defined by the Mujoco camera corresponding to
  the `camera_id`. Multiple `Camera` instances may exist for a single
  `camera_id`, for example to render the same view at different resolutions.
  """

  def __init__(
      self,
      physics: Physics,
      height: int = 240,
      width: int = 320,
      camera_id: Union[int, str] = -1,
      max_geom: Optional[int] = None,
      scene_callback: Optional[Callable[[Physics, mujoco.MjvScene],
                                        None]] = None,
  ):
    """Initializes a new `Camera`.

    Args:
      physics: Instance of `Physics`.
      height: Optional image height. Defaults to 240.
      width: Optional image width. Defaults to 320.
      camera_id: Optional camera name or index. Defaults to -1, the free
        camera, which is always defined. A nonnegative integer or string
        corresponds to a fixed camera, which must be defined in the model XML.
        If `camera_id` is a string then the camera must also be named.
      max_geom: Optional integer specifying the maximum number of geoms that can
        be rendered in the same scene. If None this will be chosen automatically
        based on the estimated maximum number of renderable geoms in the model.
      scene_callback: Called after the scene has been created and before
        it is rendered. Can be used to add more geoms to the scene.
    Raises:
      ValueError: If `camera_id` is outside the valid range, or if `width` or
        `height` exceed the dimensions of MuJoCo's offscreen framebuffer.
    """
    buffer_width = physics.model.vis.global_.offwidth
    buffer_height = physics.model.vis.global_.offheight
    if width > buffer_width:
      raise ValueError('Image width {} > framebuffer width {}. Either reduce '
                       'the image width or specify a larger offscreen '
                       'framebuffer in the model XML using the clause\n'
                       '<visual>\n'
                       '  <global offwidth="my_width"/>\n'
                       '</visual>'.format(width, buffer_width))
    if height > buffer_height:
      raise ValueError('Image height {} > framebuffer height {}. Either reduce '
                       'the image height or specify a larger offscreen '
                       'framebuffer in the model XML using the clause\n'
                       '<visual>\n'
                       '  <global offheight="my_height"/>\n'
                       '</visual>'.format(height, buffer_height))
    if isinstance(camera_id, str):
      camera_id = physics.model.name2id(camera_id, 'camera')
    if camera_id < -1:
      raise ValueError('camera_id cannot be smaller than -1.')
    if camera_id >= physics.model.ncam:
      raise ValueError('model has {} fixed cameras. camera_id={} is invalid.'.
                       format(physics.model.ncam, camera_id))

    self._width = width
    self._height = height
    self._physics = physics
    self._scene_callback = scene_callback

    # Variables corresponding to structs needed by Mujoco's rendering functions.
    self._scene = wrapper.MjvScene(model=physics.model, max_geom=max_geom)
    self._scene_option = wrapper.MjvOption()

    self._perturb = wrapper.MjvPerturb()
    self._perturb.active = 0
    self._perturb.select = 0

    self._rect = mujoco.MjrRect(0, 0, self._width, self._height)

    self._render_camera = wrapper.MjvCamera()
    self._render_camera.fixedcamid = camera_id

    if camera_id == -1:
      self._render_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
      mujoco.mjv_defaultFreeCamera(physics.model._model, self._render_camera)
    else:
      # As defined in the Mujoco documentation, mjCAMERA_FIXED refers to a
      # camera explicitly defined in the model.
      self._render_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED

    # Internal buffers.
    self._rgb_buffer = np.empty((self._height, self._width, 3), dtype=np.uint8)
    self._depth_buffer = np.empty((self._height, self._width), dtype=np.float32)

    if self._physics.contexts.mujoco is not None:
      with self._physics.contexts.gl.make_current() as ctx:
        ctx.call(mujoco.mjr_setBuffer, mujoco.mjtFramebuffer.mjFB_OFFSCREEN,
                 self._physics.contexts.mujoco.ptr)

  @property
  def width(self):
    """Returns the image width (number of pixels)."""
    return self._width

  @property
  def height(self):
    """Returns the image height (number of pixels)."""
    return self._height

  @property
  def option(self):
    """Returns the camera's visualization options."""
    return self._scene_option

  @property
  def scene(self):
    """Returns the `mujoco.MjvScene` instance used by the camera."""
    return self._scene

  def matrices(self) -> CameraMatrices:
    """Computes the component matrices used to compute the camera matrix.

    Returns:
      An instance of `CameraMatrices` containing the image, focal, rotation, and
      translation matrices of the camera.
    """
    camera_id = self._render_camera.fixedcamid
    if camera_id == -1:
      # If the camera is a 'free' camera, we get its position and orientation
      # from the scene data structure. It is a stereo camera, so we average over
      # the left and right channels. Note: we call `self.update()` in order to
      # ensure that the contents of `scene.camera` are correct.
      self.update()
      pos = np.mean([camera.pos for camera in self.scene.camera], axis=0)
      z = -np.mean([camera.forward for camera in self.scene.camera], axis=0)
      y = np.mean([camera.up for camera in self.scene.camera], axis=0)
      rot = np.vstack((np.cross(y, z), y, z))
      fov = self._physics.model.vis.global_.fovy
    else:
      pos = self._physics.data.cam_xpos[camera_id]
      rot = self._physics.data.cam_xmat[camera_id].reshape(3, 3).T
      fov = self._physics.model.cam_fovy[camera_id]

    # Translation matrix (4x4).
    translation = np.eye(4)
    translation[0:3, 3] = -pos
    # Rotation matrix (4x4).
    rotation = np.eye(4)
    rotation[0:3, 0:3] = rot
    # Focal transformation matrix (3x4).
    focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * self.height / 2.0
    focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]
    # Image matrix (3x3).
    image = np.eye(3)
    image[0, 2] = (self.wi