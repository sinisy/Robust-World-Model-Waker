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

"""Utilities for testing rendering."""

import collections
import functools
import io
import os
import sys
from dm_control import _render
from dm_control import mujoco
from dm_control.mujoco.testing import assets
import numpy as np
from PIL import Image


BACKEND_STRING = 'hardware' if _render.USING_GPU else 'software'


class ImagesNotCloseError(AssertionError):
  """Exception raised when two images are not sufficiently close."""

  def __init__(self, message, expected, actual):
    super().__init__(message)
    self.expected = expected
    self.actual = actual


_CameraSpec = collections.namedtuple(
    '_CameraSpec', ['height', 'width', 'camera_id', 'render_flag_overrides'])


_SUBDIR_TEMPLATE = (
    '{name}_seed_{seed}_camera_{camera_id}_{width}x{height}_{backend_string}'
    '{render_flag_overrides_string}'
)


def _get_subdir(name, seed, backend_string, camera_spec):
  if camera_spec.render_flag_overrides:
    overrides = ('{}_{}'.format(k, v) for k, v in
                 sorted(camera_spec.render_flag_overrides.items()))
    render_flag_overrides_string = '_' + '_'.join(overrides)
  else:
    render_flag_overrides_string = ''
  return _SUBDIR_TEMPLATE.format(
      name=name,
      seed=seed,
      camera_id=camera_spec.camera_id,
      width=camera_spec.width,
      height=camera_spec.height,
      backend_string=backend_string,
      render_flag_overrides_string=render_flag_overrides_string,
  )


class _FrameSequence:
  """A sequence of pre-rendered frames used in integration tests."""

  _ASSETS_DIR = 'assets'
  _FRAMES_DIR = 'frames'
  _FILENAME_TEMPLATE = 'frame_{frame_num:03}.png'

  def __init__(self,
               name,
               xml_string,
               camera_specs,
               num_frames=20,
               steps_per_frame=10,
               seed=0):
    """Initializes a new `_FrameSequence`.

    Args:
      name: A string containing the name to be used for the sequence.
      xml_string: An MJCF XML string containing the model to be rendered.
      camera_specs: A list of `_CameraSpec` instances specifying the cameras to
        render on each frame.
      num_frames: The number of frames to render.
      steps_per_frame: The interval between frames, in simulation steps.
      seed: Integer or None, used to initialize the random number generator for
        generating actions.
    """
    self._name = name
    self._xml_string = xml_string
    self._camera_specs = camera_specs
    self._num_frames = num_frames
    self._steps_per_frame = steps_per_frame
    self._seed = seed

  @property
  def num_cameras(self):
    return len(self._camera_specs)

  def iter_render(self):
    """Returns an iterator that yields newly rendered frames as numpy arrays."""
    random_state = np.random.RandomState(self._seed)
    physics = mujoco.Physics.from_xml_string(self._xml_string)
    action_spec = mujoco.action_spec(physics)
    for _ in range(self._num_frames):
      for _ in range(self._steps_per_frame):
        actions = random_state.uniform(action_spec.minimum, action_spec.maximum)
        physics.set_control(actions)
        physics.step()
      for camera_spec in self._camera_specs:
        yield physics.render(**camera_spec._asdict())

  def iter_load(self):
    """Returns an iterator that yields saved frames as numpy arrays."""
    for directory, filename in self._iter_paths():
      path = os.path.join(directory, filename)
      yield _load_pixels(path)

  def save(self):
    """Saves a new set of golden output frames to disk."""
    for pixels, (relative_to_assets, filename) in zip(self.iter_render(),
                                                      self._iter_paths()):
      full_directory_path = os.path.join(self._ASSETS_DIR, relative_to_assets)
      if not os.path.exists(full_directory_path):
        os.makedirs(full_directory_path)
      path = os.path.join(full_directory_path, filename)
      _save_pixels(pixels, path)

  def _iter_paths(self):
    """Returns an iterator over paths to the reference images."""
    for frame_num in range(self._num_frames):
      filename = self._FILENAME_TEMPLATE.format(frame_num=frame_num)
      for camera_spec in self._camera_specs:
        subdir_name = _get_subdir(
            name=self._name,
            seed=self._seed,
            backend_string=BACKEND_STRING,
            camera_spec=camera_spec)
        directory = os.path.join(self._FRAMES_DIR, subdir_name)
  