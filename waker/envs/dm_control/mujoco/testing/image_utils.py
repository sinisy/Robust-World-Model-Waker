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
  """A sequence of 