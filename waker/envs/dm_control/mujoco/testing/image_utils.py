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
  """