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
    'Select