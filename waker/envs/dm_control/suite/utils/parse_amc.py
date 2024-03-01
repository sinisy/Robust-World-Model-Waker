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

"""Parse and convert amc motion capture data."""

import collections

from dm_control.mujoco import math as mjmath
import numpy as np
from scipy import interpolate

MOCAP_DT = 1.0/120.0
CONVERSION_LENGTH = 0.056444

_CMU_MOCAP_JOINT_ORDER = (
    'root0', 'root1', 'root2', 'root3', 'root4', 'root5', 'lowerbackrx',
    'lowerbackry', 'lowerbackrz', 'upperbackrx', 'upperbackry', 'upperbackrz',
    'thoraxrx', 'thoraxry', 'thoraxrz', 'lowerneckrx', 'lowerneckry',
    'lowerneckrz', 'upperneckrx', 'upperneckry', 'upperneckrz', 'headrx',
    'headry', 'headrz', 'rclaviclery', 'rclaviclerz', 'rhumerusrx',
    'rhumerusry', 'rhumerusrz', 'rradiusrx', 'rwristry', 'rhandrx', 'rhandrz',
    'rfingersrx', 'rthumbrx', 'rthumbrz', 'lclaviclery', 'lclaviclerz',
    'lhumerusrx', 'lhumerusry', 'lhumerusrz', 'lradiusrx', 'lwristry',
    'lhandrx', 'lhandrz', 'lfingersrx', 'lthumbrx', 'lthumbrz', 'rfemurrx',
    'rfemurry', 'rfemurrz', 'rtibiarx', 'rfootrx', 'rfootrz', 'rtoesrx',
    'lfemurrx', 'lfemurry', 'lfemurrz', 'ltibiarx', 'lfootrx', 'lfootrz',
    'ltoesrx'
)

Converted = collections.namedtuple('Converted',
                                   ['qpos', 'qvel', 'time'])


def convert(file_name, physics, timestep):
  """Converts the parsed .amc values into qpos and qvel values and resamples.

  Args:
    file_name: The .amc file to be parsed and converted.
    physics: The corresponding physics instance.
    timestep: Desired output interval between resampled frames.

  Returns:
    A namedtuple with fields:
        `qpos`, a numpy array containing converted positional variables.
        `qvel`, a numpy array containing converted velocity variables.
        `time`, a numpy array containing the corresponding times.
  """
  frame_values = parse(file_name)
  joint2index = {}
  for name in physics.named.data.qpos.axes.row.names:
    joint2index[name] = physics.named.data.qpos.axes.row.convert_key_item(name)
  index2joint = {}
  for joint, index in joint2index.items():
    if isinstance(index, slice):
      indices = range(index.start, index.stop)
    else:
      indices = [index]
    for ii in indices:
      index2joint[ii] = joint

  # Convert frame_values to qpos
  amcvals2qpos_transformer = Amcvals2qpos(index2joint, _CMU_MOCAP_JOINT_ORDER)
  qpos_values = []
  for frame_value in frame_values:
    qpos_values.append(amcva