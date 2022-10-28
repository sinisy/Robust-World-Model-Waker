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

"""Mujoco functions to support named indexing.

The Mujoco name structure works as follows:

In mjxmacro.h, each "X" entry denotes a type (a), a field name (b) and a list
of dimension size metadata (c) which may contain both numbers and names, for
example

   X(int,    name_bodyadr, nbody, 1) // or
   X(mjtNum, body_pos,     nbody, 3)
     a       b             c ----->

The sec