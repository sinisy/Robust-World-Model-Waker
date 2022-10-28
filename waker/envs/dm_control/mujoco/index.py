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

The second declaration states that the field `body_pos` has type `mjtNum` and
dimension sizes `(nbody, 3)`, i.e. the first axis is indexed by body number.
These and other named dimensions are sized based on the loaded model. This
information is parsed and stored in `mjbindings.sizes`.

In mjmodel.h, the struct mjModel contains an array of element name addresses
for each size name.

   int* name_bodyadr; // body name pointers (nbody x 1)

By iterating over each of these element name address arrays, we first obtain a
mapping from size names to a list of element names.

    {'nbody': ['cart', 'pole'], 'njnt': ['free', 