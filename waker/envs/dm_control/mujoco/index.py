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

    {'nbody': ['cart', 'pole'], 'njnt': ['free', 'ball', 'hinge'], ...}

In addition to the element names that are derived from the mjModel struct at
runtime, we also assign hard-coded names to certain dimensions where there is an
established naming convention (e.g. 'x', 'y', 'z' for dimensions that correspond
to Cartesian positions).

For some dimensions, a single element name maps to multiple indices within the
underlying field. For example, a single joint name corresponds to a variable
number of indices within `qpos` that depends on the number of degrees of freedom
associated with that joint type. These are referred to as "ragged" dimensions.

In such cases we determine the size of each named element by examining the
address arrays (e.g. `jnt_qposadr`), and construct a mapping from size name to
element sizes:

    {'nq': [7, 3, 1], 'nv': [6, 3, 1], ...}

Given these two dictionaries, we then create an `Axis` instance for each size
name. These objects have a `convert_key_item` method that handles the conversion
from indexing expressions containing element names to valid numpy indices.
Different implementations of `Axis` are used to handle "ragged" and "non-ragged"
dimensions.

    {'nbody': RegularNamedAxis(names=['cart', 'pole']),
     'nq': RaggedNamedAxis(names=['free', 'ball', 'hinge'], sizes=[7, 4, 1])}

We construct this dictionary once using `make_axis_indexers`.

Finally, for each field we construct a `FieldIndexer` class. A `FieldIndexer`
instance encapsulates a field together with a list of `Axis` instances (one per
dimension), and implements the named indexing logic by calling their respective
`convert_key_item` methods.

Summary of terminology:

* _size name_ or _size_ A dimension size name, e.g. `nbody` or `ngeom`.
* _element name_ or _name_ A named element in a Mujoco model, e.g. 'cart' or
  'pole