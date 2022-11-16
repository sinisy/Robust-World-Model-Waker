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
  'pole'.
* _element index_ or _index_ The index of an element name, for a specific size
  name.
"""

import abc
import collections
import weakref

from dm_control.mujoco.wrapper import util
from dm_control.mujoco.wrapper.mjbindings import sizes
import numpy as np


# Mapping from {size_name: address_field_name} for ragged dimensions.
_RAGGED_ADDRS = {
    'nq': 'jnt_qposadr',
    'nv': 'jnt_dofadr',
    'na': 'actuator_actadr',
    'nsensordata': 'sensor_adr',
    'nnumericdata': 'numeric_adr',
}

# Names of columns.
_COLUMN_NAMES = {
    'xyz': ['x', 'y', 'z'],
    'quat': ['qw', 'qx', 'qy', 'qz'],
    'mat': ['xx', 'xy', 'xz',
            'yx', 'yy', 'yz',
            'zx', 'zy', 'zz'],
    'rgba': ['r', 'g', 'b', 'a'],
}

# Mapping from keys of _COLUMN_NAMES to sets of field names whose columns are
# addressable using those names.
_COLUMN_ID_TO_FIELDS = {
    'xyz': set([
        'body_pos',
        'body_ipos',
        'body_inertia',
        'jnt_pos',
        'jnt_axis',
        'geom_size',
        'geom_pos',
        'site_size',
        'site_pos',
        'cam_pos',
        'cam_poscom0',
        'cam_pos0',
        'light_pos',
        'light_dir',
        'light_poscom0',
        'light_pos0',
        'light_dir0',
        'mesh_vert',
        'mesh_normal',
        'mocap_pos',
        'xpos',
        'xipos',
        'xanchor',
        'xaxis',
        'geom_xpos',
        'site_xpos',
        'cam_xpos',
        'light_xpos',
        'light_xdir',
        'subtree_com',
        'wrap_xpos',
        'subtree_linvel',
        'subtree_angmom',
    ]),
    'quat': set([
        'body_quat',
        'body_iquat',
        'geom_quat',
        'site_quat',
        'cam_quat',
        'mocap_quat',
        'xquat',
    ]),
    'mat': set([
        'cam_mat0',
        'xmat',
        'ximat',
        'geom_xmat',
        'site_xmat',
        'cam_xmat',
    ]),
    'rgba': set([
        'geom_rgba',
        'site_rgba',
        'skin_rgba',
        'mat_rgba',
        'tendon_rgba',
    ])
}


def _get_size_name_to_element_names(model):
  """Returns a dict that maps size names to element names.

  Args:
    model: An instance of `mjbindings.mjModelWrapper`.

  Returns:
    A `dict` mapping from a size name (e.g. `'nbody'`) to a list of element
    names.
  """

  names = model.names
  size_name_to_element_names = {}

  for field_name in dir(model.ptr):
    if not _is_name_pointer(field_name):
      continue

    # Get addresses of element names in `model.names` array, e.g.
    # field name: `name_nbodyadr` and name_addresses: `[86, 92, 101]`, and skip
    # when there are no elements for this type in the model.
    name_addresses = getattr(model, field_name).ravel()
    if not name_addresses.size:
      continue

    # Get the element names.
    element_names = []
    for start_index in name_addresses:
      end_index = names.find(b'\0', start_index)
      name = names[start_index:end_index]
      element_names.append(str(name, 'utf-8'))

    # String identifier for the size of the first dimension, e.g. 'nbody'.
    size_name = _get_size_name(field_name)

    size_name_to_element_names[size_name] = element_names

  # Add custom element names for certain columns.
  for size_name, element_names in _COLUMN_NAMES.items():
    size_name_to_element_names[size_name] = element_names

  # "Ragged" axes inherit their element names from other "non-ragged" axes.
  # For example, the element names for "nv" axis come from "njnt".
  for size_name, address_field_name in _RAGGED_ADDRS.items():
    donor = 'n' + address_field_name.split('_')[0]
    if donor == 'nactuator':
      donor = 'nu'
    if donor in size_name_to_element_names:
      size_name_to_element_names[size_name] = size_name_to_element_names[donor]

  # Mocap bodies are a special subset of bodies.
  mocap_body_names = [None] * model.nmocap
  for body_id, body_name in enumerate(size_name_to_element_names['nbody']):
    body_mocapid = model.body_mocapid[body_id]
    if body_mocapid != -1:
      mocap_body_names[body_mocapid] = body_name
  assert None not in mocap_body_names
  size_name_to_element_names['nmocap'] = mocap_body_names

  return size_name_to_element_names


def _get_size_name_to_element_sizes(model):
  """Returns a dict that maps size names to element sizes for ragged axes.

  Args:
    model: An instance of `mjbindings.mjModelWrapper`.

  Returns:
    A `dict` mapping from a size name (e.g. `'nv'`) to a numpy array of element
      sizes. Size names corresponding to non-ragged axes are omitted.
  """

  size_name_to_element_sizes = {}

  for size_name, address_field_name in _RAGGED_ADDRS.items():
    addresses = getattr(model, address_field_name).ravel()
    if size_name == 'na':
      element_sizes = np.where(addresses == -1, 0, 1)
    else:
      total_length = getattr(model, size_name)
      element_sizes = np.diff(np.r_[addresses, total_length])
    size_name_to_element_sizes[size_name] = element_sizes

  return size_name_to_element_sizes


def make_axis_indexers(model):
  """Returns a dict that maps size names to `Axis` indexers.

  Args:
    model: An instance of `mjbindings.MjModelWrapper`.

  Returns:
    A `dict` mapping from a size name (e.g. `'nbody'`) to an `Axis` instance.
  """

  size_name_to_element_names = _get_size_name_to_element_names(model)
  size_name_to_element_sizes = _get_size_name_to_element_sizes(model)

  # Unrecognized size names are treated as unnamed axes.
  axis_indexers = collections.defaultdict(UnnamedAxis)

  for size_name in size_name_to_element_names:
    element_names = size_name_to_element_names[size_name]
    if size_name in _RAGGED_ADDRS:
      element_sizes = size_name_to_element_sizes[size_name]
      singleton = (size_name == 'na')
      indexer = RaggedNamedAxis(element_names, element_sizes,
                                singleton=singleton)
    else:
      indexer = RegularNamedAxis(element_names)
    axis_indexers[size_name] = indexer

  return axis_indexers


def _is_name_pointer(field_name):
  """Returns True for name pointer field names such as `name_bodyadr`."""
  # Denotes name pointer fields in mjModel.
  prefix, suffix = 'name_', 'adr'
  return field_name.startswith(prefix) and field_name.endswith(suffix)


def _get_size_name(field_name, struct_name='mjmodel'):
  # Look up size name in metadata.
  return sizes.array_sizes[struct_name][field_name][0]


def _validate_key_item(key_item):
  if isinstance(key_item, (list, np.ndarray)):
    for sub in key_item:
      _validate_key_item(sub)  # Recurse into nested arrays and lists.
  elif key_item is Ellipsis:
    raise IndexError('Ellipsis indexing not supported.')
  elif key_item is None:
    raise IndexError('None indexing not supported.')
  elif key_item in (b'', u''):
    raise IndexError('Empty strings are not allowed.')


class Axis(metaclass=abc.ABCMeta):
  """Handles the conversion of named indexing expressions into numpy indices."""

  @abc.abstractmethod
  def convert_key_item(self, key_item):
    """Converts a (possibly named) indexing expression to a numpy index."""


class UnnamedAxis(Axis):
  """An object representing an axis where the elements are not named."""

  def convert_key_item(self, key_item):
    """Validate the indexing expression and return it unmodified."""
    _validate_key_item(key_item)
    return key_item


class RegularNamedAxis(Axis):
  """Represents an axis where each named element has a fixed size of 1."""

  def __init__(self, names):
    """Initializes a new `RegularNamedAxis` instance.

    Args:
      names: A list or array of element names.
    """
    self._names