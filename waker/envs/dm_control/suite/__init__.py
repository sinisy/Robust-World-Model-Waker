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

"""A collection of MuJoCo-based Reinforcement Learning environments."""

import collections
import inspect
import itertools

from envs.dm_control.rl import control

from envs.dm_control.suite import base
from envs.dm_control.suite import terrainwalker
from envs.dm_control.suite import terrainhopper

# Find all domains imported.
_DOMAINS = {name: module for name, module in locals().items()
            if inspect.ismodule(module) and hasattr(module, 'SUITE')}


def _get_tasks(tag):
  """Returns a sequence of (domain name, task name) pairs for the given tag."""
  result = []

  for domain_name in sorted(_DOMAINS.keys()):

    domain = _DOMAINS[domain_name]

    if tag is None:
      tasks_in_domain = domain.SUITE
    else:
      tasks_in_domain = domain.SUITE.tagged(tag)

    for task_name in tasks_in_domain.keys():
      result.append((domain_name, task_name))

  return tuple(result)


def _get_tasks_by_domain(tasks):
  """Returns a dict mapping from task name to a tuple of domain names."""
  result = collections.defaultdict(list)

  for domain_name, task_name in tasks:
    result[domain_name].append(task_name)

  return {k: tuple(v) for k, v in result.items()}


# 