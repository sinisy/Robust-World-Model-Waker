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

"""Decorators used in MuJoCo tests."""

import functools
import threading


def run_threaded(num_threads=4, calls_per_thread=10):
  """A decorator that executes the same test repeatedly in multiple threads.

  Note: `setUp` and `tearDown` methods will only be called once from the main
        thread, so all thread-local setup must be done within the test method.

  Args:
    num_threads: N