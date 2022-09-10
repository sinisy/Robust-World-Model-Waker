# MuJoCo Python bindings

This submodule contains Python bindings for the MuJoCo physics engine. See our
[tech report](https://arxiv.org/abs/1801.00690) for further details.

## Quickstart

```python
from dm_control import mujoco

# Load a model from an MJCF XML string.
xml_string = """
<mujoco>
  <world