# MuJoCo Python bindings

This submodule contains Python bindings for the MuJoCo physics engine. See our
[tech report](https://arxiv.org/abs/1801.00690) for further details.

## Quickstart

```python
from dm_control import mujoco

# Load a model from an MJCF XML string.
xml_string = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1.5"/>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body name="box" pos="0 0 .3">
      <joint name="up_down" type="slide" axis="0 0 1"/>
      <geom name="box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
physics = mujoco.Physics.from_xml_string(xml_string)

# Render the default camera view as a numpy array of pixels.
pixels = physics.render()

# Reset the simulation, move the slide joint upwards and recompute derived
# quantiti