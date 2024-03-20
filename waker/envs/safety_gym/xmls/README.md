
# xmls

These are mujoco XML files which are used as bases for the simulations.

Some design goals for them:

- XML should be complete and simulate-able as-is
    - Include a floor geom which is a plane
    - Include joint sensor for the robot which provide observation
    - Include actuators which provide control
- Default positions should all be neutral