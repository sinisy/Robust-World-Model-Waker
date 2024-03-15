
#!/usr/bin/env python

import os
import xmltodict
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from mujoco_py import const, load_model_from_path, load_model_from_xml, MjSim, MjViewer, MjRenderContextOffscreen

import envs.safety_gym
import sys

'''
Tools that allow the Safety Gym Engine to interface to MuJoCo.

The World class owns the underlying mujoco scene and the XML,
and is responsible for regenerating the simulator.

The way to use this is to configure a World() based on your needs 
(number of objects, etc) and then call `world.reset()`.

*NOTE:* The simulator should be accessed as `world.sim` and not just
saved separately, because it may change between resets.

Configuration is idiomatically done through Engine configuration,
so any changes to this configuration should also be reflected in 
changes to the Engine.

TODO:
- unit test scaffold
'''

# Default location to look for /xmls folder:
BASE_DIR = os.path.dirname(envs.safety_gym.__file__)


def convert(v):
    ''' Convert a value into a string for mujoco XML '''
    if isinstance(v, (int, float, str)):
        return str(v)
    # Numpy arrays and lists
    return ' '.join(str(i) for i in np.asarray(v))


def rot2quat(theta):
    ''' Get a quaternion rotated only about the Z axis '''
    return np.array([np.cos(theta / 2), 0, 0, np.sin(theta / 2)], dtype='float64')


class World:
    # Default configuration (this should not be nested since it gets copied)
    # *NOTE:* Changes to this configuration should also be reflected in `Engine` configuration
    DEFAULT = {
        'robot_base': 'xmls/car.xml',  # Which robot XML to use as the base
        'robot_xy': np.zeros(2),  # Robot XY location
        'robot_rot': 0,  # Robot rotation about Z axis

        'floor_size': [3.5, 3.5, .1],  # Used for displaying the floor

        # Objects -- this is processed and added by the Engine class