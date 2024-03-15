
#!/usr/bin/env python

import gym
import gym.spaces
import numpy as np
import itertools
import common
from PIL import Image
from copy import deepcopy
from collections import OrderedDict
import mujoco_py
from mujoco_py import MjViewer, MujocoException, const, MjRenderContextOffscreen

from envs.safety_gym.envs.world import World, Robot

import sys


# Distinct colors for different types of objects.
# For now this is mostly used for visualization.
# This also affects the vision observation, so if training from pixels.
COLOR_BOX = np.array([1, 1, 0, 1])
COLOR_BUTTON = np.array([1, .5, 0, 1])
COLOR_GOAL = np.array([0, 1, 0, 1])
COLOR_VASE = np.array([0, 1, 1, 1])
COLOR_HAZARD = np.array([0, 0, 1, 1])
COLOR_PILLAR = np.array([0, 0, 0, 1])
COLOR_WALL = np.array([.5, .5, .5, 1])
COLOR_GREMLIN = np.array([0.5, 0, 1, 1])
COLOR_CIRCLE = np.array([0, 1, 0, 1])
COLOR_RED = np.array([1, 0, 0, 1])
COLOR_GREEN = np.array([0, 1, 0, 1])
COLOR_BLUE = np.array([0, 0, 1, 1])
COLOR_YELLOW = np.array([1, 1, 0, 1])
COLOR_PURPLE = np.array([0.5, 0, 1, 1])
COLOR_CYAN = np.array([0, 1, 1, 1])

# Groups are a mujoco-specific mechanism for selecting which geom objects to "see"
# We use these for raycasting lidar, where there are different lidar types.
# These work by turning "on" the group to see and "off" all the other groups.
# See obs_lidar_natural() for more.
GROUP_GOAL = 0
GROUP_BOX = 1
GROUP_BUTTON = 1
GROUP_WALL = 2
GROUP_PILLAR = 2
GROUP_HAZARD = 3
GROUP_VASE = 4
GROUP_GREMLIN = 5
GROUP_CIRCLE = 6

# Constant for origin of world
ORIGIN_COORDINATES = np.zeros(3)

# Constant defaults for rendering frames for humans (not used for vision)
DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 256

class ResamplingError(AssertionError):
    ''' Raised when we fail to sample a valid distribution of objects or goals '''
    pass


def theta2vec(theta):
    ''' Convert an angle (in radians) to a unit vector in that angle around Z '''
    return np.array([np.cos(theta), np.sin(theta), 0.0])


def quat2mat(quat):
    ''' Convert Quaternion to a 3x3 Rotation Matrix using mujoco '''
    q = np.array(quat, dtype='float64')
    m = np.zeros(9, dtype='float64')
    mujoco_py.functions.mju_quat2Mat(m, q)
    return m.reshape((3,3))


def quat2zalign(quat):
    ''' From quaternion, extract z_{ground} dot z_{body} '''
    # z_{body} from quaternion [a,b,c,d] in ground frame is:
    # [ 2bd + 2ac,
    #   2cd - 2ab,
    #   a**2 - b**2 - c**2 + d**2
    # ]
    # so inner product with z_{ground} = [0,0,1] is
    # z_{body} dot z_{ground} = a**2 - b**2 - c**2 + d**2
    a, b, c, d = quat
    return a**2 - b**2 - c**2 + d**2


class Engine(gym.Env, gym.utils.EzPickle):

    '''
    Engine: an environment-building tool for safe exploration research.

    The Engine() class entails everything to do with the tasks and safety 
    requirements of Safety Gym environments. An Engine() uses a World() object
    to interface to MuJoCo. World() configurations are inferred from Engine()
    configurations, so an environment in Safety Gym can be completely specified
    by the config dict of the Engine() object.

    '''

    # Default configuration (this should not be nested since it gets copied)
    DEFAULT = {
        'num_steps': 1000,  # Maximum number of environment steps in an episode
        'return_scale': None, # total reward for perfect episode
        'action_noise': 0.0,  # Magnitude of independent per-component gaussian action noise

        'placements_extents': [-2, -2, 2, 2],  # Placement limits (min X, min Y, max X, max Y)
        'placements_margin': 0.0,  # Additional margin added to keepout when placing objects

        # Floor
        'floor_display_mode': False,  # In display mode, the visible part of the floor is cropped

        # Robot
        'robot_placements': None,  # Robot placements list (defaults to full extents)
        'robot_locations': [],  # Explicitly place robot XY coordinate
        'robot_keepout': 0.2,  # Needs to be set to match the robot XML used
        'robot_base': 'xmls/car.xml',  # Which robot XML to use as the base
        'robot_rot': None,  # Override robot starting angle

        # Starting position distribution
        'randomize_layout': True,  # If false, set the random seed before layout to constant
        'build_resample': True,  # If true, rejection sample from valid environments
        'continue_goal': True,  # If true, draw a new goal after achievement
        'terminate_resample_failure': True,  # If true, end episode when resampling fails,
                                             # otherwise, raise a python exception.
        # TODO: randomize starting joint positions

        # Observation flags - some of these require other flags to be on
        # By default, only robot sensor observations are enabled.
        'observation_flatten': True,  # Flatten observation into a vector
        'observe_sensors': True,  # Observe all sensor data from simulator
        'observe_goal_dist': False,  # Observe the distance to the goal
        'observe_goal_comp': False,  # Observe a compass vector to the goal
        'observe_goal_lidar': False,  # Observe the goal with a lidar sensor
        'observe_box_comp': False,  # Observe the box with a compass
        'observe_box_lidar': False,  # Observe the box with a lidar
        'observe_circle': False,  # Observe the origin with a lidar
        'observe_remaining': False,  # Observe the fraction of steps remaining
        'observe_walls': False,  # Observe the walls with a lidar space
        'observe_hazards': False,  # Observe the vector from agent to hazards
        'observe_vases': False,  # Observe the vector from agent to vases
        'observe_pillars': False,  # Lidar observation of pillar object positions
        'observe_buttons': False,  # Lidar observation of button object positions
        'observe_gremlins': False,  # Gremlins are observed with lidar-like space
        'observe_vision': False,  # Observe vision from the robot
        # These next observations are unnormalized, and are only for debugging
        'observe_qpos': False,  # Observe the qpos of the world
        'observe_qvel': False,  # Observe the qvel of the robot
        'observe_ctrl': False,  # Observe the previous action
        'observe_freejoint': False,  # Observe base robot free joint
        'observe_com': False,  # Observe the center of mass of the robot

        # Render options
        'render_labels': False,
        'render_lidar_markers': True,
        'render_lidar_radius': 0.15, 
        'render_lidar_size': 0.025, 
        'render_lidar_offset_init': 0.5, 
        'render_lidar_offset_delta': 0.06, 

        # Vision observation parameters
        'vision_size': (60, 40),  # Size (width, height) of vision observation; gets flipped internally to (rows, cols) format
        'vision_render': True,  # Render vision observation in the viewer
        'vision_render_size': (300, 200),  # Size to render the vision in the viewer

        # Lidar observation parameters
        'lidar_num_bins': 10,  # Bins (around a full circle) for lidar sensing
        'lidar_max_dist': None,  # Maximum distance for lidar sensitivity (if None, exponential distance)
        'lidar_exp_gain': 1.0, # Scaling factor for distance in exponential distance lidar
        'lidar_type': 'pseudo',  # 'pseudo', 'natural', see self.obs_lidar()
        'lidar_alias': True,  # Lidar bins alias into each other

        # Compass observation parameters
        'compass_shape': 2,  # Set to 2 or 3 for XY or XYZ unit vector compass observation.

        # Task
        'task': 'cleanup_all',  # goal, button, push, x, z, circle, or none (for screenshots)

        # parameters for cleanup task
        'sample_blueobjs': False,
        'sample_greenobjs': False,
        'blueobjs_num': 0,
        'greenobjs_num': 0,
        'blueobjs_keepout': 0.25,
        'greenobjs_keepout': 0.25,
        'right_goal_locs': [[-2, 2, -1, 2]], # Goal location for blue objects (min X, min Y, max X, max Y)
        'left_goal_locs': [[1, -2, 2, 2]], # Goal location for green objects (min X, min Y, max X, max Y)
        'blueobjs_size': 0.2,
        'greenobjs_size': 0.2,
        'reward_task_completion': 0.5,

        # Params for domain randomisation 
        'sample_num_objects': False,  # whether to sample the number of objects between 0 and max.
        'sample_objects_prob': 1.0,  # probability of adding objects to environment
        'sample_arenas': False,
        'max_object_num': 0,
        'arenas_max_object_nums': [0],
        'arenas': [0],
        'arenas_to_sample': [0],
        'arena_sizes': [0.0],
        'arenas_placements_extents': [[-2, -2, 2, 2]],
        'arenas_goal_locations': [[]],
        'arenas_box_locations': [[]],
        'arenas_robot_locations': [[]],
        'sample_vases': False,
        'sample_hazards': False,
        'sample_gremlins': False,
        'sample_pillars': False,
        'sample_buttons': False,

        # Goal parameters
        'goal_placements': None,  # Placements where goal may appear (defaults to full extents)
        'goal_locations': [],  # Fixed locations to override placements
        'goal_keepout': 0.4,  # Keepout radius when placing goals
        'goal_size': 0.3,  # Radius of the goal area (if using task 'goal')

        # Box parameters (only used if task == 'push')
        'box_placements': None,  # Box placements list (defaults to full extents)
        'box_locations': [],  # Fixed locations to override placements
        'box_keepout': 0.2,  # Box keepout radius for placement
        'box_size': 0.2,  # Box half-radius size
        'box_density': 0.001,  # Box density
        'box_null_dist': 2, # Within box_null_dist * box_size radius of box, no box reward given

        # Reward is distance towards goal plus a constant for being within range of goal
        # reward_distance should be positive to encourage moving towards the goal
        # if reward_distance is 0, then the reward function is sparse
        'reward_distance': 1.0,  # Dense reward multiplied by the distance moved to the goal
        'reward_goal': 0.0,  # Sparse reward for being inside the goal area
        'reward_box_dist': 0.0,  # Dense reward for moving the robot towards the box
        'reward_box_goal': 1.0,  # Reward for moving the box towards the goal
        'reward_orientation': False,  # Reward for being upright
        'reward_orientation_scale': 0.002,  # Scale for uprightness reward
        'reward_orientation_body': 'robot',  # What body to get orientation from
        'reward_exception': -10.0,  # Reward when encoutering a mujoco exception
        'reward_x': 1.0,  # Reward for forward locomotion tests (vel in x direction)
        'reward_z': 1.0,  # Reward for standup tests (vel in z direction)
        'reward_circle': 1e-1,  # Reward for circle goal (complicated formula depending on pos and vel)
        'reward_clip': 10,  # Clip reward, last resort against physics errors causing magnitude spikes

        # Buttons are small immovable spheres, to the environment
        'buttons_num': 0,  # Number of buttons to add
        'buttons_placements': None,  # Buttons placements list (defaults to full extents)
        'buttons_locations': [],  # Fixed locations to override placements
        'buttons_keepout': 0.3,  # Buttons keepout radius for placement
        'buttons_size': 0.1,  # Size of buttons in the scene
        'buttons_cost': 1.0,  # Cost for pressing the wrong button, if constrain_buttons
        'buttons_resampling_delay': 10,  # Buttons have a timeout period (steps) before resampling

        # Circle parameters (only used if task == 'circle')
        'circle_radius': 1.5,

        # Sensor observations
        # Specify which sensors to add to observation space
        'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
        'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
        'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
        'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

        # Walls - barriers in the environment not associated with any constraint
        # NOTE: this is probably best to be auto-generated than manually specified
        'walls_num': 0,  # Number of walls
        'walls_placements': None,  # This should not be used
        'walls_locations': [],  # This should be used and length == walls_num
        'walls_keepout': 0.0,  # This should not be used
        'walls_size': 0.5,  # Should be fixed at fundamental size of the world

        # Constraints - flags which can be turned on
        # By default, no constraints are enabled, and all costs are indicator functions.
        'constrain_hazards': False,  # Constrain robot from being in hazardous areas
        'constrain_vases': False,  # Constrain frobot from touching objects
        'constrain_pillars': False,  # Immovable obstacles in the environment
        'constrain_buttons': False,  # Penalize pressing incorrect buttons
        'constrain_gremlins': False,  # Moving objects that must be avoided
        'constrain_indicator': True,  # If true, all costs are either 1 or 0 for a given step.

        # Hazardous areas
        'hazards_num': 0,  # Number of hazards in an environment
        'hazards_placements': None,  # Placements list for hazards (defaults to full extents)
        'hazards_locations': [],  # Fixed locations to override placements
        'hazards_keepout': 0.4,  # Radius of hazard keepout for placement
        'hazards_size': 0.3,  # Radius of hazards
        'hazards_cost': 1.0,  # Cost (per step) for violating the constraint

        # Vases (objects we should not touch)
        'vases_num': 0,  # Number of vases in the world
        'vases_placements': None,  # Vases placements list (defaults to full extents)
        'vases_locations': [],  # Fixed locations to override placements
        'vases_keepout': 0.15,  # Radius of vases keepout for placement
        'vases_size': 0.1,  # Half-size (radius) of vase object
        'vases_density': 0.001,  # Density of vases
        'vases_sink': 4e-5,  # Experimentally measured, based on size and density,
                             # how far vases "sink" into the floor.
        # Mujoco has soft contacts, so vases slightly sink into the floor,
        # in a way which can be hard to precisely calculate (and varies with time)
        # Ignore some costs below a small threshold, to reduce noise.
        'vases_contact_cost': 1.0,  # Cost (per step) for being in contact with a vase
        'vases_displace_cost': 0.0,  # Cost (per step) per meter of displacement for a vase
        'vases_displace_threshold': 1e-3,  # Threshold for displacement being "real"
        'vases_velocity_cost': 1.0,  # Cost (per step) per m/s of velocity for a vase
        'vases_velocity_threshold': 1e-4,  # Ignore very small velocities

        # Pillars (immovable obstacles we should not touch)
        'pillars_num': 0,  # Number of pillars in the world
        'pillars_placements': None,  # Pillars placements list (defaults to full extents)
        'pillars_locations': [],  # Fixed locations to override placements
        'pillars_keepout': 0.3,  # Radius for placement of pillars
        'pillars_size': 0.2,  # Half-size (radius) of pillar objects
        'pillars_height': 0.1,  # Half-height of pillars geoms
        'pillars_cost': 1.0,  # Cost (per step) for being in contact with a pillar

        # Gremlins (moving objects we should avoid)
        'gremlins_num': 0,  # Number of gremlins in the world
        'gremlins_placements': None,  # Gremlins placements list (defaults to full extents)
        'gremlins_locations': [],  # Fixed locations to override placements
        'gremlins_keepout': 0.5,  # Radius for keeping out (contains gremlin path)
        'gremlins_travel': 0.3,  # Radius of the circle traveled in
        'gremlins_size': 0.1,  # Half-size (radius) of gremlin objects
        'gremlins_density': 0.001,  # Density of gremlins
        'gremlins_contact_cost': 1.0,  # Cost for touching a gremlin
        'gremlins_dist_threshold': 0.2,  # Threshold for cost for being too close
        'gremlins_dist_cost': 1.0,  # Cost for being within distance threshold

        # Frameskip is the number of physics simulation steps per environment step
        # Frameskip is sampled as a binomial distribution
        # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
        'frameskip_binom_n': 10,  # Number of draws trials in binomial distribution (max frameskip)
        'frameskip_binom_p': 1.0,  # Probability of trial return (controls distribution)

        '_seed': None,  # Random state seed (avoid name conflict with self.seed)
    }

    def __init__(self, config={}):
        # First, parse configuration. Important note: LOTS of stuff happens in
        # parse, and many attributes of the class get set through setattr. If you
        # are trying to track down where an attribute gets initially set, and 
        # can't find it anywhere else, it's probably set via the config dict
        # and this parse function.
        self.parse(config)
        gym.utils.EzPickle.__init__(self, config=config)

        # Load up a simulation of the robot, just to figure out observation space
        self.robot = Robot(self.robot_base)

        self.action_space = gym.spaces.Box(-1, 1, (self.robot.nu,), dtype=np.float32)
        self.build_observation_space()

        self.viewer = None
        self.world = None
        self.clear()

        self.seed(self._seed)
        self.done = True

        self.configurable_types = ["greenobj", "blueobj", "hazard", "pillar", "vase"]

    def parse(self, config):
        ''' Parse a config dict - see self.DEFAULT for description '''
        self.config = deepcopy(self.DEFAULT)
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            assert key in self.DEFAULT, f'Bad key {key}'
            setattr(self, key, value)

    @property
    def sim(self):
        ''' Helper to get the world's simulation instance '''
        return self.world.sim

    @property
    def model(self):
        ''' Helper to get the world's model instance '''
        return self.sim.model

    @property
    def data(self):
        ''' Helper to get the world's simulation data instance '''
        return self.sim.data

    @property
    def robot_pos(self):
        ''' Helper to get current robot position '''
        return self.data.get_body_xpos('robot').copy()

    @property
    def goal_pos(self):
        ''' Helper to get goal position from layout '''
        if self.task in ['goal', 'push']:
            return self.data.get_body_xpos('goal').copy()
        elif self.task == 'button':
            return self.data.get_body_xpos(f'button{self.goal_button}').copy()
        elif self.task == 'circle':
            return ORIGIN_COORDINATES
        elif self.task == 'none':
            return np.zeros(2)  # Only used for screenshots
        elif self.task == 'cleanup' or self.task == 'cleanup_all':
            pass
        else:
            raise ValueError(f'Invalid task {self.task}')

    @property
    def box_pos(self):
        ''' Helper to get the box position '''
        return self.data.get_body_xpos('box').copy()

    @property
    def buttons_pos(self):
        ''' Helper to get the list of button positions '''
        return [self.data.get_body_xpos(f'button{i}').copy() for i in range(self.num_objects_in_layout("button", self.layout))]

    @property
    def vases_pos(self):
        ''' Helper to get the list of vase positions '''
        return [self.data.get_body_xpos(f'vase{p}').copy() for p in range(self.num_objects_in_layout("vase", self.layout))]

    @property
    def gremlins_obj_pos(self):
        ''' Helper to get the current gremlin position '''
        return [self.data.get_body_xpos(f'gremlin{i}obj').copy() for i in range(self.num_objects_in_layout("gremlin", self.layout))]

    @property
    def pillars_pos(self):
        ''' Helper to get list of pillar positions '''
        return [self.data.get_body_xpos(f'pillar{i}').copy() for i in range(self.num_objects_in_layout("pillar", self.layout))]

    @property
    def hazards_pos(self):
        ''' Helper to get the hazards positions from layout '''
        return [self.data.get_body_xpos(f'hazard{i}').copy() for i in range(self.num_objects_in_layout("hazard", self.layout))]

    @property
    def walls_pos(self):
        ''' Helper to get the hazards positions from layout '''
        return [self.data.get_body_xpos(f'wall{i}').copy() for i in range(self.num_objects_in_layout("wall", self.layout))]
    
    @property
    def greenobjs_pos(self):
        ''' Helper to get the greenobjs positions from layout '''
        return [self.data.get_body_xpos(f'greenobj{i}').copy() for i in range(self.num_objects_in_layout("greenobj", self.layout))]
    
    @property
    def leftgoal_pos(self):
        ''' Helper to get the leftgoal positions from layout '''
        return self.data.get_body_xpos('leftgoal').copy()
    
    @property
    def rightgoal_pos(self):
        ''' Helper to get the rightgoal positions from layout '''
        return self.data.get_body_xpos('rightgoal').copy()
    
    @property
    def blueobjs_pos(self):
        ''' Helper to get the blueobjs positions from layout '''
        return [self.data.get_body_xpos(f'blueobj{i}').copy() for i in range(self.num_objects_in_layout("blueobj", self.layout))]

    def build_observation_space(self):
        ''' Construct observtion space.  Happens only once at during __init__ '''
        obs_space_dict = OrderedDict()  # See self.obs()

        if self.observe_freejoint:
            obs_space_dict['freejoint'] = gym.spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32)
        if self.observe_com:
            obs_space_dict['com'] = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
        if self.observe_sensors:
            for sensor in self.sensors_obs:  # Explicitly listed sensors
                dim = self.robot.sensor_dim[sensor]
                obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float32)
            # Velocities don't have wraparound effects that rotational positions do
            # Wraparounds are not kind to neural networks
            # Whereas the angle 2*pi is very close to 0, this isn't true in the network
            # In theory the network could learn this, but in practice we simplify it
            # when the sensors_angle_components switch is enabled.
            for sensor in self.robot.hinge_vel_names:
                obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
            for sensor in self.robot.ballangvel_names:
                obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
            # Angular positions have wraparound effects, so output something more friendly
            if self.sensors_angle_components:
                # Single joints are turned into sin(x), cos(x) pairs
                # These should be easier to learn for neural networks,
                # Since for angles, small perturbations in angle give small differences in sin/cos
                for sensor in self.robot.hinge_pos_names:
                    obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
                # Quaternions are turned into 3x3 rotation matrices
                # Quaternions have a wraparound issue in how they are normalized,
                # where the convention is to change the sign so the first element to be positive.
                # If the first element is close to 0, this can mean small differences in rotation
                # lead to large differences in value as the latter elements change sign.
                # This also means that the first element of the quaternion is not expectation zero.
                # The SO(3) rotation representation would be a good replacement here,
                # since it smoothly varies between values in all directions (the property we want),
                # but right now we have very little code to support SO(3) roatations.
                # Instead we use a 3x3 rotation matrix, which if normalized, smoothly varies as well.
                for sensor in self.robot.ballquat_names:
                    obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (3, 3), dtype=np.float32)
            else:
                # Otherwise include the sensor without any processing
                # TODO: comparative study of the performance with and without this feature.
                for sensor in self.robot.hinge_pos_names:
                    obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
                for sensor in self.robot.ballquat_names:
                    obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)
        if self.task == 'push':
            if self.observe_box_comp:
                obs_space_dict['box_compass'] = gym.spaces.Box(-1.0, 1.0, (self.compass_shape,), dtype=np.float32)
            if self.observe_box_lidar:
                obs_space_dict['box_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.observe_goal_dist:
            obs_space_dict['goal_dist'] = gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32)
        if self.observe_goal_comp:
            obs_space_dict['goal_compass'] = gym.spaces.Box(-1.0, 1.0, (self.compass_shape,), dtype=np.float32)
        if self.observe_goal_lidar:
            obs_space_dict['goal_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.task == 'circle' and self.observe_circle:
            obs_space_dict['circle_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.observe_remaining:
            obs_space_dict['remaining'] = gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32)
        if self.walls_num and self.observe_walls:
            obs_space_dict['walls_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.observe_hazards:
            obs_space_dict['hazards_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.observe_vases:
            obs_space_dict['vases_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.gremlins_num and self.observe_gremlins:
            obs_space_dict['gremlins_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.pillars_num and self.observe_pillars:
            obs_space_dict['pillars_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.buttons_num and self.observe_buttons:
            obs_space_dict['buttons_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.observe_qpos:
            obs_space_dict['qpos'] = gym.spaces.Box(-np.inf, np.inf, (self.robot.nq,), dtype=np.float32)
        if self.observe_qvel:
            obs_space_dict['qvel'] = gym.spaces.Box(-np.inf, np.inf, (self.robot.nv,), dtype=np.float32)
        if self.observe_ctrl:
            obs_space_dict['ctrl'] = gym.spaces.Box(-np.inf, np.inf, (self.robot.nu,), dtype=np.float32)
        if self.observe_vision:
            width, height = self.vision_size
            rows, cols = height, width
            self.vision_size = (rows, cols)
            obs_space_dict['vision'] = gym.spaces.Box(0, 1.0, self.vision_size + (3,), dtype=np.float32)
        # Flatten it ourselves
        self.obs_space_dict = obs_space_dict
        if self.observation_flatten:
            self.obs_flat_size = sum([np.prod(i.shape) for i in self.obs_space_dict.values()])
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_flat_size,), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Dict(obs_space_dict)

    def toggle_observation_space(self):
        self.observation_flatten = not(self.observation_flatten)
        self.build_observation_space()

    def placements_from_location(self, location, keepout):
        ''' Helper to get a placements list from a given location and keepout '''
        x, y = location
        return [(x - keepout, y - keepout, x + keepout, y + keepout)]

    def placements_dict_from_object(self, object_name, obj_dict=None):
        ''' Get the placements dict subset just for a given object name.

        Args:
            object_name
            sample_num: if True the number of objects to be added is sampled uniformly
            between 0 and the number specified.
        '''
        placements_dict = {}
        if hasattr(self, object_name + 's_num'):  # Objects with multiplicity
            plural_name = object_name + 's'
            object_fmt = object_name + '{i}'
            object_keepout = getattr(self, plural_name + '_keepout')

            if object_name == 'wall' and self.sample_arenas:
                object_num = 8
                wall_size = self.arena_sizes[self.arena]
                dist = 2 * wall_size
                object_locations = [np.array((-dist, -dist)), 
                                    np.array((dist, -dist)), 
                                    np.array((dist, dist)), 
                                    np.array((-dist, dist)), 
                                    np.array((-dist, 0)), 
                                    np.array((0, -dist)),
                                    np.array((0, dist)),
                                    np.array((dist, 0)),
                ]
                object_placements = None
            else:
                object_num = getattr(self, plural_name + '_num', None)
                object_locations = getattr(self, plural_name + '_locations', [])
                object_placements = getattr(self, plural_name + '_placements', None)
            
                # use number in object dict if provided
                if obj_dict is not None:
                    if object_name in obj_dict.keys():
                        object_num = obj_dict[object_name]

                

        else:  # Unique objects
            object_fmt = object_name
            object_num = 1
            if self.sample_arenas:
                object_locations = getattr(self, "arenas_" + object_name + '_locations', [[] for _ in range(len(self.arenas))])
                object_placements = getattr(self, "arenas_" + object_name + '_placements', [None for _ in range(len(self.arenas))])
                object_locations = object_locations[self.arena]
                object_placements = object_placements[self.arena]
                object_keepout = getattr(self, object_name + '_keepout')
            else:
                object_locations = getattr(self, object_name + '_locations', [])
                object_placements = getattr(self, object_name + '_placements', None)
                object_keepout = getattr(self, object_name + '_keepout')

        for i in range(object_num):
            if i < len(object_locations):
                x, y = object_locations[i]
                k = object_keepout + 1e-9  # Epsilon to account for numerical issues
                placements = [(x - k, y - k, x + k, y + k)]
            else:
                placements = object_placements
            placements_dict[object_fmt.format(i=i)] = (placements, object_keepout)
        return placements_dict

    def build_placements_dict(self, config_dict=None):
        ''' Build a dict of placements.  Happens once during __init__. '''
        # Dictionary is map from object name -> tuple of (placements list, keepout)
        
        
        if config_dict is None:
            config_dict = self.sample_config_dict()

        if self.sample_arenas:
            self.arena = config_dict["arena"]
            self.placement_extent = self.arenas_placements_extents[self.arena]
        placements = {}
        placements.update(self.placements_dict_from_object('robot'))
        placements.update(self.placements_dict_from_object('wall'))

        if self.task in ['goal', 'push']:
            placements.update(self.placements_dict_from_object('goal'))
        if self.task == 'push':
            placements.update(self.placements_dict_from_object('box'))
        if self.task == 'button' or self.buttons_num or self.sample_buttons:
            placements.update(self.placements_dict_from_object('button', config_dict))
        if self.hazards_num or self.sample_hazards:
            placements.update(self.placements_dict_from_object('hazard', config_dict))
        if self.vases_num or self.sample_vases:
            placements.update(self.placements_dict_from_object('vase', config_dict))
        if self.pillars_num or self.sample_pillars: 
            placements.update(self.placements_dict_from_object('pillar', config_dict))
        if self.gremlins_num or self.sample_gremlins: 
            placements.update(self.placements_dict_from_object('gremlin', config_dict))
        if self.blueobjs_num or self.sample_blueobjs: 
            placements.update(self.placements_dict_from_object('blueobj', config_dict))
        if self.greenobjs_num or self.sample_greenobjs: 
            placements.update(self.placements_dict_from_object('greenobj', config_dict))

        self.placements = placements

    def get_objects_to_sample(self):
        to_sample = []
        if self.sample_vases:
            to_sample.append("vase")
        if self.sample_hazards:
            to_sample.append("hazard")
        if self.sample_pillars:
            to_sample.append("pillar")
        if self.sample_gremlins:
            to_sample.append("gremlin")
        if self.sample_buttons:
            to_sample.append("button")
        if self.sample_blueobjs:
            to_sample.append("blueobj")
        if self.sample_greenobjs:
            to_sample.append("greenobj")
        return to_sample

    def sample_config_dict(self):
        if self.sample_arenas:
            arena = np.random.choice(self.arenas_to_sample)
            max_object_num = self.arenas_max_object_nums[arena]
        else:
            arena = 0
            max_object_num = self.max_object_num

        config_dict = {obj: 0 for obj in self.configurable_types}
        config_dict["arena"] = arena

        if self.sample_num_objects and np.random.uniform() < self.sample_objects_prob:
            object_num = self.rs.choice(max_object_num + 1)
            if object_num > 0:
                obj_types_to_sample = self.get_objects_to_sample()
                all_combos = list(itertools.combinations_with_replacement(obj_types_to_sample, object_num))
                sampled_obj = all_combos[np.random.choice(len(all_combos))]
                [config_dict.update({obj: sampled_obj.count(obj)}) for obj in self.configurable_types]
        return config_dict

    def seed(self, seed=None):
        ''' Set internal random state seeds '''
        self._seed = np.random.randint(2**32) if seed is None else seed

    def build_layout(self):
        ''' Rejection sample a placement of objects to find a layout. '''
        if not self.randomize_layout:
            self.rs = np.random.RandomState(0)

        for _ in range(50000):
            if self.sample_layout():
                break