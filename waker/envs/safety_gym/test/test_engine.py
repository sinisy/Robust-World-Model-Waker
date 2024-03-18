#!/usr/bin/env python

import unittest
import numpy as np
import gym.spaces

from safety_gym.envs.engine import Engine


class TestEngine(unittest.TestCase):
    def test_timeout(self):
        ''' Test that episode is over after num_steps '''
        p = Engine({'num_steps': 10})
        p.reset()
        for _ in range(10):
            self.assertFalse(p.done)
            p.step(np.zeros(p.action_space.shape))
        self.assertTrue(p.done)
        with self.assertRaises(AssertionError):
            p.step(np.zeros(p.action_space.shape))

    def test_flatten(self):
        ''' Test that physics can flatten observations '''
        p = Engine({'observation_flatten': True})
        obs = p.reset()
        self.assertIsInstance(p.observation_space, gym.spaces.Box)
        self.assertEqual(len(p.observation_space.shape), 1)
        self.assertTrue(p.observation_space.contains(obs))

        p = Engine({'observation_flatten': False})
        obs = p.reset()
        self.assertIsInstance(p.observation_space, gym.spaces.Dict)
        self.assertTrue(p.observation_space.contains(obs))

    def test_angle_components(self):
        ''' Test that the angle components are about correct '''
        p = Engine({'robot_base': 'xmls/doggo.xml',
                     'observation_flatten': False,
                     'sensors_angle_components': True,
                     'robot_rot': .3})
        p.reset()
        p.step(p.action_space.high)
        p.step(p.action_space.high)
        p.step(p.action_space.low)
        theta = p.data.get_joint_qpos('hip_1_z