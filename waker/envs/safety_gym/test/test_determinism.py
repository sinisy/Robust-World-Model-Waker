#!/usr/bin/env python

import unittest
import numpy as np
import gym
import envs.safety_gym  # noqa


class TestDeterminism(unittest.TestCase):
    def check_qpos(self, env_name):
        ''' Check that a single environment is seed-stable at init '''
        for seed in [0, 1, 123456789]:
            print('running', env_name, seed)
            env1 = gym.make(env_name)
            env1.seed(np.random.randint(123456789))
            env1.reset()
            env1.seed(seed)
            env1.reset()
            env2 = gym.make(env_name)
            env2.seed(seed)
            env2.