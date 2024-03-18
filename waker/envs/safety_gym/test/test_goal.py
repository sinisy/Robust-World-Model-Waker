#!/usr/bin/env python

import unittest
import numpy as np

from safety_gym.envs.engine import Engine, ResamplingError


class TestGoal(unittest.TestCase):
    def rollout_env(self, env):
        ''' roll an environment until it is done '''
        done = False
        while not done:
            _, _, done, _ = env.step([1,0])

    def test_resample(self):
