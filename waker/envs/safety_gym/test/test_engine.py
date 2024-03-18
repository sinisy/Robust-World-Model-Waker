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
            p.step(np.zeros(p.acti