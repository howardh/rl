import unittest
import numpy as np

from cartpole import features

class TestCartPoleFeatures(unittest.TestCase):
    
    def setUp(self):
        pass

    def testRealFeatures(self):
        import gym
        env = gym.make('CartPole-v0')
        obs = env.reset()
        # This should not give an error
        # Ideally, should return a column vector
        features.one_hot(obs)
