import unittest
import pytest
import numpy as np
import scipy.sparse
import gym

from learner import learner 

class TestTabularLearner(unittest.TestCase):

    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.9
    
    def setUp(self):
        self.learner = learner.TabularLearner(
                action_space=gym.spaces.Discrete(1),
                discount_factor=self.DISCOUNT_FACTOR,
                learning_rate=self.LEARNING_RATE,
        )

    def test_non_existant_state(self):
        val = self.learner.get_state_action_value(0,0)
        self.assertEqual(val, 0)

    def test_observe_step(self):
        self.learner.observe_step(0,0,1,1)
        val1 = self.learner.get_state_action_value(0,0)
        assert val1 == pytest.approx(self.LEARNING_RATE)

        self.learner.observe_step(1,0,1,0)
        val2 = self.learner.get_state_action_value(1,0)
        assert val2 == pytest.approx(self.LEARNING_RATE*(1+self.DISCOUNT_FACTOR*val1))

        self.learner.observe_step(0,0,2,1)
        val3 = self.learner.get_state_action_value(0,0)
        assert val3 == pytest.approx(val1 + self.LEARNING_RATE*(2+self.DISCOUNT_FACTOR*val2-val1))

if __name__ == "__main__":
    unittest.main()
