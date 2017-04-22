import unittest
import numpy as np

from src import learner 

class TestTabularLearner(unittest.TestCase):

    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.9
    
    def setUp(self):
        self.learner = learner.TabularLearner()
        self.learner.action_space = np.array([0])
        self.learner.learning_rate = self.LEARNING_RATE
        self.learner.discount_factor = self.DISCOUNT_FACTOR

    def test_non_existant_state(self):
        val = self.learner.get_state_action_value(0,0)
        self.assertEqual(val, 0)

    def test_observe_step(self):
        self.learner.observe_step(0,0,1,1)
        val1 = self.learner.get_state_action_value(0,0)
        self.assertEqual(val1, self.LEARNING_RATE)

        self.learner.observe_step(1,0,1,0)
        val2 = self.learner.get_state_action_value(1,0)
        self.assertEqual(val2, self.LEARNING_RATE*(1+self.DISCOUNT_FACTOR*val1))

        self.learner.observe_step(0,0,2,1)
        val3 = self.learner.get_state_action_value(0,0)
        self.assertEqual(val3, val1 + self.LEARNING_RATE*(2+self.DISCOUNT_FACTOR*val2-val1))

class TestLSTDLearner(unittest.TestCase):

    NUM_FEATURES = 3

    def setUp(self):
        self.learner = learner.LSTDLearner(
                num_features=self.NUM_FEATURES,
                action_space=np.array([0,1])
        )

    def test_combine_state_action(self):
        state = np.array([[1,2,3]]).transpose()
        action = 0
        expectedOutput = np.array([[1,2,3,0,0,0]]).transpose()
        actualOutput = self.learner.combine_state_action(state, action)
        self.assertTrue((actualOutput == expectedOutput).all())

        state = np.array([[1,2,3]]).transpose()
        action = 1
        expectedOutput = np.array([[0,0,0,1,2,3]]).transpose()
        actualOutput = self.learner.combine_state_action(state, action)
        self.assertTrue((actualOutput == expectedOutput).all())

if __name__ == "__main__":
    unittest.main()
