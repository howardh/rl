import unittest
import numpy as np

from learner import learner 

class TestTabularLearner(unittest.TestCase):

    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.9
    
    def setUp(self):
        self.learner = learner.TabularLearner(
                action_space=np.array([0]),
                discount_factor=self.DISCOUNT_FACTOR,
                learning_rate=self.LEARNING_RATE,
        )

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
    NUM_ACTIONS = 2
    DISCOUNT_FACTOR = 0.9

    def setUp(self):
        self.learner = learner.LSTDLearner(
                num_features=self.NUM_FEATURES,
                discount_factor=self.DISCOUNT_FACTOR,
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

    def test_observe_step(self):
        a_mat = self.learner.a_mat
        expected_a_mat = np.zeros(self.NUM_ACTIONS*self.NUM_FEATURES)
        self.assertTrue(np.allclose(a_mat, expected_a_mat), "Incorrect A matrix. Expected %s, Received %s" % (expected_a_mat, a_mat))

        s1 = np.array([[1,0,0]]).transpose()
        a1 = 0
        r = 0
        s2 = np.array([[0,1,0]]).transpose()
        term = True
        self.learner.observe_step(s1,a1,r,s2,term)
        a_mat = self.learner.a_mat
        expected_a_mat = np.matrix([
            [1,0,0, 0,0,0],
            [0,0,0, 0,0,0],
            [0,0,0, 0,0,0],
            [0,0,0, 0,0,0],
            [0,0,0, 0,0,0],
            [0,0,0, 0,0,0]])
        self.assertTrue(np.allclose(a_mat, expected_a_mat), "Incorrect A matrix. Expected %s, Received %s" % (expected_a_mat, a_mat))

if __name__ == "__main__":
    unittest.main()
