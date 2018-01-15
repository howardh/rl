import unittest
import numpy as np
import scipy.sparse
from tqdm import tqdm

from learner.linear_learner import LinearLearner

class TestTabularLearner(unittest.TestCase):

    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.9
    
    def setUp(self):
        self.learner = LinearLearner(
                num_features = 3,
                action_space=np.array([0,1]),
                discount_factor=self.DISCOUNT_FACTOR,
                learning_rate=self.LEARNING_RATE,
        )

    def test_get_state_action_value(self):
        val1 = self.learner.get_state_action_value(np.array([1,1,-1]),0)

    def test_observe_step(self):
        self.learner.weights.data *= 0

        self.learner.observe_step(
                np.array([1,0,0]),
                0,
                1,
                np.array([1,0,0]),
                False
        )
        """
        Target = 1+gamma*0 = 1
        prediction = wx = 0
        loss = (1-wx)^2
        dloss/dw = -2(1-wx)x = [-2 0 0]
        times learning rate of 0.1, and negative: [0.2 0 0]
        """
        expected = np.array([[0.2,0,0],[0,0,0]])
        output = self.learner.weights.data.numpy()
        diff = np.sum(expected-output)
        self.assertAlmostEqual(diff, 0, msg="Gradient is wrong")

if __name__ == "__main__":
    unittest.main()
