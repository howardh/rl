import unittest
import numpy as np
import scipy.sparse

from learner import lstd_learner as learner

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
        actualOutput = self.learner.combine_state_action(state, action).numpy()
        self.assertTrue((actualOutput == expectedOutput).all())

        state = np.array([[1,2,3]]).transpose()
        action = 1
        expectedOutput = np.array([[0,0,0,1,2,3]]).transpose()
        actualOutput = self.learner.combine_state_action(state, action).numpy()
        self.assertTrue((actualOutput == expectedOutput).all())

    def test_observe_step(self):
        a_mat = self.learner.a_mat.numpy()
        expected_a_mat = np.zeros(self.NUM_ACTIONS*self.NUM_FEATURES)
        self.assertTrue(np.allclose(a_mat, expected_a_mat), "Incorrect A matrix. Expected %s, Received %s" % (expected_a_mat, a_mat))

        s1 = np.array([[1,0,0]]).transpose()
        a1 = 0
        r = 0
        s2 = np.array([[0,1,0]]).transpose()
        term = True
        self.learner.observe_step(s1,a1,r,s2,term)
        a_mat = self.learner.a_mat.numpy()
        expected_a_mat = np.matrix([
            [1,0,0, 0,0,0],
            [0,0,0, 0,0,0],
            [0,0,0, 0,0,0],
            [0,0,0, 0,0,0],
            [0,0,0, 0,0,0],
            [0,0,0, 0,0,0]])
        self.assertTrue(np.allclose(a_mat, expected_a_mat), "Incorrect A matrix. Expected %s, Received %s" % (expected_a_mat, a_mat))


class TestLSTDLearnerBase(unittest.TestCase):

    NUM_FEATURES = 3
    NUM_ACTIONS = 1
    DISCOUNT_FACTOR = 0.876

    def setUp(self):
        self.learner = learner.LSTDLearner(
                num_features=self.NUM_FEATURES,
                discount_factor=self.DISCOUNT_FACTOR,
                action_space=np.array([0])
        )

    def test_learning(self):
        s1 = np.array([[1,0,0]]).transpose()
        s2 = np.array([[0,1,0]]).transpose()
        s3 = np.array([[0,0,1]]).transpose()
        self.learner.observe_step(s1,0,0,s2,False)
        self.learner.observe_step(s2,0,1,s3,True)
        self.learner.update_weights()

        expected = self.DISCOUNT_FACTOR
        actual = self.learner.get_state_action_value(s1, 0)
        self.assertAlmostEqual(expected, actual, 6)

        expected = 1
        actual = self.learner.get_state_action_value(s2, 0)
        self.assertAlmostEqual(expected, actual, 6)

class TestLSTDTraceLearner(TestLSTDLearnerBase):
    def setUp(self):
        self.learner = learner.LSTDTraceLearner(
                num_features=self.NUM_FEATURES,
                discount_factor=self.DISCOUNT_FACTOR,
                action_space=np.array([0]),
                trace_factor=0
        )

class TestLSTDTraceLearner2(TestLSTDLearnerBase):
    def setUp(self):
        self.learner = learner.LSTDTraceLearner(
                num_features=self.NUM_FEATURES,
                discount_factor=self.DISCOUNT_FACTOR,
                action_space=np.array([0]),
                trace_factor=0.123
        )

class TestLSTDTraceQsLearner2(TestLSTDLearnerBase):
    def setUp(self):
        self.learner = learner.LSTDTraceQsLearner(
                num_features=self.NUM_FEATURES,
                discount_factor=self.DISCOUNT_FACTOR,
                action_space=np.array([0]),
                trace_factor = 0,
                sigma = 0,
        )

class TestLSTDTraceQsLearner3(TestLSTDLearnerBase):
    def setUp(self):
        self.learner = learner.LSTDTraceQsLearner(
                num_features=self.NUM_FEATURES,
                discount_factor=self.DISCOUNT_FACTOR,
                action_space=np.array([0]),
                trace_factor = 0.538,
                sigma = 0,
        )

class TestLSPILearner(TestLSTDLearnerBase):
    def setUp(self):
        self.learner = learner.LSPILearner(
                num_features=self.NUM_FEATURES,
                discount_factor=self.DISCOUNT_FACTOR,
                action_space=np.array([0])
        )

class TestLSTDLearnerBaseTwoActions(unittest.TestCase):

    NUM_FEATURES = 3
    NUM_ACTIONS = 1
    DISCOUNT_FACTOR = 0.876

    def setUp(self):
        self.learner = learner.LSTDLearner(
                num_features=self.NUM_FEATURES,
                discount_factor=self.DISCOUNT_FACTOR,
                action_space=np.array([0,1])
        )

    def test_learning(self):
        s1 = np.array([[1,0,0]]).transpose()
        s2 = np.array([[0,1,0]]).transpose()
        s3 = np.array([[0,0,1]]).transpose()
        self.learner.observe_step(s1,0,0,s2,False)
        self.learner.observe_step(s2,0,1,s3,True)
        self.learner.update_weights()
        print(self.learner.a_mat)
        print(self.learner.b_mat)
        print(self.learner.weights)

        expected = self.DISCOUNT_FACTOR
        actual = self.learner.get_state_action_value(s1, 0)
        self.assertAlmostEqual(expected, actual, 6)

        expected = 0
        actual = self.learner.get_state_action_value(s1, 1)
        self.assertAlmostEqual(expected, actual, 6)

        expected = 1
        actual = self.learner.get_state_action_value(s2, 0)
        self.assertAlmostEqual(expected, actual, 6)

        expected = 0
        actual = self.learner.get_state_action_value(s2, 1)
        self.assertAlmostEqual(expected, actual, 6)


class TestLSTDTraceLearnerBaseTwoActions(TestLSTDLearnerBaseTwoActions):
    def setUp(self):
        self.learner = learner.LSTDTraceLearner(
                num_features=self.NUM_FEATURES,
                discount_factor=self.DISCOUNT_FACTOR,
                action_space=np.array([0,1]),
                trace_factor=0
        )


class TestLSTDTraceQsLearnerBaseTwoActions(TestLSTDLearnerBaseTwoActions):
    def setUp(self):
        self.learner = learner.LSTDTraceQsLearner(
                num_features=self.NUM_FEATURES,
                discount_factor=self.DISCOUNT_FACTOR,
                action_space=np.array([0,1]),
                trace_factor=0,
                sigma=1
        )


class TestLSTDTraceQsLearner(unittest.TestCase):

    NUM_FEATURES = 3
    NUM_ACTIONS = 2
    DISCOUNT_FACTOR = 0.9

    def setUp(self):
        self.learner = learner.LSTDTraceQsLearner(
                num_features=self.NUM_FEATURES,
                discount_factor=self.DISCOUNT_FACTOR,
                action_space=np.array([0,1]),
                trace_factor = 0,
                sigma = 0,
        )

    def test_get_all_state_action_pairs(self):
        state = np.array([[1,2,3]]).transpose()
        expectedOutput = np.array([[1,2,3,0,0,0],[0,0,0,1,2,3]]).transpose()
        actualOutput = self.learner.get_all_state_action_pairs(state).numpy()
        self.assertTrue((actualOutput == expectedOutput).all())

if __name__ == "__main__":
    unittest.main()
