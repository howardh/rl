import pytest
import unittest
import numpy as np

import torch

from rl.learner import learner 
from rl.learner.rbf_learner import RBFFunction
from rl.learner.rbf_learner import RBFLearner

class TestRBFFunction(unittest.TestCase):

    def setUp(self):
        self.rbf = RBFFunction()

    @pytest.mark.skip(reason="Unimportant for now. Fix later.")
    def test_updates_weights(self):
        n = 3 # number of RBFs
        d = 1 # number of dimensions
        x = torch.ones([1,d],requires_grad=False).float()
        w = torch.zeros([n,d], requires_grad=True).float()
        s = torch.tensor([1], requires_grad=False).float()
        c = torch.tensor(np.array([list(range(n))]*d).transpose(),requires_grad=False).float()

        y_pred1 = self.rbf(x,c,w,s)
        y = torch.tensor(np.array([[1]]), requires_grad=False).float()
        loss = (y_pred1-y).pow(2)
        loss.backward()

        self.assertTrue(w.grad is not None, "Gradient was not computed.")
        self.assertNotEqual(w.grad.data.sum(), 0, "Gradient is 0")
        w.data -= 0.01*w.grad.data
        w.grad.data.zero_()

        y_pred2 = self.rbf(x,c,w,s)
        y = torch.tensor(np.array([[1]]), requires_grad=False).float()
        loss2 = (y_pred2-y).pow(2)

        self.assertTrue(loss.data[0][0] > loss2.data[0][0], "Error is increasing")

    @pytest.mark.skip(reason="Unimportant for now. Fix later.")
    def test_1d_forward(self):
        n = 3 # number of RBFs
        d = 1 # number of dimensions

        # weights = 0
        x = torch.tensor(np.array([[1]]), requires_grad=False).float()
        w = torch.zeros([n,1], requires_grad=False).float()
        s = torch.tensor([1], requires_grad=False).float()
        c = torch.tensor(np.array([list(range(n))]*d).transpose(),requires_grad=False).float()
        y_pred = self.rbf(x,c,w,s)
        self.assertEqual(y_pred.data[0][0], 0, "Incorrect output")

        # exp(-(1-0)^2) + exp(-(1-1)^2) + exp(-(1-2)^2)
        w = torch.ones([n,1], requires_grad=True).float()
        y_pred = self.rbf(x,c,w,s)
        self.assertAlmostEqual(y_pred.item(), 1.73575888234, places=5,
                msg="Incorrect output.\nx: %sw: %ss: %sc: %s" % (x,w,s,c))

        # -5*exp(-(1-0)^2) + 2*exp(-(1-1)^2) + 9*exp(-(1-2)^2)
        w = torch.tensor(np.array([[-5,2,9]]).transpose(), requires_grad=True, dtype=torch.float)
        y_pred = self.rbf(x,c,w,s)
        self.assertAlmostEqual(y_pred.item(), 3.47151776469, places=5,
                msg="Incorrect output.\nx: %sw: %ss: %sc: %s" % (x,w,s,c))

        # -5*exp(-0.1*(1-0)^2) + 2*exp(-0.1*(1-1)^2) + 9*exp(-0.1*(1-2)^2)
        s = torch.tensor([0.1], requires_grad=False).float()
        y_pred = self.rbf(x,c,w,s)
        self.assertAlmostEqual(y_pred.item(), 5.61934967214, places=5,
                msg="Incorrect output.\nx: %sw: %ss: %sc: %s" % (x,w,s,c))

    @pytest.mark.skip(reason="Unimportant for now. Fix later.")
    def test_2d_forward(self):
        n = 3 # number of RBFs
        d = 2 # number of dimensions

        # weights = 0
        x = torch.tensor(np.array([[2,3]]), requires_grad=False).float()
        w = torch.zeros([n,1], requires_grad=True).float()
        s = torch.tensor([1], requires_grad=False).float()
        c = torch.tensor(np.array([list(range(n))]*d).transpose(),requires_grad=False).float()
        y_pred = self.rbf(x,c,w,s)
        self.assertEqual(y_pred.data[0][0], 0, "Incorrect output")

        # exp(-((2-0)^2+(3-0)^2)) + exp(-((2-1)^2+(3-1)^2)) + exp(-((2-1)^2+(3-1)^2))
        w = torch.ones([n,1], requires_grad=True).float()
        y_pred = self.rbf(x,c,w,s)
        self.assertAlmostEqual(y_pred.item(), 0.3746196485, places=5,
                msg="Incorrect output.\nx: %sw: %ss: %sc: %s" % (x,w,s,c))

        # -5*exp(-((2-0)^2+(3-0)^2)) + 2*exp(-((2-1)^2+(3-1)^2)) + 9*exp(-((2-2)^2+(3-2)^2))
        w = torch.tensor(np.array([[-5,2,9]]).transpose(), dtype=torch.float, requires_grad=True)
        y_pred = self.rbf(x,c,w,s)
        self.assertAlmostEqual(y_pred.item(), 3.32437956289, places=5,
                msg="Incorrect output.\nx: %sw: %ss: %sc: %s" % (x,w,s,c))

        # -5*exp(-((2-0)^2+(3-0)^2)*.1) + 2*exp(-((2-1)^2+(3-1)^2)*.1) + 9*exp(-((2-2)^2+(3-2)^2)*.1)
        s = torch.tensor([0.1], requires_grad=False).float()
        y_pred = self.rbf(x,c,w,s)
        self.assertAlmostEqual(y_pred.item(), 7.99393911658, places=5,
                msg="Incorrect output.\nx: %sw: %ss: %sc: %s" % (x,w,s,c))

    #def test_backward(self):
    #    x = Variable(torch.from_numpy(np.array([[0,0]])).float(), requires_grad=False)
    #    w = Variable(torch.zeros(1,1).float(), requires_grad=True)
    #    s = Variable(torch.Tensor([1]).float(), requires_grad=False)
    #    c = Variable(torch.from_numpy(np.array([[0,0]])).float(),
    #            requires_grad=False)

    #    y_pred = self.rbf(x,c,w,s)
    #    self.assertEqual(y_pred.data[0][0], 0, "Incorrect output")

#class TestRBFLearner(unittest.TestCase):
#
#    def setUp(self):
#        self.learner = RBFLearner(
#                action_space=np.array([0]),
#                observation_space=np.array([0,1]),
#                discount_factor=0.9,
#                learning_rate=0.05,
#                dimensions=[2]
#        )

if __name__ == "__main__":
    unittest.main()
