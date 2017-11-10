import unittest
import numpy as np

import torch
from torch.autograd import Variable

from learner import learner 
from learner.rbf_learner import RBFFunction
from learner.rbf_learner import RBFLearner

class TestRBFFunction(unittest.TestCase):

    def setUp(self):
        self.rbf = RBFFunction()

    def test_updates_weights(self):
        n = 3 # number of RBFs
        d = 1 # number of dimensions
        x = Variable(torch.ones(1,d).float(), requires_grad=False)
        w = Variable(torch.zeros(n,d).float(), requires_grad=True)
        s = Variable(torch.Tensor([1]).float(), requires_grad=False)
        c = Variable(torch.from_numpy(np.array([list(range(n))]*d).transpose()).float(),
                requires_grad=False)

        y_pred1 = self.rbf(x,c,w,s)
        y = Variable(torch.from_numpy(np.array([[1]])).float(), requires_grad=False)
        loss = (y_pred1-y).pow(2)
        loss.backward()

        self.assertTrue(w.grad is not None, "Gradient was not computed.")
        self.assertNotEqual(w.grad.data.sum(), 0, "Gradient is 0")
        w.data -= 0.01*w.grad.data
        w.grad.data.zero_()

        y_pred2 = self.rbf(x,c,w,s)
        y = Variable(torch.from_numpy(np.array([[1]])).float(), requires_grad=False)
        loss2 = (y_pred2-y).pow(2)

        self.assertTrue(loss.data[0][0] > loss2.data[0][0], "Error is increasing")

    def test_1d_forward(self):
        n = 3 # number of RBFs
        d = 1 # number of dimensions

        # weights = 0
        x = Variable(torch.from_numpy(np.array([[1]])).float(), requires_grad=False)
        w = Variable(torch.zeros(n,1).float(), requires_grad=True)
        s = Variable(torch.Tensor([1]).float(), requires_grad=False)
        c = Variable(torch.from_numpy(np.array([list(range(n))]*d).transpose()).float(),
                requires_grad=False)
        y_pred = self.rbf(x,c,w,s)
        self.assertEqual(y_pred.data[0][0], 0, "Incorrect output")

        # exp(-(1-0)^2) + exp(-(1-1)^2) + exp(-(1-2)^2)
        w = Variable(torch.ones(n,1).float(), requires_grad=True)
        y_pred = self.rbf(x,c,w,s)
        self.assertAlmostEqual(y_pred.data[0][0], 1.73575888234, places=5,
                msg="Incorrect output.\nx: %sw: %ss: %sc: %s" % (x,w,s,c))

        # -5*exp(-(1-0)^2) + 2*exp(-(1-1)^2) + 9*exp(-(1-2)^2)
        w = Variable(torch.from_numpy(np.array([[-5,2,9]]).transpose()).float(), requires_grad=True)
        y_pred = self.rbf(x,c,w,s)
        self.assertAlmostEqual(y_pred.data[0][0], 3.47151776469, places=5,
                msg="Incorrect output.\nx: %sw: %ss: %sc: %s" % (x,w,s,c))

        # -5*exp(-0.1*(1-0)^2) + 2*exp(-0.1*(1-1)^2) + 9*exp(-0.1*(1-2)^2)
        s = Variable(torch.Tensor([0.1]).float(), requires_grad=False)
        y_pred = self.rbf(x,c,w,s)
        self.assertAlmostEqual(y_pred.data[0][0], 5.61934967214, places=5,
                msg="Incorrect output.\nx: %sw: %ss: %sc: %s" % (x,w,s,c))

    def test_2d_forward(self):
        n = 3 # number of RBFs
        d = 2 # number of dimensions

        # weights = 0
        x = Variable(torch.from_numpy(np.array([[2,3]])).float(), requires_grad=False)
        w = Variable(torch.zeros(n,1).float(), requires_grad=True)
        s = Variable(torch.Tensor([1]).float(), requires_grad=False)
        c = Variable(torch.from_numpy(np.array([list(range(n))]*d).transpose()).float(),
                requires_grad=False)
        y_pred = self.rbf(x,c,w,s)
        self.assertEqual(y_pred.data[0][0], 0, "Incorrect output")

        # exp(-((2-0)^2+(3-0)^2)) + exp(-((2-1)^2+(3-1)^2)) + exp(-((2-1)^2+(3-1)^2))
        w = Variable(torch.ones(n,1).float(), requires_grad=True)
        y_pred = self.rbf(x,c,w,s)
        self.assertAlmostEqual(y_pred.data[0][0], 0.3746196485, places=5,
                msg="Incorrect output.\nx: %sw: %ss: %sc: %s" % (x,w,s,c))

        # -5*exp(-((2-0)^2+(3-0)^2)) + 2*exp(-((2-1)^2+(3-1)^2)) + 9*exp(-((2-2)^2+(3-2)^2))
        w = Variable(torch.from_numpy(np.array([[-5,2,9]]).transpose()).float(), requires_grad=True)
        y_pred = self.rbf(x,c,w,s)
        self.assertAlmostEqual(y_pred.data[0][0], 3.32437956289, places=5,
                msg="Incorrect output.\nx: %sw: %ss: %sc: %s" % (x,w,s,c))

        # -5*exp(-((2-0)^2+(3-0)^2)*.1) + 2*exp(-((2-1)^2+(3-1)^2)*.1) + 9*exp(-((2-2)^2+(3-2)^2)*.1)
        s = Variable(torch.Tensor([0.1]).float(), requires_grad=False)
        y_pred = self.rbf(x,c,w,s)
        self.assertAlmostEqual(y_pred.data[0][0], 7.99393911658, places=5,
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
#                np.array([0]),
#        )

if __name__ == "__main__":
    unittest.main()
