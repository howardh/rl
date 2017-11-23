import itertools
import numpy as np

import torch
from torch.autograd import Variable

from learner.learner import Learner

class RBFFunction(torch.autograd.Function):
    def forward(self, x, c, w, s=None):
        # f(x,c,w) = w*exp(-s*(c-x)^2)
        diff = c-x
        dist = (diff*diff).sum(dim=1, keepdim=True)
        v = torch.exp(-1*s*dist)
        self.dist = dist
        self.unweighted_rbf = v
        y = torch.mm(w.t(),v)
        self.save_for_backward(x,c,w,s,y)
        return y

    def backward(self, grad_output):
        # df/dx = w*exp(-s*(c-x)^2)*(-s)*2*(c-x)*(-1)
        # df/dc = w*exp(-s*(c-x)^2)*(-s)*2*(c-x)
        # df/dw = exp(-s*(c-x)^2)
        # f(r) = w*exp(-s*r^2)
        x,c,w,s,y = self.saved_variables
        v = self.unweighted_rbf
        x = x.data
        c = c.data
        w = w.data
        dist = self.dist
        #grad_c = -w*torch.exp(-s*dist)*s*2*(c-x) # TODO: Doesn't work
        grad_c = None
        grad_w = grad_output*v
        grad_s = grad_output*(w*v*dist).sum()
        return None, grad_c, grad_w, grad_s

class RBFLearner(Learner):

    def __init__(self, action_space, observation_space, discount_factor, learning_rate,
            initial_value=0, optimizer=None):
        Learner.__init__(self)
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.initial_value = initial_value
        self.observation_space = observation_space
        
        self.rbf = RBFFunction()
        centres = list(
                itertools.product(
                    np.linspace(0,1,8),
                    np.linspace(0,1,8)
                )
        )
        weights = [[self.initial_value]*len(centres)]*len(self.action_space)
        self.spread = Variable(torch.Tensor([100]).float(),
                requires_grad=False)
        self.centres = Variable(
                torch.from_numpy(np.array(centres)).float(),
                requires_grad=False)
        self.weights = [Variable(
                torch.from_numpy(np.array([w]).transpose()).float(),
                requires_grad=True) for w in weights]
        self.ones_weights = Variable(
                torch.ones(len(centres),1),
                requires_grad=False)
        self.old_weights = [None]*len(self.weights)
        self.reset_weight_change()

        self.optimizer = torch.optim.SGD(self.weights, lr=self.learning_rate)
        #self.optimizer = torch.optim.Adam(self.weights, lr=self.learning_rate)

    def observe_step(self, state1, action1, reward2, state2, terminal=False):
        alpha = self.learning_rate
        gamma = self.discount_factor

        action_val_target= Variable(
                torch.from_numpy(np.array([reward2+gamma*self.get_state_value(state2)])).float(),
                requires_grad=False)
        action_val_pred = self.get_state_action_value_graph(state1, action1)
        # TODO: How to make it ignore the instances of the weight vector in
        # action_val_target? It should be treated as a constant. Semi-gradient
        # and stuff. Is detach() used for this?
        loss = (action_val_pred - action_val_target).pow(2)
        #print(loss.data[0][0])

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        #self.weights[action1].data -= alpha*self.weights[action1].grad.data
        #self.weights[action1].grad.data.zero_()

    def get_state_action_value_graph(self, state, action):
        if any(state < 0) or any(state > 1):
            raise ValueError("State is not normalize: %s", state)
        x = Variable(torch.from_numpy(state).float(), requires_grad=False)
        # Plain RBF
        return self.rbf(x, self.centres, self.weights[action], self.spread)
        # Normalized RBF
        # TODO: This doesn't work. Need to find an element-wise division
        # function
        #num = self.rbf(x, self.centres, self.weights[action], self.spread)
        #den = self.rbf(x, self.centres, self.ones_weights, self.spread)
        #return torch.div(num,den)

    def get_state_action_value(self, state, action):
        val = self.get_state_action_value_graph(state, action).data[0][0]
        if np.isinf(val) or np.isnan(val):
            raise ValueError("Weights diverged. state-action value is %s" % val)
        return val

    def get_weight_change(self):
        s = 0
        for a in self.action_space:
            s += self.old_weights[a]-self.weights[a].data
        return s.sum()
    
    def reset_weight_change(self):
        for a in self.action_space:
            self.old_weights[a] = self.weights[a].data.clone()

class RBFTracesLearner(RBFLearner):

    def __init__(self, action_space, observation_space, discount_factor, learning_rate,
            initial_value=0, optimizer=None, trace_factor=0):
        Learner.__init__(self)
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.initial_value = initial_value
        self.observation_space = observation_space
        self.trace_factor = trace_factor
        
        self.rbf = RBFFunction()
        centres = list(
                itertools.product(
                    np.linspace(0,1,8),
                    np.linspace(0,1,8)
                )
        )
        weights = [[self.initial_value]*len(centres)]*len(self.action_space)
        traces = [[self.initial_value]*len(centres)]*len(self.action_space)
        self.spread = Variable(torch.Tensor([100]).float(),
                requires_grad=False)
        self.centres = Variable(
                torch.from_numpy(np.array(centres)).float(),
                requires_grad=False)
        self.weights = [Variable(
                torch.from_numpy(np.array([w]).transpose()).float(),
                requires_grad=True) for w in weights]
        self.traces = [Variable(
                torch.from_numpy(np.array([t]).transpose()).float(),
                requires_grad=False) for t in traces]
        self.ones_weights = Variable(
                torch.ones(len(centres),1),
                requires_grad=False)
        self.old_weights = [None]*len(self.weights)
        self.reset_weight_change()

        if optimizer is None:
            self.optimizer = torch.optim.SGD(self.weights, lr=self.learning_rate)
        elif optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.weights, lr=self.learning_rate)

        self.prev_sars = None

    def observe_step(self, state1, action1, reward2, state2, terminal=False):
        alpha = self.learning_rate
        gamma = self.discount_factor
        lam = self.trace_factor

        if self.prev_sars is not None:
            state0, action0, reward1, state1 = self.prev_sars
            # delta
            sav0 = self.get_state_action_value(state0,action0)
            sav1 = self.get_state_action_value(state1,action1)
            delta = reward1+gamma*sav1 - sav0
            # Traces
            sav1_graph = self.get_state_action_value_graph(state1,action1)
            sav1_graph.backward(retain_graph=True)
            for a in self.action_space:
                e = self.traces[a].data
                if self.weights[a].grad is not None:
                    g = self.weights[a].grad.data
                else:
                    g = 0
                self.traces[a].data = gamma*lam*e + g
            for a in self.action_space:
                if self.weights[a].grad is not None:
                    self.weights[a].grad.data.zero_()
            # Weights
            for a in self.action_space:
                self.weights[a].data += alpha*delta*e[a]
        self.prev_sars = (state1, action1, reward2, state2)

