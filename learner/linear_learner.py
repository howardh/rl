from learner.learner import Learner

import numpy as np

import torch
from torch.autograd import Variable
        
class LinearLearner(Learner):
    def __init__(self, num_features, action_space, discount_factor,
            learning_rate):
        self.num_features = num_features
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.target_policy = self.get_epsilon_greedy(0)
        self.behaviour_policy = self.get_epsilon_greedy(0.1)

        self.weights = Variable(
                torch.from_numpy(np.random.rand(len(self.action_space), self.num_features)).float().cuda(),
                requires_grad=True)
        self.opt = torch.optim.SGD([self.weights], lr=self.learning_rate)

    def observe_step(self, state1, action1, reward2, state2, terminal=False):
        """
        state1
            Original state of the system
        action1
            Action taken by the agent at state1
        reward2
            Reward obtained for taking action1 at state1
        state2
            State reached by taking action1 at state1
        terminal : bool
            True if state2 is a terminal state.
            False otherwise
        """
        gamma = self.discount_factor
        target = reward2 + gamma * self.get_state_value(state2)
        target = Variable(torch.from_numpy(np.array([target])).float().cuda(),
                requires_grad=False)

        state_var = Variable(torch.from_numpy(state1).float().cuda(),
                requires_grad=False)
        output = torch.dot(self.weights[action1,:],state_var)

        loss = (target-output).pow(2)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def get_state_action_value(self, state, action):
        """Return the value of the given state-action pair"""
        state_var = torch.from_numpy(state).float().cuda()
        output = torch.dot(self.weights[action,:].data,state_var)
        return output
