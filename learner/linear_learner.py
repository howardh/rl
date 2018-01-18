from learner.learner import Learner

import numpy as np

import torch
from torch.autograd import Variable
        
class LinearLearner(Learner):
    def __init__(self, num_features, action_space, discount_factor,
            learning_rate, trace_factor=0, replacing_traces=False):
        self.num_features = num_features
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.trace_factor = trace_factor
        self.learning_rate = learning_rate
        self.replacing_traces = replacing_traces

        self.target_policy = self.get_epsilon_greedy(0)
        self.behaviour_policy = self.get_epsilon_greedy(0.1)

        self.weights = torch.from_numpy(np.random.rand(len(self.action_space), self.num_features)).float().cuda()
        self.traces = torch.zeros(self.weights.size()).float().cuda()

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
        alpha = self.learning_rate
        gamma = self.discount_factor
        lam = self.trace_factor
        target = reward2 + gamma * self.get_state_value(state2)
        target = torch.from_numpy(np.array([target])).float().cuda()

        state_tensor = torch.from_numpy(state1).float().cuda()
        output = torch.dot(self.weights[action1,:],state_tensor)

        delta = target-output
        self.traces *= lam
        if self.replacing_traces:
            self.traces[action1,:] = torch.max(self.traces[action1,:], state_tensor)
        else:
            self.traces[action1,:] += state_tensor

        self.weights += alpha*delta*self.traces

        if terminal:
            self.traces *= 0

    def get_state_action_value(self, state, action):
        """Return the value of the given state-action pair"""
        state_tensor = torch.from_numpy(state).float().cuda()
        output = torch.dot(self.weights[action,:],state_tensor)
        return output
