from learner.learner import Learner

import numpy as np

import torch
from torch.autograd import Variable
        
class LinearLearner(Learner):
    def __init__(self, num_features, action_space, discount_factor,
            learning_rate, trace_factor=0, replacing_traces=False, device=torch.device('cpu')):
        self.num_features = num_features
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.trace_factor = trace_factor
        self.learning_rate = learning_rate
        self.replacing_traces = replacing_traces
        self.device = device

        self.target_policy = self.get_epsilon_greedy(0)
        self.behaviour_policy = self.get_epsilon_greedy(0.1)

        #self.weights = torch.from_numpy(np.random.rand(len(self.action_space), self.num_features)).float().to(device)
        self.weights = torch.zeros((len(self.action_space), self.num_features)).float().to(device)
        self.traces = torch.zeros(self.weights.size()).float().to(device)

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
        target = torch.from_numpy(np.array([target])).float().to(self.device)

        state_tensor = torch.from_numpy(state1).float().to(self.device)
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
        state_tensor = torch.from_numpy(state).float().to(self.device)
        output = torch.dot(self.weights[action,:],state_tensor)
        return output

class LinearQsLearner(Learner):
    def __init__(self, num_features, action_space, discount_factor,
            learning_rate, trace_factor=0, sigma=0, trace_type='accumulating', device=torch.device('cpu')):
        self.device = device
        self.num_features = num_features
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.trace_factor = trace_factor
        self.learning_rate = learning_rate
        self.sigma = sigma
        if trace_type not in ['replacing', 'accumulating']:
            raise ValueError("Invalid trace type: %s. Must be either 'replacing' or 'accumulating'" % trace_type)
        self.trace_type = trace_type

        self.target_policy = self.get_epsilon_greedy(0)
        self.behaviour_policy = self.get_epsilon_greedy(0.1)

        self.weights = torch.from_numpy(np.random.rand(len(self.action_space), self.num_features)).float().to(device)
        self.traces = torch.zeros(self.weights.size()).float().to(device)

        self.prev_sars = None

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
        sigma = self.sigma
        if self.prev_sars is not None:
            state0, action0, reward1, _ = self.prev_sars

            try:
                #target = reward1 + gamma * (sigma*self.get_state_action_value(state1,action1) + (1-sigma)*self.get_state_value(state1))
                target = reward1
                target += gamma * sigma*self.get_state_action_value(state1,action1)
                target += gamma * (1-sigma)*self.get_state_value(state1)
            except Warning as w:
                print("BROKEN!")
                print(w)
            target = torch.from_numpy(np.array([target])).float().to(self.device)

            state_tensor = torch.from_numpy(state0).float().to(self.device)
            output = torch.dot(self.weights[action0,:],state_tensor.view(-1))

            delta = target-output
            self.traces *= lam*gamma
            self.traces *= ((1-sigma)*self.target_policy(state0)[action0] + sigma)
            if self.trace_type == 'accumulating':
                self.traces[action0,:] += state_tensor.view(-1)
            elif self.trace_type == 'replacing':
                self.traces[action0,:] = torch.max(self.traces[action0,:],state_tensor.view(-1))
            else:
                raise ValueError("Invalid trace type: %s" % self.trace_type)

            self.weights += alpha*delta*self.traces

        self.prev_sars = (state1, action1, reward2, state2)

        if terminal:
            state0, action0, reward1, _ = self.prev_sars

            target = reward1
            target = torch.from_numpy(np.array([target])).float().to(self.device)

            state_tensor = torch.from_numpy(state0).float().to(self.device)
            output = torch.dot(self.weights[action0,:],state_tensor.view(-1))

            delta = target-output
            self.traces *= lam*gamma
            self.traces *= ((1-sigma)*self.target_policy(state0)[action0] + sigma)
            if self.trace_type == 'accumulating':
                self.traces[action0,:] += state_tensor.view(-1)
            elif self.trace_type == 'replacing':
                self.traces[action0,:] = torch.max(self.traces[action0,:],state_tensor.view(-1))
            else:
                raise ValueError("Invalid trace type: %s" % self.trace_type)

            self.weights += alpha*delta*self.traces

            self.traces *= 0
            self.prev_sars = None

    def get_state_action_value(self, state, action):
        """Return the value of the given state-action pair"""
        state_tensor = torch.from_numpy(state).float().to(self.device)
        output = torch.dot(self.weights[action,:],state_tensor.view(-1))
        return output
