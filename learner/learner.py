import collections
import torch
import numpy as np
from enum import Enum
import scipy.sparse
import scipy.sparse.linalg
import timeit
import dill
import concurrent.futures
import utils

#from enum import auto
from agent.policy import get_greedy_epsilon_policy

class Learner(object):
    """
    The Leaner takes a sequence of observations and actions, and uses this data to learn about a target policy.

    Attributes
    ----------
    action_space : numpy.array
    target_policy : callable
        numpy.array -> numpy.array
    behaviour_policy : callable
        numpy.array -> numpy.array
    """

    def __init__(self):
        self.action_space = None

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
        pass

    def get_state_value(self, state):
        """Return the value of the given state"""
        values = self.get_all_state_action_values(state)
        policy = self.target_policy(values)
        return (values @ policy.probs.t()).item()

    def get_state_action_value(self, state, action):
        """Return the value of the given state-action pair"""
        raise NotImplementedError

    def get_all_state_action_values(self, state):
        """Return an np.array with the values of each action at the given state"""
        return torch.tensor([[self.get_state_action_value(state, action) for action in range(self.action_space.n)]]).float()

    def get_behaviour_policy(self, state):
        """Return a probability distribution over actions"""
        return self.behaviour_policy(state)

    def get_target_policy(self, state):
        """Return a probability distribution over actions"""
        values = self.get_all_state_action_values(state)
        return self.target_policy(values)

    def set_target_policy(self, policy):
        """
        policy
            A function which takes as input an np.array of state-action values, and returns an np.array with the probability of choosing each action.
        """
        self.target_policy = policy

    def set_behaviour_policy(self, policy):
        self.behaviour_policy = policy

    def get_softmax(self, temperature):
        def f(state):
            w = self.get_all_state_action_values(state)
            e = np.exp(np.array(w) / temperature)
            dist = e / np.sum(e)
            return dist
        return f

class TabularLearner(Learner):

    def __init__(self, action_space, discount_factor, learning_rate,
            initial_value=0):
        Learner.__init__(self)
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.initial_value = initial_value
        
        self.q = collections.defaultdict(lambda: self.initial_value)
        self.old_q = self.q.copy()

    def observe_step(self, state1, action1, reward2, state2, terminal=False):
        alpha = self.learning_rate
        gamma = self.discount_factor
        delta = reward2 + gamma * self.get_state_value(state2) - self.get_state_action_value(state1, action1)
        index = str((state1,action1))
        self.q[index] += alpha * delta

    def get_state_action_value(self, state, action):
        return self.q[str((state,action))]
        
    def get_weight_change(self):
        change_sum = 0
        val_count = 0
        for key, val in self.q.items():
            val_count += 1
            diff = abs(self.q[key] - self.old_q[key])
            change_sum += diff
        return change_sum/val_count
    
    def reset_weight_change(self):
        self.old_q = self.q.copy()

class TabularQsLearner(Learner):

    def __init__(self, action_space, discount_factor, learning_rate,
            trace_factor=0, sigma=0, initial_value=0,
            target_policy=get_greedy_epsilon_policy(0)):
        super().__init__()
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.initial_value = initial_value
        self.trace_factor = trace_factor
        self.sigma = sigma
        self.target_policy = target_policy
        
        self.q = collections.defaultdict(lambda: self.initial_value)
        self.old_q = self.q.copy()
        self.e = collections.defaultdict(lambda: 0)

        self.prev_sars = None

    def observe_step(self, state1, action1, reward2, state2, terminal=False):
        alpha = self.learning_rate
        gamma = self.discount_factor
        lam = self.trace_factor
        sigma = self.sigma
        if self.prev_sars is not None:
            state0, action0, reward1, _ = self.prev_sars

            # delta
            v = self.get_state_value(state1)
            q = self.get_state_action_value(state1,action1)
            delta = reward1 + gamma*(sigma*q+(1-sigma)*v) - self.get_state_action_value(state0, action0)

            # Trace
            p = self.get_target_policy(state0).probs[0,action0] # TODO: Should this be state0 or state1?
            decay = gamma*lam*((1-sigma)*p+sigma)
            for k in self.e.keys():
                self.e[k] *= decay

            index = str((state0,action0))
            self.e[index] += 1

            # Weights
            for k in self.q.keys():
                self.q[k] += alpha * delta * self.e[k]

        self.prev_sars = (state1,action1,reward2,state2)

        if terminal:
            state0, action0, reward1, _ = self.prev_sars

            # delta
            delta = reward1 - self.get_state_action_value(state0, action0)

            # Trace
            p = self.get_target_policy(state0).probs[0,action0] # TODO: Should this be state0 or state1?
            decay = gamma*lam*((1-sigma)*p+sigma)
            for k in self.e.keys():
                self.e[k] *= decay

            index = str((state0,action0))
            self.e[index] += 1

            # Weights
            for k in self.q.keys():
                self.q[k] += alpha * delta * self.e[k]

            # Reset everything
            for k in self.e.keys():
                self.e[k] = 0
            self.prev_sars = None

    def get_state_action_value(self, state, action):
        return self.q[str((state,action))]
        
    def get_weight_change(self):
        change_sum = 0
        val_count = 0
        for key, val in self.q.items():
            val_count += 1
            diff = abs(self.q[key] - self.old_q[key])
            change_sum += diff
        return change_sum/val_count
    
    def reset_weight_change(self):
        self.old_q = self.q.copy()
