import collections
import numpy as np
from enum import Enum
import scipy.sparse
import scipy.sparse.linalg
import timeit
import dill
import concurrent.futures
import utils

#from enum import auto

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
        self.target_policy = self.get_epsilon_greedy(0)
        self.behaviour_policy = self.get_epsilon_greedy(0.1)

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
        policy = self.target_policy(state)
        return np.dot(values,policy)

    def get_state_action_value(self, state, action):
        """Return the value of the given state-action pair"""
        raise NotImplementedError

    def get_all_state_action_values(self, state):
        """Return an np.array with the values of each action at the given state"""
        return np.array([self.get_state_action_value(state, action) for action in self.action_space])

    def get_behaviour_policy(self, state):
        """Return a probability distribution over actions"""
        return self.behaviour_policy(state)

    def get_target_policy(self, state):
        """Return a probability distribution over actions"""
        return self.target_policy(state)

    def set_target_policy(self, policy):
        """
        policy
            A function which takes as input an np.array of state-action values, and returns an np.array with the probability of choosing each action.
        """
        self.target_policy = policy

    def set_behaviour_policy(self, policy):
        self.behaviour_policy = policy

    def get_epsilon_greedy(self, epsilon):
        def f(state):
            w = self.get_all_state_action_values(state)
            m = np.argmax(w)
            max_count = sum([x == w[m] for x in w])
            if len(w) == max_count:
                return np.array(len(w)*[1.0/len(w)])
            return np.array([(1-epsilon)/max_count if x == w[m] else epsilon/(len(w)-max_count) for x in w])
        return f
    
    def get_softmax(self, temperature):
        def f(state):
            w = self.get_all_state_action_values(state)
            e = np.exp(np.array(w) / temperature)
            dist = e / np.sum(e)
            return dist
        return f

class Optimizer(Enum):
    NONE = 1
    MOMENTUM = 2
    ADA_GRAD = 3
    RMS_PROP = 4
    ADAM = 5
    KSGD = 6

class TabularLearner(Learner):

    def __init__(self, action_space, discount_factor, learning_rate,
            initial_value=0, optimizer=Optimizer.NONE):
        Learner.__init__(self)
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.initial_value = initial_value
        
        self.q = collections.defaultdict(lambda: self.initial_value)
        self.old_q = self.q.copy()
        if self.optimizer == Optimizer.NONE:
            pass
        elif self.optimizer == Optimizer.RMS_PROP:
            self.v = collections.defaultdict(lambda: self.initial_value)
        else:
            raise NotImplementedError

    def observe_step(self, state1, action1, reward2, state2, terminal=False):
        alpha = self.learning_rate
        gamma = self.discount_factor
        delta = reward2 + gamma * self.get_state_value(state2) - self.get_state_action_value(state1, action1)
        index = str((state1,action1))
        if self.optimizer == Optimizer.NONE:
            self.q[index] += alpha * delta
        elif self.optimizer == Optimizer.RMS_PROP:
            forget = 0.1 # TODO: Forgetting Factor
            self.v[index] = forget*self.v[index] + (1-forget)*(delta*delta)
            if delta != 0:
                self.q[index] += alpha/np.sqrt(self.v[index])*delta
                if self.v[index] == 0:
                    print("ERROR AND SHIT")
        else:
            raise NotImplementedError

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

