import collections
import numpy as np

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

class TabularLearner(Learner):

    def __init__(self, action_space, discount_factor, learning_rate):
        Learner.__init__(self)
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.q = collections.defaultdict(lambda: 0)
        self.old_q = self.q.copy()

    def observe_step(self, state1, action1, reward2, state2, terminal=False):
        alpha = self.learning_rate
        gamma = self.discount_factor
        delta = reward2 + gamma * self.get_state_value(state2) - self.get_state_action_value(state1, action1)
        self.q[str((state1,action1))] += alpha * delta

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

class LSTDLearner(Learner):
    """
    Attributes
    ----------
    discount_factor : float
    num_features : int
        Number of features in the state
    """

    def __init__(self, num_features, action_space):
        Learner.__init__(self)
        self.discount_factor = 0.9
        self.num_features = num_features
        self.action_space = action_space
        self.a_mat = np.matrix(np.identity(self.num_features*len(self.action_space)))*0.0001
        self.b_mat = np.matrix(np.zeros([self.num_features*len(self.action_space),1]))
        self.weights = np.matrix(np.zeros([self.num_features*len(self.action_space),1]))

    def combine_state_action(self, state, action):
        # Get index of given action
        action_index = self.action_space.tolist().index(action)
        result = np.matrix(np.zeros([self.num_features*len(self.action_space),1]))
        start_index = action_index*self.num_features
        end_index = start_index+self.num_features
        result[start_index:end_index,0] = state
        return result

    def combine_state_target_action(self, state):
        """Return a state-action pair corresponding to the given state, associated with the action(s) under the target policy"""
        policy = self.get_target_policy(state)
        result = np.matrix(np.zeros([self.num_features*len(self.action_space),1]))
        for a in range(len(policy)):
            start_index = a*self.num_features
            end_index = start_index+self.num_features
            result[start_index:end_index,0] = policy[a]*state
        return result

    def update_weights(self):
        self.weights = np.linalg.inv(self.a_mat)*self.b_mat

    def observe_step(self, state1, action1, reward2, state2, terminal=False):
        """
        state1 : numpy.array
            A column vector representing the starting state
        """
        x1 = self.combine_state_action(state1, action1)
        x2 = self.combine_state_target_action(state2)
        gamma = self.discount_factor
        a_mat[action] += x1.transpose()*(x1-gamma*x2)
        b_mat[action] += reward1*x1.transpose()

    def get_state_action_value(self, state, action):
        return self.weights.transpose()*self.combine_state_action(state, action)
