import collections
import numpy as np
from enum import Enum
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

class LSTDLearner(Learner):
    """
    States are expected to be a column vector of values.
    i.e. The shape should be (*,1)

    Actions must be integers.

    Attributes
    ----------
    discount_factor : float
    num_features : int
        Number of features in the state
    action_space
    """

    def __init__(self, num_features, action_space, discount_factor, use_importance_sampling=False):
        Learner.__init__(self)

        self.use_importance_sampling = use_importance_sampling
        self.discount_factor = discount_factor
        self.num_features = num_features
        self.action_space = action_space

        self.a_mat = np.matrix(np.zeros([self.num_features*len(self.action_space)]*2))
        self.b_mat = np.matrix(np.zeros([self.num_features*len(self.action_space),1]))

        self.weights = np.matrix(np.zeros([self.num_features*len(self.action_space),1]))
        self.old_weights = np.matrix(np.zeros([self.num_features*len(self.action_space),1]))

    def combine_state_action(self, state, action):
        self.validate_state(state)
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
        self.weights = np.linalg.pinv(self.a_mat)*self.b_mat

    def observe_step(self, state1, action1, reward2, state2, terminal=False):
        """
        state1 : numpy.array
            A column vector representing the starting state
        """
        self.validate_state(state1)
        self.validate_state(state2)
        if self.use_importance_sampling:
            rho = self.get_importance_sampling_ratio(state1, action1)
        else:
            rho = 1
        gamma = self.discount_factor
        x1 = self.combine_state_action(state1, action1)
        if not terminal:
            x2 = self.combine_state_target_action(state2)
            self.a_mat += rho*x1*(x1-gamma*x2).transpose()
        else:
            self.a_mat += rho*x1*x1.transpose()
        self.b_mat += rho*reward2*x1

    def get_importance_sampling_ratio(self, state, action):
        mu = self.get_behaviour_policy(state)
        pi = self.get_target_policy(state)
        return pi[action]/mu[action]

    def get_state_action_value(self, state, action):
        self.validate_state(state)
        return (self.weights.transpose()*self.combine_state_action(state, action)).item(0)

    def get_weight_change(self):
        diff = np.abs(self.old_weights - self.weights)
        return np.mean(diff)
    
    def reset_weight_change(self):
        # We can simply copy the reference, because the values are never modified
        self.old_weights = self.weights

    def validate_state(self, state):
        if type(state).__module__ != np.__name__:
            raise TypeError("Invalid state: %s. Received an object of type %s. Expected an np.array with %d features." % (state, type(state), self.num_features))
        if state.size != self.num_features:
            raise ValueError("Invalid state: %s. Expected an np.array with %d features." % (state, self.num_features))

class LSTDTraceLearner(LSTDLearner):
    def __init__(self, num_features, action_space, discount_factor, trace_factor):
        LSTDLearner.__init__(self, num_features=num_features, action_space=action_space, discount_factor=discount_factor)

        self.trace_factor = trace_factor

        # Trace vector
        self.e_mat = np.matrix(np.zeros([1,self.num_features*len(self.action_space)]))

    def observe_step(self, state1, action1, reward2, state2, terminal=False):
        self.validate_state(state1)
        self.validate_state(state2)
        gamma = self.discount_factor
        lam = self.trace_factor

        x1 = self.combine_state_action(state1, action1)

        self.e_mat = lam*gamma*self.e_mat + x1.transpose()

        if not terminal:
            x2 = self.combine_state_target_action(state2)
            self.a_mat += self.e_mat.transpose()*(x1-gamma*x2).transpose()
        else:
            self.a_mat += self.e_mat.transpose()*x1.transpose()

        self.b_mat += reward2*self.e_mat.transpose()

        if terminal:
            self.e_mat *= 0

class LSTDTraceQsLearner(LSTDLearner):
    def __init__(self, num_features, action_space, discount_factor,
            trace_factor, sigma, tree_backup_policy=None):
        LSTDLearner.__init__(self, num_features=num_features, action_space=action_space, discount_factor=discount_factor)

        if trace_factor is None:
            raise TypeError("Missing Trace Factor.")
        if sigma is None:
            raise TypeError("Missing Sigma.")

        self.trace_factor = trace_factor
        self.sigma = sigma
        self.tb_policy = tree_backup_policy

        # Trace vector
        self.e_mat = np.matrix(np.zeros([1,self.num_features*len(self.action_space)]))

        self.prev_sars = None

    def get_all_state_action_pairs(self, state):
        num_actions = len(self.action_space)
        results = np.matrix(np.zeros([self.num_features*num_actions,num_actions]))
        for a in range(num_actions):
            results[:,a:(a+1)] = self.combine_state_action(state, self.action_space.item(a))
        return results

    def observe_step(self, state1, action1, reward2, state2, terminal=False):
        self.validate_state(state1)
        self.validate_state(state2)
        if self.prev_sars is not None:
            if any(state1 != self.prev_sars[3]):
                raise ValueError("States from the last two state-action pairs don't match up. Expected %s, but received %s." % (self.prev_sars[3], state1))
            state0,action0,reward1,_ = self.prev_sars

            gamma = self.discount_factor
            lam = self.trace_factor
            sigma = self.sigma

            x0 = self.combine_state_action(state0, action0)
            x1 = self.combine_state_action(state1, action1)
            x1_all = self.get_all_state_action_pairs(state1)
            pi0 = self.get_tree_backup_policy(state0).reshape([len(self.action_space),1])
            pi1 = self.get_tree_backup_policy(state1).reshape([len(self.action_space),1])

            self.e_mat = lam*gamma*((1-sigma)*pi0.item(action0)+sigma)*self.e_mat + x0.transpose()
            self.a_mat += self.e_mat.transpose()*(x0-gamma*(sigma*x1 + (1-sigma)*(x1_all*pi1))).transpose()
            self.b_mat += reward1*self.e_mat.transpose()

        if terminal:
            self.e_mat = lam*gamma*((1-sigma)*pi1[action1]+sigma)*self.e_mat + x1.transpose()
            self.a_mat += self.e_mat.transpose()*x1.transpose()
            self.b_mat += reward2*self.e_mat.transpose()

            self.e_mat *= 0
            self.prev_sars = None
        else:
            self.prev_sars = (state1, action1, reward2, state2)

    def set_tree_backup_policy(self, policy):
        self.tb_policy = policy

    def get_tree_backup_policy(self, state):
        if self.tb_policy is not None:
            return self.tb_policy(state)
        return self.get_target_policy(state)
