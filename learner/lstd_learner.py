import torch

from learner.learner import Learner

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

        self.a_mat = torch.from_numpy(self.a_mat)
        self.b_mat = torch.from_numpy(self.b_mat)
        self.weights= torch.from_numpy(self.weights)
        self.old_weights= torch.from_numpy(self.old_weights)

    def combine_state_action(self, state, action):
        self.validate_state(state)
        # Get index of given action
        action_index = self.action_space.tolist().index(action)
        result = torch.zeros([self.num_features*len(self.action_space),1])
        start_index = action_index*self.num_features
        end_index = start_index+self.num_features
        result[start_index:end_index,0] = torch.from_numpy(state)
        return result

    def combine_state_target_action(self, state):
        """Return a state-action pair corresponding to the given state, associated with the action(s) under the target policy"""
        policy = self.get_target_policy(state)
        result = torch.zeros([self.num_features*len(self.action_space),1])
        for a in range(len(policy)):
            start_index = a*self.num_features
            end_index = start_index+self.num_features
            result[start_index:end_index,0] = torch.from_numpy(policy[a]*state)
        return result

    def update_weights(self): # TODO: This is not necessarily invertible
        #self.weights = np.linalg.pinv(self.a_mat)*self.b_mat
        self.weights = self.a_mat.inverse()*self.b_mat

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
            self.a_mat += rho*x1*(x1-gamma*x2).t()
        else:
            self.a_mat += rho*x1*x1.t()
        self.b_mat += rho*reward2*x1

    def get_importance_sampling_ratio(self, state, action):
        mu = self.get_behaviour_policy(state)
        pi = self.get_target_policy(state)
        return pi[action]/mu[action]

    def get_state_action_value(self, state, action):
        self.validate_state(state)
        return (self.weights.t()*self.combine_state_action(state, action))[0][0]

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
