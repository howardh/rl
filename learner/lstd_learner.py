import numpy as np
import torch

import utils

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

    def __init__(self, num_features, action_space, discount_factor,
            use_importance_sampling=False, cuda=False):
        Learner.__init__(self)

        self.cuda = cuda
        self.use_importance_sampling = use_importance_sampling
        self.discount_factor = discount_factor
        self.num_features = num_features
        self.action_space = action_space

        self.a_mat = np.matrix(np.zeros([self.num_features*len(self.action_space)]*2))
        self.b_mat = np.matrix(np.zeros([self.num_features*len(self.action_space),1]))

        self.weights = np.matrix(np.zeros([self.num_features*len(self.action_space),1]))
        self.old_weights = np.matrix(np.zeros([self.num_features*len(self.action_space),1]))

        self.a_mat = torch.from_numpy(self.a_mat).float()
        self.b_mat = torch.from_numpy(self.b_mat).float()
        self.weights = torch.from_numpy(self.weights).float()
        self.old_weights = torch.from_numpy(self.old_weights).float()

        if self.cuda:
            self.a_mat = self.a_mat.cuda()
            self.b_mat = self.b_mat.cuda()
            self.weights = self.weights.cuda()
            self.old_weights = self.old_weights.cuda()

    def combine_state_action(self, state, action):
        self.validate_state(state)
        # Get index of given action
        action_index = self.action_space.tolist().index(action)
        result = torch.zeros([self.num_features*len(self.action_space),1]).float()
        start_index = action_index*self.num_features
        end_index = start_index+self.num_features
        result[start_index:end_index,0] = torch.from_numpy(state).float()
        if self.cuda:
            result = result.cuda()
        return result

    def combine_state_target_action(self, state):
        """Return a state-action pair corresponding to the given state, associated with the action(s) under the target policy"""
        policy = self.get_target_policy(state)
        result = torch.zeros([self.num_features*len(self.action_space),1]).float()
        for a in range(len(policy)):
            start_index = a*self.num_features
            end_index = start_index+self.num_features
            result[start_index:end_index,0] = torch.from_numpy(policy[a]*state).float()
        if self.cuda:
            result = result.cuda()
        return result

    def update_weights(self): # TODO: This is not necessarily invertible
        #self.weights = np.linalg.pinv(self.a_mat)*self.b_mat
        #self.weights = self.a_mat.inverse()*self.b_mat
        self.weights = torch.mm(utils.torch_svd_inv(self.a_mat), self.b_mat)

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
        return (torch.mm(self.weights.t(),self.combine_state_action(state, action)))[0][0]

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
    def __init__(self, num_features, action_space, discount_factor,
            trace_factor, cuda=True):
        LSTDLearner.__init__(self, num_features=num_features,
                action_space=action_space, discount_factor=discount_factor,
                cuda=cuda)

        self.trace_factor = trace_factor

        # Trace vector
        self.e_mat = torch.from_numpy(np.zeros([1,self.num_features*len(self.action_space)])).float()

        if self.cuda:
            self.e_mat = self.e_mat.cuda()

    def observe_step(self, state1, action1, reward2, state2, terminal=False):
        self.validate_state(state1)
        self.validate_state(state2)
        gamma = self.discount_factor
        lam = self.trace_factor

        x1 = self.combine_state_action(state1, action1)

        self.e_mat = lam*gamma*self.e_mat + torch.t(x1)

        if not terminal:
            x2 = self.combine_state_target_action(state2)
            self.a_mat += torch.t(self.e_mat)*torch.t(x1-gamma*x2)
        else:
            self.a_mat += torch.t(self.e_mat)*torch.t(x1)

        self.b_mat += reward2*torch.t(self.e_mat)

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

class SparseLSTDLearner(LSTDLearner):
    def __init__(self, num_features, action_space, discount_factor, use_importance_sampling=False):
        Learner.__init__(self)

        self.use_importance_sampling = use_importance_sampling
        self.discount_factor = discount_factor
        self.num_features = num_features
        self.action_space = action_space

        self.a_mat = scipy.sparse.csc_matrix((self.num_features*len(self.action_space),)*2)
        self.b_mat = scipy.sparse.csc_matrix((self.num_features*len(self.action_space),1))

        self.weights = np.matrix(np.zeros([self.num_features*len(self.action_space),1]))
        self.old_weights = np.matrix(np.zeros([self.num_features*len(self.action_space),1]))

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.ls_future = None

    def combine_state_action(self, state, action):
        self.validate_state(state)
        # Get index of given action
        action_index = self.action_space.tolist().index(action)
        result = scipy.sparse.lil_matrix((self.num_features*len(self.action_space),1))
        start_index = action_index*self.num_features
        end_index = start_index+self.num_features
        result[start_index:end_index,0] = state
        return result.tocsc()

    def combine_state_target_action(self, state):
        """Return a state-action pair corresponding to the given state, associated with the action(s) under the target policy"""
        policy = self.get_target_policy(state)
        result = scipy.sparse.lil_matrix((self.num_features*len(self.action_space),1))
        for a in range(len(policy)):
            start_index = a*self.num_features
            end_index = start_index+self.num_features
            result[start_index:end_index,0] = policy[a]*state
        return result.tocsc()

    def update_weights(self):
        self.weights = utils.solve_approx(self.a_mat, self.b_mat)

    def check_weights(self):
        if self.ls_future is None:
            print("Submitting task")
            self.ls_future = self.executor.submit(utils.solve_approx, self.a_mat, self.b_mat)
        elif self.ls_future.done():
            print("Solution found")
            self.old_weights = self.weights
            self.weights = self.ls_future.result()
            self.ls_future = None
            return True
        return False

    def observe_step(self, state1, action1, reward2, state2, terminal=False):
        """
        state1 : numpy.array
            A column vector representing the starting state
        """
        #start_time = timeit.default_timer()

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

        #elapsed = timeit.default_timer()-start_time
        #print(elapsed)

    def validate_state(self, state):
        if not scipy.sparse.issparse(state):
            raise TypeError("Invalid state: %s. Received an object of type %s. Expected a scipy.sparse column matrix with %d features." % (state, type(state), self.num_features))
        if state.shape[0] != self.num_features:
            raise ValueError("Invalid state: %s. Expected a sparse matrix with %d features. Received a matrix with shape %s" % (state, self.num_features, state.shape))
        if state.shape[1] != 1:
            raise ValueError("Invalid state: %s. Expected a column matrix. Received a matrix with shape %s" % (state, self.num_features, state.shape))

    def save(self, file_name):
        data = {'a': self.a_mat, 'b': self.b_mat}
        with open(file_name, 'wb')as f:
            dill.dump(data, f)

    def load(self, file_name):
        with open(file_name, 'rb')as f:
            data = dill.load(f)
            self.a_mat = data['a']
            self.b_mat = data['b']
