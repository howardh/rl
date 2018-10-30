import numpy as np
import torch
torch.set_num_threads(1) # torch.svd takes up multiple cores
import itertools

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

        self.prev_sars = None

    def combine_state_action(self, state, action):
        self.validate_state(state)
        # Get index of given action
        action_index = self.action_space.tolist().index(action)
        result = torch.zeros([int(self.num_features*len(self.action_space)),1]).float()
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

    def update_weights(self):
        self.weights = torch.mm(utils.torch_svd_inv(self.a_mat), self.b_mat)

    def observe_step(self, state1, action1, reward2, state2, terminal=False):
        """
        state1 : numpy.array
            A column vector representing the starting state
        """
        self.validate_state(state1)
        self.validate_state(state2)
        
        gamma = self.discount_factor
        
        x1 = self.combine_state_action(state1, action1)

        if self.prev_sars is not None:
            if any(state1 != self.prev_sars[3]):
                raise ValueError("States from the last two state-action pairs don't match up. Expected %s, but received %s." % (self.prev_sars[3], state1))
            state0,action0,reward1,_ = self.prev_sars

            x0 = self.combine_state_action(state0, action0)
            self.a_mat += x0*(x0-gamma*x1).t()
            self.b_mat += reward1*x0

        if terminal:
            self.a_mat += x1*x1.t()
            self.b_mat += reward2*x1
            self.prev_sars = None
        else:
            self.prev_sars = (state1, action1, reward2, state2)

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


class LSPILearner(LSTDLearner):
    def __init__(self, num_features, action_space, discount_factor, cuda=False):
        LSTDLearner.__init__(self, num_features=num_features,
                action_space=action_space, discount_factor=discount_factor,
                cuda=cuda)
        self.data = []

    def update_weights(self): 
        gamma = self.discount_factor
        self.b_mat *= 0
        for sa0,r1,s1,t in self.data:
            self.b_mat += r1*sa0
        old_a = None
        for i in itertools.count():
            old_a = self.a_mat
            self.a_mat *= 0
            for sa0,r1,s1,t in self.data:
                sa1 = self.combine_state_target_action(s1)
                if not t:
                    self.a_mat += sa0 @ (sa0-gamma*sa1).t()
                else:
                    self.a_mat += sa0 @ sa0.t()
            self.old_weights = self.weights
            self.weights = torch.mm(utils.torch_svd_inv(self.a_mat), self.b_mat)
            w1 = torch.mm(utils.torch_svd_inv(self.a_mat), self.b_mat)
            w2 = torch.mm(utils.torch_svd_inv(self.a_mat), self.b_mat)
            diff = (self.old_weights-self.weights).pow(2).mean()
            diff2 = (w1-w2).pow(2).mean()
            print(self.old_weights-self.weights)
            print("Diff: %f" % diff2)
            if diff < 0.001:
                break

    def observe_step(self, state1, action1, reward2, state2, terminal=False):
        self.validate_state(state1)
        self.validate_state(state2)
        x1 = self.combine_state_action(state1, action1)
        self.data.append((x1, reward2, state2, terminal))


class LSTDTraceLearner(LSTDLearner):
    def __init__(self, num_features, action_space, discount_factor,
            trace_factor, cuda=False):
        LSTDLearner.__init__(self, num_features=num_features,
                action_space=action_space, discount_factor=discount_factor,
                cuda=cuda)

        self.trace_factor = trace_factor

        # Trace vector
        self.e_mat = torch.zeros([1,self.num_features*len(self.action_space)]).float()

        if self.cuda:
            self.e_mat = self.e_mat.cuda()

    def observe_step(self, state1, action1, reward2, state2, terminal=False):
        self.validate_state(state1)
        self.validate_state(state2)
        gamma = self.discount_factor
        lam = self.trace_factor

        x1 = self.combine_state_action(state1, action1)

        self.e_mat = lam*gamma*self.e_mat + x1.t()

        if not terminal:
            x2 = self.combine_state_target_action(state2)
            self.a_mat += self.e_mat.t()*torch.t(x1-gamma*x2)
        else:
            self.a_mat += self.e_mat.t()*x1.t()

        self.b_mat += reward2*self.e_mat.t()

        if terminal:
            self.e_mat *= 0


class LSTDTraceQsLearner(LSTDLearner):
    def __init__(self, num_features, action_space, discount_factor,
            trace_factor, sigma, trace_type='accumulating', decay=1):
        LSTDLearner.__init__(self, num_features=num_features, action_space=action_space, discount_factor=discount_factor)

        if trace_factor is None:
            raise TypeError("Missing Trace Factor.")
        if sigma is None:
            raise TypeError("Missing Sigma.")

        self.trace_factor = trace_factor
        self.sigma = sigma
        self.trace_type = trace_type
        self.decay = decay

        # Trace vector
        self.e_mat = torch.zeros([1,int(self.num_features*len(self.action_space))])

        self.prev_sars = None

    def get_all_state_action_pairs(self, state):
        num_actions = len(self.action_space)
        results = torch.zeros([int(self.num_features*num_actions),num_actions])
        for a in range(num_actions):
            results[:,a:(a+1)] = self.combine_state_action(state, self.action_space.item(a))
        return results

    def observe_step(self, state1, action1, reward2, state2, terminal=False):
        self.validate_state(state1)
        self.validate_state(state2)
        if self.prev_sars is not None:
            if (state1 != self.prev_sars[3]).any():
                raise ValueError("States from the last two state-action pairs don't match up. Expected %s, but received %s." % (self.prev_sars[3], state1))
            state0,action0,reward1,_ = self.prev_sars

            gamma = self.discount_factor
            lam = self.trace_factor
            sigma = self.sigma

            x0 = self.combine_state_action(state0, action0)
            x1 = self.combine_state_action(state1, action1)
            x1_all = self.get_all_state_action_pairs(state1)
            pi0 = torch.from_numpy(self.get_target_policy(state0)).float().view(len(self.action_space),1)
            pi1 = torch.from_numpy(self.get_target_policy(state1)).float().view(len(self.action_space),1)

            #self.e_mat = lam*gamma*((1-sigma)*pi0.view(-1)[action0]+sigma)*self.e_mat + x0.t()
            self.e_mat *= lam*gamma*((1-sigma)*pi0.view(-1)[action0]+sigma)
            if self.trace_type == 'accumulating':
                self.e_mat += x0.t()
            elif self.trace_type == 'replacing':
                self.e_mat = torch.max(self.e_mat, x0.t())
            else:
                raise ValueError("Invalid trace type: %s" % self.trace_type)
            self.a_mat += self.e_mat.t() @ (x0-gamma*(sigma*x1 + (1-sigma)*(x1_all @ pi1))).t()
            self.b_mat += reward1*self.e_mat.t()

        if terminal:
            #self.e_mat = lam*gamma*((1-sigma)*pi1.view(-1)[action1]+sigma)*self.e_mat + x1.t()
            self.e_mat *= lam*gamma*((1-sigma)*pi1.view(-1)[action1]+sigma)
            if self.trace_type == 'accumulating':
                self.e_mat += x1.t()
            elif self.trace_type == 'replacing':
                self.e_mat = torch.max(self.e_mat, x1.t())
            else:
                raise ValueError("Invalid trace type: %s" % self.trace_type)
            self.a_mat += self.e_mat.t() @ x1.t()
            self.b_mat += reward2*self.e_mat.t()

            self.e_mat *= 0
            self.prev_sars = None

            self.a_mat *= self.decay
            self.b_mat *= self.decay
        else:
            self.prev_sars = (state1, action1, reward2, state2)


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
