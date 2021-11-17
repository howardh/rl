import copy
import itertools
from collections import defaultdict
from typing import Dict, Generic, TypeVar, Optional, Tuple, List, Any, Union
from typing_extensions import TypedDict

import numpy as np
import torch
import torch.nn
import torch.utils.data
import torch.distributions
import torch.optim
import gym.spaces
from torchtyping import TensorType

from experiment.logger import Logger, SubLogger
import rl.debug_tools.frozenlake
from rl.agent.agent import DeployableAgent

class PolicyValueNetworkOutput(TypedDict, total=False):
    """ Discrete action space """
    value: torch.Tensor
    # Discrete action spaces
    action: torch.Tensor
    # Continuous action spaces
    action_mean: torch.Tensor
    action_std: torch.Tensor
    # Recurrent networks
    hidden: Tuple[torch.Tensor,torch.Tensor]
class QNetworkCNN(torch.nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,out_channels=32,kernel_size=8,stride=4),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=32,out_channels=64,kernel_size=4,stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=64,out_channels=64,kernel_size=3,stride=1),
            torch.nn.LeakyReLU(),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=64*7*7,out_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=512,out_features=num_actions),
        )
    def forward(self, obs):
        x = obs
        x = self.conv(x)
        x = x.view(-1,7*7*64)
        x = self.fc(x)
        return x
class PolicyValueNetwork(torch.nn.Module):
    """ A model which acts both as a policy network and a state-value estimator. """
    def __init__(self):
        super().__init__()
    def __call__(self, *args, **kwargs) -> PolicyValueNetworkOutput:
        return super().__call__(*args, **kwargs)
    def forward(self, x) -> PolicyValueNetworkOutput:
        x = x
        raise NotImplementedError()
class PolicyValueNetworkRecurrent(PolicyValueNetwork):
    def __init__(self):
        super().__init__()
    def __call__(self, *args, **kwargs) -> PolicyValueNetworkOutput:
        return super().__call__(*args, **kwargs)
    def forward(self, x, hidden) -> PolicyValueNetworkOutput:
        x = x
        hidden = hidden
        raise NotImplementedError()
    def init_hidden(self, batch_size=1) -> Tuple[torch.Tensor,torch.Tensor]:
        batch_size=batch_size
        raise NotImplementedError()

class PolicyValueNetworkCNN(PolicyValueNetwork):
    """ A model which acts both as a policy network and a state-value estimator. """
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,out_channels=32,kernel_size=8,stride=4),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=32,out_channels=64,kernel_size=4,stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=64,out_channels=64,kernel_size=3,stride=1),
            torch.nn.LeakyReLU(),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=64*7*7,out_features=512),
            torch.nn.LeakyReLU(),
        )
        self.v = torch.nn.Linear(in_features=512,out_features=1)
        self.pi = torch.nn.Linear(in_features=512,out_features=num_actions)
    def forward(self, x) -> PolicyValueNetworkOutput:
        x = self.conv(x)
        x = x.view(-1,64*7*7)
        x = self.fc(x)
        v = self.v(x)
        pi = self.pi(x)
        return {
                'value': v,
                'action': pi, # Unnormalized action probabilities
        }
class PolicyValueNetworkLinear(PolicyValueNetwork):
    """ A model which acts both as a policy network and a state-value estimator. """
    def __init__(self, num_features, num_actions,bias=False):
        super().__init__()
        self.num_actions = num_actions
        self.num_features = num_features
        self.v = torch.nn.Linear(in_features=num_features,out_features=1,bias=bias)
        self.pi = torch.nn.Linear(in_features=num_features,out_features=num_actions,bias=bias)
    def forward(self, x) -> PolicyValueNetworkOutput:
        v = self.v(x)
        pi = self.pi(x)
        return {
                'value': v,
                'action': pi # Unnormalized action probabilities
        }
class PolicyValueNetworkFCNN(PolicyValueNetwork):
    """ A model which acts both as a policy network and a state-value estimator. """
    def __init__(self, num_features : int,
            num_actions : int,
            structure : List[int] = [256,256],
            hidden_size : int = 128,
            shared_std : bool = True):
        super().__init__()
        self.num_actions = num_actions
        self.num_features = num_features
        self.hidden_size = hidden_size

        layers = []
        for in_size, out_size in zip([num_features,*structure],structure):
            layers.append(torch.nn.Linear(in_features=in_size,out_features=out_size))
            layers.append(torch.nn.ReLU())
        self.fc = torch.nn.Sequential(*layers)
        fc_output_size = structure[-1] if len(structure) > 0 else num_features
        self.v = torch.nn.Linear(in_features=fc_output_size,out_features=1)
        self.pi_mean = torch.nn.Linear(in_features=fc_output_size,out_features=num_actions)
        if shared_std:
            self.pi_std = torch.nn.Parameter(torch.zeros(num_actions))
        else:
            self.pi_std = torch.nn.Linear(in_features=fc_output_size,out_features=num_actions)
    def forward(self, x) -> PolicyValueNetworkOutput:
        x = self.fc(x)
        v = self.v(x)
        pi_mean = self.pi_mean(x)
        if isinstance(self.pi_std, torch.nn.Parameter):
            pi_std = torch.log(1+self.pi_std.exp())
        else:
            pi_std = torch.log(1+self.pi_std(x).exp())
        pi_std += 1e-6 # For numerical stability
        return {
                'value': v,
                'action_mean': pi_mean,
                'action_std': pi_std,
        }
class PolicyValueNetworkRecurrentFCNN(PolicyValueNetworkRecurrent):
    """ A model which acts both as a policy network and a state-value estimator. """
    def __init__(self, num_features : int,
            num_actions : int,
            structure : List[int] = [256,256],
            hidden_size : int = 128,
            shared_std : bool = True):
        super().__init__()
        self.num_actions = num_actions
        self.num_features = num_features
        self.hidden_size = hidden_size

        layers = []
        for in_size, out_size in zip([num_features,*structure],structure):
            layers.append(torch.nn.Linear(in_features=in_size,out_features=out_size))
            layers.append(torch.nn.ReLU())
        self.fc = torch.nn.Sequential(*layers)
        self.lstm = torch.nn.LSTMCell(input_size=structure[-1],hidden_size=hidden_size)
        self.v = torch.nn.Linear(in_features=hidden_size,out_features=1)
        self.pi_mean = torch.nn.Linear(in_features=hidden_size,out_features=num_actions)
        if shared_std:
            self.pi_std = torch.nn.Parameter(torch.zeros(num_actions))
        else:
            self.pi_std = torch.nn.Linear(in_features=hidden_size,out_features=num_actions)
    def forward(self, x, hidden) -> PolicyValueNetworkOutput:
        x = self.fc(x)
        h,x = self.lstm(x,hidden)
        v = self.v(x)
        pi_mean = self.pi_mean(x)
        if isinstance(self.pi_std, torch.nn.Parameter):
            pi_std = torch.log(1+self.pi_std.exp())
        else:
            pi_std = torch.log(1+self.pi_std(x).exp())
        return {
                'value': v,
                'action_mean': pi_mean,
                'action_std': pi_std,
                'hidden': (h,x)
        }
    def init_hidden(self, batch_size=1):
        return (
                torch.zeros([batch_size,self.hidden_size]),
                torch.zeros([batch_size,self.hidden_size]),
        )

ObsType = TypeVar('ObsType')
ActionType = TypeVar('ActionType')
class ObservationStack(Generic[ObsType,ActionType]):
    def __init__(self) -> None:
        self.prev : Optional[Tuple[ObsType,Optional[float],bool]] = None
        self.curr : Optional[Tuple[ObsType,Optional[float],bool]] = None
        self.prev_a : Optional[ActionType] = None
        self.curr_a : Optional[ActionType] = None
    def append_obs(self, obs : ObsType, reward : Optional[float], terminal : bool):
        if self.curr is not None and self.curr[2]: # If the last observation was terminal
            self.prev = None
            self.curr = None
            self.prev_a = None
            self.curr_a = None
        else:
            self.prev = self.curr
            self.prev_a = self.curr_a
        self.curr = (obs, reward, terminal)
        self.curr_a = None
    def append_action(self, action : ActionType):
        if self.curr_a is not None:
            raise Exception('`append_action` was called twice in a row without a new observation. Make sure to call `append_obs` each time time an observation is received.')
        self.curr_a = action
    def get_transition(self) -> Optional[Tuple[ObsType,ActionType,float,ObsType,bool]]:
        """ Return the most recent transition tuple.

        Returns:
            Tuple: (`s0`, `a0`, `r1`, `s1`, `t1`)

                - `s0`: Starting state
                - `a0`: Action taken at state `s0`
                - `r1`: Reward obtained from taking action `a0` at state `s0` and transitioning to `s1`
                - `s1`: State reached by taking action `a0` at state `s0`
                - `t1`: `True` if this transition is terminal, and `False` otherwise.
            If no transition is available, return None.
        """
        if self.prev is None or self.curr is None:
            return None
        if self.prev_a is None:
            return None
        s0, _, _ = self.prev
        a0 = self.prev_a
        s1, r1, t1 = self.curr
        if r1 is None:
            return None
        return s0, a0, r1, s1, t1
    def get_obs(self) -> ObsType:
        if self.curr is None:
            raise Exception('No observation available.')
        return self.curr[0]
    def state_dict(self):
        return {
                'prev': self.prev,
                'curr': self.curr,
                'prev_a': self.prev_a,
                'curr_a': self.curr_a
        }
    def load_state_dict(self,state):
        self.prev = state['prev']
        self.curr = state['curr']
        self.prev_a = state['prev_a']
        self.curr_a = state['curr_a']

def compute_mc_state_value_loss(
        state_values : List[torch.Tensor],
        last_state_target_value: float,
        rewards : List[Optional[float]],
        terminals : List[bool],
        discounts : List[float],
    ) -> torch.Tensor:
    """ Monte-Carlo loss for a state-value function.
    """
    device = state_values[0].device
    num_transitions = len(state_values)-1
    losses = []
    for j in range(num_transitions):
        if terminals[j]:
            losses.append(torch.tensor(0,device=device))
            continue
        # Value of the state predicted by the model
        v_pred = state_values[j]
        # Value of the state sampled from a partial rollout and bootstrapped
        v_target = torch.tensor(0., device=device)
        discount = 1
        for i in range(j+1,num_transitions+1):
            reward = rewards[i]
            if reward is None:
                raise Exception()
            v_target += discount*reward
            discount *= discounts[i-1]
            if terminals[i]:
                break
            # Bootstrap on the final iteration
            if i == num_transitions:
                v_target += discount*last_state_target_value
                break
        v_target.detach()
        # Loss
        loss = (v_pred-v_target)**2
        losses.append(loss.squeeze())
    return torch.stack(losses)
def compute_mc_state_value_loss_tensor(
        state_values : List[torch.Tensor],
        last_state_target_value: float,
        rewards : List[Optional[float]],
        terminals : List[bool],
        discounts : List[float],
    ) -> torch.Tensor:
    """ Monte-Carlo loss for a state-value function.
    """
    n = len(state_values)
    reward = torch.tensor([0 if r is None else r for r in rewards[1:]] + [last_state_target_value])
    discount = torch.tensor(discounts)
    discount_mat = torch.eye(n)[:-1,:]
    for i in range(n-1):
        discount_mat[i,i+1:] = discount[i:-1]
    for i in range(n-1):
        discount_mat[:i+1,i+1] *= discount_mat[:i+1,i]
        if terminals[i]:
            discount_mat[:i+1,i+1:] = 0
    state_value = torch.stack(state_values[:-1])
    loss = torch.tensor(terminals[:-1]).logical_not()*(state_value-discount_mat@reward)**2
    return loss
def compute_mc_state_value_loss_tensor_batch(
        state_values : TensorType['batch_size','num_steps', float],
        last_state_target_value: TensorType['batch_size', float],
        rewards : TensorType['batch_size','num_steps', float],
        terminals : TensorType['batch_size','num_steps', bool],
        discounts : TensorType['batch_size','num_steps', float],
    ) -> TensorType['batch_size','num_steps']:
    n = state_values.shape[0]
    reward = torch.tensor([0 if r is None else r for r in rewards[1:]] + [last_state_target_value])
    discount = torch.tensor(discounts)
    discount_mat = torch.eye(n)[:-1,:]
    for i in range(n-1):
        discount_mat[i,i+1:] = discount[i:-1]
    for i in range(n-1):
        discount_mat[:i+1,i+1] *= discount_mat[:i+1,i]
        if terminals[i]:
            discount_mat[:i+1,i+1:] = 0
    state_value = torch.stack(state_values[:-1])
    loss = torch.tensor(terminals[:-1]).logical_not()*(state_value-discount_mat@reward)**2
    breakpoint()
    return loss
def compute_advantage_policy_gradient(
        log_action_probs : List[torch.Tensor],
        target_state_values : Union[List[torch.Tensor],torch.Tensor],
        rewards : List[Optional[float]],
        terminals : List[bool],
        discounts : List[float],
    ) -> torch.Tensor:
    """
    Advantage policy gradient for discrete action spaces.
    Given a sequence of length n, we observe states/actions/rewards r_0,s_0,a_0,r_1,s_1,a_1,r_2,s_2,...,r_{n-1},s_{n-1},a_{n-1}.

    Args:
        log_action_probs: A list of length n (last element is ignored). The element at index i is a 0-dimensional tensor containing the log probability of selection action a_i at state s_i.
        target_state_values: A list of length n or a 1D tensor. The element at index i is the state value of state s_i as predicted by the target Q network.
        rewards: A list of length n (first element is ignored). The element at index i is r_{i+1}. The value can be None if state s_{i} was terminal and s_{i+1} is the initial state of a new episode.
        terminals: A list of length n. The element at index i is True if s_i is a terminal state, and False otherwise.
        discounts: ???
    """
    device = log_action_probs[0].device
    def compute_v_targets(target_state_values, rewards, terminals, discounts):
        vt = [0]*(len(target_state_values)+1)
        if not terminals[-1]:
            vt[-1] = target_state_values[-1].item()
        for i in reversed(range(1,len(target_state_values))):
            if not terminals[i-1]:
                vt[i] = rewards[i]+discounts[i]*vt[i+1]
        return vt[1:-1]
    vt = compute_v_targets(target_state_values, rewards, terminals, discounts)
    #vt2 = []
    losses = []
    for j in range(len(target_state_values)-1):
        if terminals[j]:
            losses.append(torch.tensor(0,device=device))
            continue
        # Value of the state predicted by the model
        v_pred = target_state_values[j]
        # Value of the state sampled from a partial rollout and bootstrapped
        #v_target = torch.tensor(0., device=device)
        #discount = 1
        #for i in range(j+1,len(target_state_values)):
        #    reward = rewards[i]
        #    if reward is None:
        #        raise Exception()
        #    v_target += discount*reward
        #    discount *= discounts[i-1]
        #    if terminals[i]:
        #        break
        #    # Bootstrap on the final iteration
        #    if i == len(target_state_values)-1:
        #        v_target += discount*target_state_values[i].item()
        #        break
        #if not (v_target-vt[j]).abs().item() < 1e-6:
        #    breakpoint()
        #vt2.append(v_target) # XXX: DEBUG
        v_target = vt[j]
        # Loss
        advantage = (v_target-v_pred).detach()
        loss = -log_action_probs[j]*advantage # The loss is stated as a gradient ascent loss in RL papers, so we take the negative for gradient descent
        losses.append(loss)
    #if any(terminals):
    #    breakpoint()
    #breakpoint()
    return torch.stack(losses)
def compute_advantage_policy_gradient_tensor(
        log_action_probs : TensorType['num_steps', float],
        target_state_values : TensorType['num_steps', float],
        rewards : TensorType['num_steps', float],
        terminals : TensorType['num_steps', bool],
        discounts : TensorType['num_steps', float],
    ) -> torch.Tensor:
    """
    Advantage policy gradient for discrete action spaces.
    Given a sequence of length n, we observe states/actions/rewards r_0,s_0,a_0,r_1,s_1,a_1,r_2,s_2,...,r_{n-1},s_{n-1},a_{n-1}.

    Args:
        log_action_probs: A list of length n (last element is ignored). The element at index i is a 0-dimensional tensor containing the log probability of selection action a_i at state s_i.
        target_state_values: A list of length n or a 1D tensor. The element at index i is the state value of state s_i as predicted by the target Q network.
        rewards: A list of length n (first element is ignored). The element at index i is r_{i+1}. The value can be None if state s_{i} was terminal and s_{i+1} is the initial state of a new episode.
        terminals: A list of length n. The element at index i is True if s_i is a terminal state, and False otherwise.
        discounts: ???
    """
    device = log_action_probs.device
    losses = []
    for j in range(len(target_state_values)-1):
        if terminals[j]:
            losses.append(torch.tensor(0,device=device))
            continue
        # Value of the state predicted by the model
        v_pred = target_state_values[j]
        # Value of the state sampled from a partial rollout and bootstrapped
        v_target = torch.tensor(0., device=device)
        discount = 1
        for i in range(j+1,len(target_state_values)):
            reward = rewards[i]
            if reward is None:
                raise Exception()
            v_target += discount*reward
            discount *= discounts[i-1]
            if terminals[i]:
                break
            # Bootstrap on the final iteration
            if i == len(target_state_values)-1:
                v_target += discount*target_state_values[i]
                break
        # Loss
        advantage = (v_target-v_pred).detach()
        loss = -log_action_probs[j]*advantage # The loss is stated as a gradient ascent loss in RL papers, so we take the negative for gradient descent
        losses.append(loss)
    return torch.stack(losses)

def compute_entropy(
        log_probs : torch.Tensor
    ) -> torch.Tensor:
    return -(log_probs.exp()*log_probs).sum()

def compute_normal_log_prob(mean,std,sample):
    dist = torch.distributions.normal.Normal(mean,std)
    return dist.log_prob(torch.tensor(sample)).sum()
def compute_normal_entropy(mean,std):
    dist = torch.distributions.normal.Normal(mean,std)
    return dist.entropy().sum()

class A2CAgent(DeployableAgent):
    """
    Implementation of A2C.

    Args:
        action_space: Gym action space.
        observation_space: Gym observation space.
    """
    def __init__(self,
            action_space : gym.spaces.Box,
            observation_space : gym.spaces.Box,
            discount_factor : float = 0.99,
            learning_rate : float = 1e-4,
            target_update_frequency : int = 40_000, # Mnih 2016 - section 8
            polyak_rate : float = 1.0,              # Mnih 2016 - section 8
            max_rollout_length : int = 5,           # Mnih 2016 - section 8 (t_max)
            training_env_keys : List = [],
            obs_scale : float = 1/255,
            reward_scale : float = 1,
            device : torch.device = torch.device('cpu'),
            net : PolicyValueNetwork = None,
            logger : Logger = None,
        ):
        self.action_space = action_space
        self.observation_space = observation_space

        self.discount_factor = discount_factor
        self.target_update_frequency = target_update_frequency
        self.polyak_rate = polyak_rate
        self.max_rollout_length = max_rollout_length
        self.obs_scale = obs_scale
        self.reward_scale = reward_scale
        self.device = device
        self.training_env_keys = training_env_keys

        # Validate input
        if max_rollout_length < 2:
            raise ValueError(f'Rollout length must be >= 2. Received {max_rollout_length}. Note that a rollout length of n consists of n-1 transitions.')

        # State (training)
        self.obs_stack = defaultdict(lambda: ObservationStack())
        self.obs_history = defaultdict(lambda: [])
        self.action_history = defaultdict(lambda: [])
        self.reward_history = defaultdict(lambda: [])
        self.terminal_history = defaultdict(lambda: [])
        self.discount_history = defaultdict(lambda: [])
        self.next_action = defaultdict(lambda: None)
        self._steps = 0 # Number of steps experienced
        self._training_steps = 0 # Number of training steps experienced

        if net is None:
            self.net = self._init_default_net(observation_space,action_space,device)
        else:
            self.net = net
            self.net.to(device)
        self.target_net = copy.deepcopy(self.net)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        # Logging
        if logger is None:
            self.logger = Logger(key_name='step', allow_implicit_key=True)
            self.logger.log(step=0)
        else:
            self.logger = logger
        self.state_values_current = [] # State values of current training loop
        self.state_values = []
        self.state_values_std = []
        self.state_values_std_ra = []
    def _init_default_net(self, observation_space, action_space,device):
        if isinstance(observation_space, gym.spaces.Box):
            assert observation_space.shape is not None
            if len(observation_space.shape) == 1: # Mujoco
                return PolicyValueNetworkFCNN(
                        observation_space.shape[0],
                        action_space.shape[0],
                        shared_std=False
                ).to(device)
            if len(observation_space.shape) == 3: # Atari
                return PolicyValueNetworkCNN(
                        action_space.n
                ).to(device)
        raise Exception('Unsupported observation space or action space.')

    def observe(self, obs, reward=None, terminal=False, testing=False, time=1, discount=None, env_key=None):
        if env_key is None:
            env_key = testing
        if discount is None:
            discount = self.discount_factor

        obs = torch.tensor(obs)*self.obs_scale
        if reward is not None:
            reward *= self.reward_scale

        self.obs_stack[env_key].append_obs(obs, reward, terminal)

        # Choose next action
        net_output = self.net(
                obs.unsqueeze(0).float().to(self.device)
        )
        if isinstance(self.action_space, gym.spaces.Discrete):
            softmax = torch.nn.Softmax(dim=1)
            assert 'action' in net_output
            action_probs_unnormalized = net_output['action']
            action_probs = softmax(action_probs_unnormalized).squeeze()
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()
        elif isinstance(self.action_space, gym.spaces.Box):
            assert 'action_mean' in net_output
            assert 'action_std' in net_output
            action_dist = torch.distributions.Normal(net_output['action_mean'],net_output['action_std'])
            action = action_dist.sample().cpu().numpy()
        else:
            raise NotImplementedError('Unsupported action space of type %s' % type(self.action_space))

        assert 'value' in net_output
        state_value = net_output['value']
        self.state_values_current.append(state_value.item())

        self.next_action[env_key] = action
        self.obs_stack[env_key].append_action(action)

        # Save training data and train
        if not testing:
            self._steps += 1
            if not isinstance(self.logger, SubLogger):
                self.logger.log(step=self._steps)

            self.logger.append(debug_state_value=state_value.item())

            self.obs_history[env_key].append(obs)
            self.reward_history[env_key].append(reward)
            self.terminal_history[env_key].append(terminal)
            self.discount_history[env_key].append(discount**time)
            self.action_history[env_key].append(action)

            self._train()
            self._update_target()
    def _train(self):
        t_max = self.max_rollout_length
        log_softmax = torch.nn.LogSoftmax(dim=1)

        # Check if we've accumulated enough data
        num_points = [len(self.obs_history[k]) for k in self.training_env_keys]
        if min(num_points) >= t_max:
            if max(num_points) != min(num_points):
                raise Exception('This should not happen')
            try:
                net_output = {ek: [
                    self.net(o.unsqueeze(0).float().to(self.device))
                    for o in self.obs_history[ek]
                ] for ek in self.training_env_keys}
                state_values = {ek: [
                    output['value'].squeeze()
                    for output in outputs
                ] for ek,outputs in net_output.items()}
                target_state_values = {ek: [
                    self.target_net(o.unsqueeze(0).float().to(self.device))['value'].squeeze()
                    for o in self.obs_history[ek]
                ] for ek in self.training_env_keys}
                if isinstance(self.action_space,gym.spaces.Discrete):
                    log_action_probs = {ek: [
                        log_softmax(o['action']).squeeze()[a]
                        for o,a in zip(outputs,self.action_history[ek])
                    ] for ek,outputs in net_output.items()}
                    entropy = {ek: [
                        compute_entropy(log_softmax(o['action']).squeeze())
                        for o in outputs
                    ] for ek,outputs in net_output.items()}
                else:
                    log_action_probs = {ek: [
                        compute_normal_log_prob(o['action_mean'],o['action_std'],a)
                        for o,a in zip(outputs,self.action_history[ek])
                    ] for ek,outputs in net_output.items()}
                    entropy = {ek: [
                        compute_normal_entropy(o['action_mean'],o['action_std'])
                        for o in outputs
                    ] for ek,outputs in net_output.items()}
            except:
                raise
            # Train policy
            loss_pi = [compute_advantage_policy_gradient(
                        log_action_probs = log_action_probs[ek],
                        target_state_values = target_state_values[ek],
                        rewards = self.reward_history[ek],
                        terminals = self.terminal_history[ek],
                        discounts = self.discount_history[ek],
                ) for ek in self.training_env_keys]
            loss_pi = torch.stack(loss_pi).mean()
            # Train value network
            loss_v = [compute_mc_state_value_loss(
                        state_values = state_values[ek],
                        last_state_target_value = float(target_state_values[ek][-1].item()),
                        rewards = self.reward_history[ek],
                        terminals = self.terminal_history[ek],
                        discounts = self.discount_history[ek],
                ) for ek in self.training_env_keys]
            loss_v = torch.stack(loss_v).mean()
            # Entropy
            loss_entropy = [
                    torch.stack(e).mean()
                    for e in entropy.values()
            ]
            loss_entropy = -torch.stack(loss_entropy).mean()
            # Take a gradient step
            loss = loss_pi+loss_v+0.01*loss_entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Check weights
            if __debug__:
                psum = 0
                for p in self.net.parameters():
                    psum += p.mean()
                #breakpoint()
            # Clear data
            for ek in self.training_env_keys:
                self.obs_history[ek] = self.obs_history[ek][-1:]
                self.reward_history[ek] = self.reward_history[ek][-1:]
                self.terminal_history[ek] = self.terminal_history[ek][-1:]
                self.discount_history[ek] = self.discount_history[ek][-1:]
                self.action_history[ek] = self.action_history[ek][-1:]
            # Log
            self._training_steps += 1
            self.state_values.append(np.mean(self.state_values_current))
            self.state_values_std.append(np.std(self.state_values_current))
            self.state_values_std_ra.append(np.mean(self.state_values_std) if self._training_steps < 100 else np.mean(self.state_values_std[-100:]))
            self.state_values_current = []
    def _update_target(self):
        if self._steps % self.target_update_frequency != 0:
            return
        tau = self.polyak_rate
        for p1,p2 in zip(self.target_net.parameters(), self.net.parameters()):
            p1.data = (1-tau)*p1+tau*p2

    def act(self, testing=False, env_key=None):
        """ Return a random action according to an epsilon-greedy policy. """
        if env_key is None:
            env_key = testing
        action = self.next_action[env_key]
        if action is None:
            raise Exception('No action found. Agent must observe the environment before taking an action.')

        if isinstance(self.action_space, gym.spaces.Discrete):
            return action
        elif isinstance(self.action_space, gym.spaces.Box):
            high = self.action_space.high
            low = self.action_space.low
            d = high-low
            assert isinstance(d,np.ndarray) # XXX: Workaround for https://github.com/microsoft/pylance-release/issues/1619. Remove when this gets fixed.
            scale = d/2
            bias = (high+low)/2
            scaled_action = np.tanh(action)*scale+bias
            return scaled_action
        else:
            raise NotImplementedError('Unsupported action space of type %s' % type(self.action_space))

    def state_dict(self):
        return {
                # Models
                'pi_net': self.net.state_dict(),
                # Optimizers
                'optimizer': self.optimizer.state_dict(),
                # Misc
                'obs_stack': {k:os.state_dict() for k,os in self.obs_stack.items()},
                'steps': self._steps,
                'training_steps': self._training_steps,
        }
    def load_state_dict(self, state):
        # TODO
        # Misc
        self._steps = state['steps']
        self._training_steps = state['training_steps']

    def state_dict_deploy(self):
        return {
            'action_space': self.action_space,
            'observation_space': self.observation_space,
            # TODO
        }
    def load_state_dict_deploy(self, state):
        state = state
        pass # TODO

class A2CAgentRecurrent(A2CAgent):
    def __init__(self, episodes_per_batch=16, reward_scale=1, net : PolicyValueNetworkRecurrent = None, **kwargs):
        super().__init__(net=net, **kwargs)
        self.episodes_per_batch = episodes_per_batch
        self.reward_scale = reward_scale

        self._training_data = []

        if net is None:
            raise Exception('Models must be provided. `net` is missing.')
        self.net = net

        self.last_hidden : Dict[Any,Any] = defaultdict(lambda: None)
    def observe(self, obs, reward=None, terminal=False, testing=False, time=1, discount=None, env_key=None):
        if env_key is None:
            env_key = testing
        if discount is None:
            discount = self.discount_factor

        if reward is not None:
            reward *= self.reward_scale
        obs = torch.tensor(obs)*self.obs_scale

        self.obs_stack[env_key].append_obs(obs, reward, terminal)

        # Choose next action
        last_hidden = self.last_hidden[env_key]
        if last_hidden is None:
            last_hidden = self.net.init_hidden()
        net_output = self.net(
                obs.unsqueeze(0).float().to(self.device),
                last_hidden
        )
        assert 'action_mean' in net_output
        assert 'action_std' in net_output
        assert 'value' in net_output
        assert 'hidden' in net_output
        assert net_output['hidden'] is not None
        action_dist = torch.distributions.Normal(net_output['action_mean'],net_output['action_std'])
        action = action_dist.sample().cpu().numpy()
        self.last_hidden[env_key] = net_output['hidden']

        state_value = net_output['value']
        self.state_values_current.append(state_value.item())

        self.next_action[env_key] = action
        self.obs_stack[env_key].append_action(action)

        # Save training data and train
        if not testing:
            self._steps += 1
            self.logger.log(step=self._steps)

            self.obs_history[env_key].append(obs)
            self.reward_history[env_key].append(reward)
            self.discount_history[env_key].append(discount**time)
            self.action_history[env_key].append(action)

            if terminal:
                # Add this episode to the training data set
                self._training_data.append({
                    'obs': self.obs_history[env_key],
                    'reward': self.reward_history[env_key],
                    'action': self.action_history[env_key],
                    'discount': self.discount_history[env_key],
                })
                # Reset current episode data
                self.obs_history[env_key] = []
                self.reward_history[env_key] = []
                self.action_history[env_key] = []
                self.discount_history[env_key] = []

            self._train()
            self._update_target()
    def _train(self):
        if len(self._training_data) < self.episodes_per_batch:
            return

        total_loss = torch.tensor(0., device=self.device)
        for episode in self._training_data:
            episode_length = len(episode['obs'])
            try:
                net_output = []
                hidden = self.net.init_hidden()
                for obs in episode['obs']:
                    obs = obs.unsqueeze(0).float().to(self.device)
                    output = self.net(obs,hidden)
                    net_output.append(output)
                    hidden = output['hidden']
                target_net_output = []
                hidden = self.net.init_hidden()
                for obs in episode['obs']:
                    obs = obs.unsqueeze(0).float().to(self.device)
                    output = self.target_net(obs,hidden)
                    target_net_output.append(output)
                    hidden = output['hidden']
                state_values = [
                        o['value'].squeeze()
                        for o in net_output
                ]
                target_state_values = [
                        o['value'].squeeze()
                        for o in target_net_output
                ]
                log_action_probs = [
                        compute_normal_log_prob(o['action_mean'],o['action_std'],a)
                        for o,a in zip(net_output,episode['action'])
                ]
                entropy = [
                    -compute_normal_entropy(o['action_mean'],o['action_std'])
                    for o in net_output
                ]
            except:
                raise

            # Train policy
            loss_pi = compute_advantage_policy_gradient(
                    log_action_probs = log_action_probs,
                    target_state_values = target_state_values,
                    rewards = episode['reward'],
                    terminals = [False]*(episode_length-1)+[True],
                    discounts = episode['discount'],
            ).mean(0)
            # Train value network
            loss_v = compute_mc_state_value_loss(
                    state_values = state_values,
                    last_state_target_value = 0,
                    rewards = episode['reward'],
                    terminals = [False]*(episode_length-1)+[True],
                    discounts = episode['discount'],
            ).mean(0)
            # Entropy
            loss_entropy = torch.stack(entropy).mean(0)
            # Take a gradient step
            total_loss += loss_pi+loss_v+0.01*loss_entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        # Clear data
        self._training_data.clear()
        # Log
        self._training_steps += 1
        self.logger.log(
                debug_loss = total_loss.item()
        )
        self.state_values.append(np.mean(self.state_values_current))
        self.state_values_std.append(np.std(self.state_values_current))
        self.state_values_std_ra.append(np.mean(self.state_values_std) if self._training_steps < 100 else np.mean(self.state_values_std[-100:]))
        self.state_values_current = []
        # Log weights
        param_vals = torch.cat([param.flatten() for param in self.net.parameters()])
        self.logger.log(
                debug_net_param_mean = torch.mean(param_vals).item(),
                debug_net_param_min = torch.min(param_vals).item(),
                debug_net_param_max = torch.max(param_vals).item(),
        )

if __name__ == "__main__":
    import os
    from tqdm import tqdm
    import torch.cuda
    import numpy as np
    import pprint
    import gym
    import gym.envs
    #import pybullet_envs
    import cv2
    from gym.wrappers import FrameStack, AtariPreprocessing
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    from rl.experiments.training.basic import TrainExperiment
    from experiment import make_experiment_runner

    def make_env(env_name, atari=False, one_hot_obs=False):
        env = gym.make(env_name)
        if atari:
            env = AtariPreprocessing(env)
            env = FrameStack(env, 4)
        if one_hot_obs:
            env = rl.debug_tools.frozenlake.OnehotObs(env)
        return env
    
    def train(train_envs, test_env, agent, training_steps=1_000_000, test_frequency=1000, render=False):
        test_results = {}

        done = [True for _ in train_envs]
        for i in tqdm(range(training_steps), desc='training'):
            if i % test_frequency == 0:
                video_file_name = os.path.join('output','video-%d.avi'%i)
                test_results[i] = [test(test_env, agent, render=(i==0 and render), video_file_name=video_file_name) for i in tqdm(range(5), desc='testing')]
                avg = np.mean([x['total_reward'] for x in test_results[i]])
                tqdm.write('Iteration {i}\t Average reward: {avg}'.format(i=i,avg=avg))
                tqdm.write(pprint.pformat(test_results[i], indent=4))

            for env_key,env in enumerate(train_envs):
                if done[env_key]:
                    obs = env.reset()
                    agent.observe(obs, testing=False, env_key=env_key)
                    done[env_key] = False
                else:
                    obs, reward, done[env_key], _ = env.step(agent.act(testing=False, env_key=env_key))
                    agent.observe(obs, reward, done[env_key], testing=False, env_key=env_key)

        for env in train_envs:
            env.close()

        return test_results

    def train_onpolicy(train_envs, agent, training_steps : int = 1_000_000, plot_frequency : int = 10):
        test_results = {}

        rewards = []
        running_averages = []
        episode_rewards = {}
        done = [True for _ in train_envs]
        for i in tqdm(range(training_steps), desc='training'):
            for env_key,env in enumerate(train_envs):
                if done[env_key]: # Start of an episode
                    obs = env.reset()
                    agent.observe(obs, testing=False, env_key=env_key)
                    done[env_key] = False
                    episode_rewards[env_key] = []
                else:
                    obs, reward, done[env_key], _ = env.step(agent.act(testing=False, env_key=env_key))
                    agent.observe(obs, reward, done[env_key], testing=False, env_key=env_key)
                    episode_rewards[env_key].append(reward)
                if done[env_key]: # End of an episode
                    rewards.append(np.sum(episode_rewards[env_key]))
                    ep_len = len(episode_rewards[env_key])
                    running_average = np.mean(rewards) if len(rewards) < 100 else np.mean(rewards[-100:])
                    running_averages.append(running_average)
                    tqdm.write('Step %d\t Episode Length: %d\t Reward: %f\t Running avg reward: %f' % (i,ep_len,rewards[-1],running_average))
                    # Plot
                    if len(running_averages) % plot_frequency == 0:
                        # Reward Running Average
                        plt.plot(range(len(running_averages)),running_averages)
                        plt.grid()
                        plt.xlabel('Episodes')
                        plt.ylabel('Running Average Reward (last 100 episodes)')
                        filename = './plot-reward.png'
                        plt.savefig(filename)
                        tqdm.write('Saved plot to %s' % os.path.abspath(filename))
                        plt.close()
                        # State values
                        plt.plot(range(len(agent.state_values)),agent.state_values)
                        plt.grid()
                        plt.xlabel('Steps')
                        plt.ylabel('State Values')
                        filename = './plot-state-value.png'
                        plt.savefig(filename)
                        tqdm.write('Saved plot to %s' % os.path.abspath(filename))
                        plt.close()
                        # State values std
                        plt.plot(range(len(agent.state_values_std_ra)),agent.state_values_std_ra)
                        plt.grid()
                        plt.xlabel('Steps')
                        plt.ylabel('State Values std running average (100 episodes)')
                        filename = './plot-state-value-std.png'
                        plt.savefig(filename)
                        tqdm.write('Saved plot to %s' % os.path.abspath(filename))
                        plt.close()

        for env in train_envs:
            env.close()

        return test_results

    def train_recurrent(env, agent, training_steps : int = 1_000_000, plot_frequency : int = 10):
        test_results = {}

        rewards = []
        running_averages = []
        episode_rewards = []
        done = True
        for i in tqdm(range(training_steps), desc='training'):
            if done: # Start of an episode
                obs = env.reset()
                agent.observe(obs, testing=False)
                done = False
                episode_rewards = []
            else:
                obs, reward, done, _ = env.step(agent.act(testing=False))
                agent.observe(obs, reward, done, testing=False)
                episode_rewards.append(reward)
            if done: # End of an episode
                rewards.append(np.sum(episode_rewards))
                ep_len = len(episode_rewards)
                running_average = np.mean(rewards) if len(rewards) < 100 else np.mean(rewards[-100:])
                running_averages.append(running_average)
                tqdm.write('Step %d\t Episode Length: %d\t Reward: %f\t Running avg reward: %f' % (i,ep_len,rewards[-1],running_average))
                # Plot
                if len(running_averages) % plot_frequency == 0:
                    # Reward Running Average
                    plt.plot(range(len(running_averages)),running_averages)
                    plt.grid()
                    plt.xlabel('Episodes')
                    plt.ylabel('Running Average Reward (last 100 episodes)')
                    filename = './plot-reward.png'
                    plt.savefig(filename)
                    tqdm.write('Saved plot to %s' % os.path.abspath(filename))
                    plt.close()
                    # State values
                    plt.plot(range(len(agent.state_values)),agent.state_values)
                    plt.grid()
                    plt.xlabel('Steps')
                    plt.ylabel('State Values')
                    filename = './plot-state-value.png'
                    plt.savefig(filename)
                    tqdm.write('Saved plot to %s' % os.path.abspath(filename))
                    plt.close()
                    # State values std
                    plt.plot(range(len(agent.state_values_std_ra)),agent.state_values_std_ra)
                    plt.grid()
                    plt.xlabel('Steps')
                    plt.ylabel('State Values std running average (100 episodes)')
                    filename = './plot-state-value-std.png'
                    plt.savefig(filename)
                    tqdm.write('Saved plot to %s' % os.path.abspath(filename))
                    plt.close()
                    # XXX: DEBUG: Neural net parameters
                    if len(agent.state_values) > 0:
                        plt.plot(*agent.logger['debug_net_param_mean'], label='mean')
                        plt.plot(*agent.logger['debug_net_param_min'], label='min')
                        plt.plot(*agent.logger['debug_net_param_max'], label='max')
                        plt.grid()
                        plt.xlabel('Steps')
                        plt.ylabel('Parameter Weights')
                        plt.title('Neural Net Parameter Weights')
                        filename = './plot-params.png'
                        plt.savefig(filename)
                        tqdm.write('Saved plot to %s' % os.path.abspath(filename))
                        plt.close()

        env.close()

        return test_results

    def test(env, agent, render=False, video_file_name='output.avi'):
        total_reward = 0
        total_steps = 0

        obs = env.reset()
        agent.observe(obs, testing=True)
        if render:
            frame = env.render(mode='rgb_array')
            video = cv2.VideoWriter(video_file_name, 0, 60, (frame.shape[0],frame.shape[1])) # type: ignore
            video.write(frame)
        for total_steps in tqdm(itertools.count(), desc='test episode'):
            obs, reward, done, _ = env.step(agent.act(testing=True))
            if render:
                frame = env.render(mode='rgb_array')
                video.write(frame) # type: ignore
            total_reward += reward
            agent.observe(obs, reward, done, testing=True)
            if done:
                break
        env.close()
        if render:
            video.release() # type: ignore

        return {
            'total_steps': total_steps,
            'total_reward': total_reward
        }

    if torch.cuda.is_available():
        print('GPU found')
        device = torch.device('cuda')
    else:
        print('No GPU found. Running on CPU.')
        device = torch.device('cpu')

    def run_atari_2():
        num_actors = 16
        env_name = 'PongNoFrameskip-v4'
        train_env_keys = list(range(num_actors))

        exp_runner = make_experiment_runner(
                TrainExperiment,
                config={
                    'agent': {
                        'type': A2CAgent,
                        'parameters': {
                            'training_env_keys': train_env_keys,
                        },
                    },
                    'env_test': {'env_name': env_name, 'atari': True},
                    'env_train': {'env_name': env_name, 'atari': True},
                    'train_env_keys': train_env_keys,
                    'test_frequency': None,
                    'save_model_frequency': 250_000,
                    'verbose': True,
                },
                #trial_id='checkpointtest',
                #checkpoint_frequency=250_000,
                checkpoint_frequency=None,
                max_iterations=1_000_000,
                verbose=True,
        )
        exp_runner.exp.logger.init_wandb({
            'project': 'A2C-%s' % env_name
        })
        exp_runner.run()

    def run_atari():
        num_actors = 16
        env_name = 'PongNoFrameskip-v4'
        train_envs = [make_env(env_name,atari=True) for _ in range(num_actors)]
        test_env = make_env(env_name,atari=True)

        action_space = test_env.action_space
        observation_space = test_env.observation_space
        assert observation_space.shape is not None
        agent = A2CAgent(
            action_space=action_space,
            observation_space=observation_space,
            training_env_keys=[i for i in range(num_actors)],
            net  = PolicyValueNetworkCNN(action_space.n).to(device),
            device=device,
        )

        results = train_onpolicy(train_envs,agent,plot_frequency=10)
        test(test_env, agent)
        breakpoint()
        return results

    def run_discrete():
        num_actors = 16
        train_envs = [make_env('FrozenLake-v0',one_hot_obs=True) for _ in range(num_actors)]
        test_env = make_env('FrozenLake-v0',one_hot_obs=True)

        action_space = test_env.action_space
        observation_space = test_env.observation_space
        assert observation_space.shape is not None
        agent = A2CAgent(
            action_space=action_space,
            observation_space=observation_space,
            training_env_keys=[i for i in range(num_actors)],
            net  = PolicyValueNetworkLinear(observation_space.shape[0], action_space.n).to(device),
            discount_factor=1,
            learning_rate=0.01,
            obs_scale=1,
            target_update_frequency=1,
            device=device,
        )

        results = train_onpolicy(train_envs,agent,plot_frequency=10)
        test(test_env, agent)
        return results

    def run_mujoco_recurrent():
        env_name = 'Hopper-v3'
        env = make_env(env_name)

        action_space = env.action_space
        observation_space = env.observation_space
        assert observation_space.shape is not None
        agent = A2CAgentRecurrent(
            action_space=action_space,
            observation_space=observation_space,
            #learning_rate = 3e-4,
            net = PolicyValueNetworkRecurrentFCNN(observation_space.shape[0], action_space.shape[0]).to(device),
            device=device,
        )

        results = train_recurrent(env,agent,plot_frequency=10, training_steps=50_000_000)
        test(env, agent)
        return results

    def run_mujoco():
        num_actors = 16
        env_name = 'Hopper-v3'
        train_envs = [make_env(env_name) for _ in range(num_actors)]
        test_env = make_env(env_name)

        action_space = test_env.action_space
        observation_space = test_env.observation_space
        assert observation_space.shape is not None
        agent = A2CAgent(
            action_space=action_space,
            observation_space=observation_space,
            training_env_keys=[i for i in range(num_actors)],
            #learning_rate = 3e-4,
            max_rollout_length=128,
            reward_scale=5,
            obs_scale=1,
            net = PolicyValueNetworkFCNN(observation_space.shape[0], action_space.shape[0], shared_std=False).to(device),
            device=device,
        )

        results = train_onpolicy(train_envs, agent, plot_frequency=10, training_steps=50_000_000)
        test(test_env, agent)
        return results

    #run_mujoco()
    #run_atari()
    run_atari_2()
