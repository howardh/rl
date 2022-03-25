import copy
from collections import defaultdict
from typing import Optional, Tuple, List, Union, Mapping
from typing_extensions import TypedDict

import numpy as np
import torch
import torch.nn
import torch.utils.data
import torch.distributions
import torch.optim
import gym.spaces
from torchtyping import TensorType
from torch.utils.data.dataloader import default_collate

from frankenstein.loss.policy_gradient import advantage_policy_gradient_loss, clipped_advantage_policy_gradient_loss, advantage_policy_gradient_loss_batch, clipped_advantage_policy_gradient_loss_batch
from frankenstein.value.monte_carlo import monte_carlo_return_iterative, monte_carlo_return_iterative_batch
from frankenstein.buffer.history import HistoryBuffer
from frankenstein.buffer.vec_history import VecHistoryBuffer
from experiment.logger import Logger, SubLogger
from rl.utils import default_state_dict, default_load_state_dict
from rl.agent.agent import DeployableAgent
from rl.experiments.training._utils import zip2

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
class PolicyValueNetworkRecurrentCNN(PolicyValueNetworkRecurrent):
    """ A model which acts both as a policy network and a state-value estimator. """
    def __init__(self, num_actions, hidden_size, in_channels=1):
        super().__init__()
        self.num_actions = num_actions
        self._hidden_size = hidden_size
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,out_channels=32,kernel_size=8,stride=4),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=32,out_channels=64,kernel_size=4,stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=64,out_channels=64,kernel_size=3,stride=1),
            torch.nn.LeakyReLU(),
        )
        self.lstm = torch.nn.LSTMCell(input_size=64*7*7,hidden_size=hidden_size)
        self.v = torch.nn.Linear(in_features=512,out_features=1)
        self.pi = torch.nn.Linear(in_features=512,out_features=num_actions)
    def forward(self, x, hidden) -> PolicyValueNetworkOutput:
        x = self.conv(x)
        x = x.view(-1,64*7*7)
        h,c = self.lstm(x,hidden)
        x = h
        v = self.v(x)
        pi = self.pi(x)
        return {
                'value': v,
                'action': pi, # Unnormalized action probabilities
                'hidden': (h,c),
        }
    def init_hidden(self, batch_size=1):
        device = next(self.parameters()).device
        return (
                torch.zeros([batch_size,self._hidden_size], device=device),
                torch.zeros([batch_size,self._hidden_size], device=device),
        )
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
            pi_std = torch.log(1+self.pi_std(x).clip(max=80).exp()) # exp will diverge for input greater than 88.7 (32-bit) or 709.7 (64-bit)
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

def compute_mc_state_value_loss(
        state_values : Union[torch.Tensor,List[torch.Tensor]],
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

def compute_ppo_policy_gradient(
        log_action_probs : torch.Tensor,
        log_action_probs_old : torch.Tensor,
        target_state_values : torch.Tensor,
        rewards : List[Optional[float]],
        terminals : List[bool],
        discounts : List[float],
        epsilon : float,
    ) -> torch.Tensor:
    """
    PPO clipped policy gradient ($L^{CLIP}$) for discrete action spaces.
    Given a sequence of length n, we observe states/actions/rewards r_0,s_0,a_0,r_1,s_1,a_1,r_2,s_2,...,r_{n-1},s_{n-1},a_{n-1}.

    Args:
        log_action_probs: A 1D tensor of length n (last element is ignored). The element at index i is the log probability of selection action a_i at state s_i.
        log_action_probs_old: A 1D tensor of length n (last element is ignored). The element at index i is the log probability of selection action a_i at state s_i according to the old policy.
        target_state_values: A 1D tensor of length n or a 1D tensor. The element at index i is the state value of state s_i as predicted by the target Q network.
        rewards: A list of length n (first element is ignored). The element at index i is r_{i+1}. The value can be None if state s_{i} was terminal and s_{i+1} is the initial state of a new episode.
        terminals: A list of length n. The element at index i is True if s_i is a terminal state, and False otherwise.
        discounts: ???
        epsilon: The importance sampling ratio is clipped between 1-epsilon and 1+epsilon.
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
    ratio = torch.exp(log_action_probs-log_action_probs_old)
    losses = []
    for j in range(len(target_state_values)-1):
        if terminals[j]:
            losses.append(torch.tensor(0,device=device))
            continue
        # Value of the state predicted by the model
        v_pred = target_state_values[j]
        v_target = vt[j]
        # Loss
        advantage = (v_target-v_pred).detach()
        loss_unclipped = ratio[j]*advantage
        loss_clipped = torch.clip(ratio[j],1-epsilon,1+epsilon)*advantage
        loss = -torch.min(loss_unclipped, loss_clipped).squeeze()
        losses.append(loss)
    return torch.stack(losses)

def compute_entropy(
        log_probs : torch.Tensor
    ) -> torch.Tensor:
    return -(log_probs.exp()*log_probs).sum(1)

def compute_normal_log_prob(mean,std,sample):
    dist = torch.distributions.normal.Normal(mean,std)
    return dist.log_prob(torch.tensor(sample)).sum(1)
def compute_normal_log_prob_batch(mean:torch.Tensor,std:torch.Tensor,sample:torch.Tensor):
    pi = np.pi
    #probs = torch.sqrt(2*pi*std.prod(dim=1))*torch.exp(-0.5*(sample-mean)**2/std)
    log_probs = 0.5*(torch.log(2*pi*std.prod(dim=1,keepdim=True))-((sample-mean)/std)**2).sum(1)
    return log_probs
def compute_normal_entropy(mean,std):
    dist = torch.distributions.normal.Normal(mean,std)
    return dist.entropy().sum(1)
#def compute_normal_entropy_batch(mean,std):
#    entropy = 0.5*np.log(2*np.pi)+0.5+torch.log(std)
#    return entropy

def sample_minibatch(shape, minibatch_size):
    indices = np.random.choice(np.prod(shape), minibatch_size, replace=False)
    return torch.tensor(np.unravel_index(indices, shape))

def zip_dict(*dcts):
    if not dcts:
        return
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i, tuple(d[i] for d in dcts))

##################################################
# Non-Vectorized Agents

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
        self.train_history_buffer = HistoryBuffer(
                num_envs=len(training_env_keys),
                max_len=max_rollout_length,
                default_action=self.action_space.sample()*0,
                device=device)
        self.test_history_buffer = defaultdict(
            lambda: HistoryBuffer(
                num_envs=1,
                max_len=1,
                default_action=self.action_space.sample()*0,
                device=device)
        )
        self._steps = 0 # Number of steps experienced
        self._training_steps = 0 # Number of training steps experienced

        if net is None:
            self.net = self._init_default_net(observation_space,action_space,device)
        else:
            self.net = net
            self.net.to(device)
        self.target_net = copy.deepcopy(self.net)
        #self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate)

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

        if testing:
            history = self.test_history_buffer[env_key][0]
        else:
            env_index = self.training_env_keys.index(env_key)
            history = self.train_history_buffer[env_index]

        if reward is not None:
            reward *= self.reward_scale

        history.append_obs(obs, reward, terminal, misc={'discount': discount**time})

        # Choose next action
        net_output = self.net(
                torch.tensor(obs,device=self.device).unsqueeze(0).float()*self.obs_scale
        )
        if isinstance(self.action_space, gym.spaces.Discrete):
            assert 'action' in net_output
            action_probs_unnormalized = net_output['action']
            action_probs = action_probs_unnormalized.softmax(1).squeeze()
            assert torch.abs(action_probs.sum()-1) < 1e-6
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()
        elif isinstance(self.action_space, gym.spaces.Box):
            assert 'action_mean' in net_output
            assert 'action_std' in net_output
            action_dist = torch.distributions.Normal(net_output['action_mean'],net_output['action_std'])
            action = action_dist.sample().cpu().numpy()
        else:
            raise NotImplementedError('Unsupported action space of type %s' % type(self.action_space))
        self.logger.log(train_entropy=action_dist.entropy().mean().item())

        assert 'value' in net_output
        state_value = net_output['value']
        self.state_values_current.append(state_value.item())

        if not terminal:
            history.append_action(action)

        # Save training data and train
        if not testing:
            self._steps += 1
            if not isinstance(self.logger, SubLogger):
                self.logger.log(step=self._steps)

            self.logger.append(debug_state_value=state_value.item())

            self._train()
            self._update_target()
    def _train(self):
        history = self.train_history_buffer
        t_max = self.max_rollout_length

        # Check if we've accumulated enough data
        num_points = [len(history[i].obs_history) for i,_ in enumerate(self.training_env_keys)]
        if min(num_points) >= t_max:
            if max(num_points) != min(num_points):
                raise Exception('This should not happen')
            n = num_points[0]
            try:
                obs = {
                    ek: history[i].obs.float()*self.obs_scale
                    for i,ek in enumerate(self.training_env_keys)
                }
                action_history = {ek: history[i].action for i,ek in enumerate(self.training_env_keys)}
                reward_history = {ek: history[i].reward for i,ek in enumerate(self.training_env_keys)}
                terminal_history = {ek: history[i].terminal for i,ek in enumerate(self.training_env_keys)}
                misc = history.misc
                assert isinstance(misc, dict)
                assert 'discount' in misc
                discount_history = {ek: misc['discount'][:,i] for i,ek in enumerate(self.training_env_keys)}
                indices = { 
                    ek: range(len(history[i].action_history))
                    for i,ek in enumerate(self.training_env_keys)
                }
                net_output = {
                    ek: self.net(obs[ek])
                    for ek in self.training_env_keys
                }
                state_values = {
                    ek: outputs['value'].squeeze()
                    for ek,outputs in net_output.items()
                }
                target_state_values = {
                    ek: self.target_net(obs[ek])['value'].squeeze()
                    for ek in self.training_env_keys
                }
                if isinstance(self.action_space,gym.spaces.Discrete):
                    log_action_probs = {ek: 
                        outputs['action'].log_softmax(1).squeeze()[indices[ek],action_history[ek]]
                        for ek,outputs in net_output.items()
                    }
                    entropy = {
                        ek: compute_entropy(outputs['action'].log_softmax(1).squeeze())
                        for ek,outputs in net_output.items()
                    }
                else:
                    log_action_probs = {
                        ek: compute_normal_log_prob_batch(
                                outputs['action_mean'],
                                outputs['action_std'],
                                torch.tensor(action_history[ek],device=self.device))
                        for ek,outputs in net_output.items()}
                    entropy = {ek: 
                        compute_normal_entropy(out['action_mean'],out['action_std'])
                        for ek,out in net_output.items()
                    }
            except:
                raise
            # Train policy
            loss_pi = [advantage_policy_gradient_loss(
                        log_action_probs = log_action_probs[ek][:n-1],
                        state_values = state_values[ek][:n-1].detach(),
                        next_state_values = target_state_values[ek][1:],
                        rewards = reward_history[ek][1:],
                        terminals = terminal_history[ek][1:],
                        prev_terminals = terminal_history[ek][:n-1],
                        discounts = discount_history[ek][1:],
                ) for ek in self.training_env_keys]
            loss_pi = torch.stack(loss_pi).mean()
            # Train value network
            state_value_estimate = {ek: monte_carlo_return_iterative(
                        state_values = target_state_values[ek][1:],
                        rewards = reward_history[ek][1:],
                        terminals = terminal_history[ek][1:],
                        discounts = discount_history[ek][1:],
                ) for ek in self.training_env_keys}
            loss_v = [
                    (state_values[ek][:-1]-state_value_estimate[ek])**2
                    for ek in self.training_env_keys
            ]
            loss_v = torch.stack(loss_v).mean()
            # Entropy
            loss_entropy = -torch.stack([e for e in entropy.values()]).mean()
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
            self.train_history_buffer.clear()
            # Log
            self._training_steps += 1
            self.logger.log(
                    loss_pi=loss_pi.item(),
                    loss_v=loss_v.item(),
                    loss_entropy=loss_entropy.item(),
                    loss_total=loss.item(),
                    train_state_value=np.array([v.mean().item() for v in state_values.values()]).mean(),
                    train_state_value_target_net=np.array([v.mean().item() for v in target_state_values.values()]).mean(),
            )
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

        if testing:
            history = self.test_history_buffer[env_key][0]
        else:
            env_index = self.training_env_keys.index(env_key)
            history = self.train_history_buffer[env_index]

        action = history.action_history[-1]
        if action is None:
            raise Exception('No action found. Agent must observe the environment before taking an action.')

        if isinstance(self.action_space, gym.spaces.Discrete):
            return action
        elif isinstance(self.action_space, gym.spaces.Box):
            high = np.array(self.action_space.high)
            low = np.array(self.action_space.low)
            d = high-low
            assert isinstance(d,np.ndarray) # XXX: Workaround for https://github.com/microsoft/pylance-release/issues/1619. Remove when this gets fixed.
            scale = d/2
            bias = (high+low)/2
            scaled_action = np.tanh(action)*scale+bias
            return scaled_action
        else:
            raise NotImplementedError('Unsupported action space of type %s' % type(self.action_space))

    def state_dict(self):
        return default_state_dict(self, [
            'train_history_buffer',
            'test_history_buffer',
            '_steps',
            '_training_steps',
            'net',
            'target_net',
            'optimizer',

            'logger',
            'state_values_current',
            'state_values',
            'state_values_std',
            'state_values_std_ra',
        ])
    def load_state_dict(self, state):
        default_load_state_dict(self,state)

    def state_dict_deploy(self):
        return default_state_dict(self, [
            'action_space',
            'observation_space',
            'net',
        ])
    def load_state_dict_deploy(self, state):
        default_load_state_dict(self,state)


class PPOAgent(A2CAgent):
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
            # PPO-specific stuff
            num_minibatches : int = 5,
        ):
        super().__init__(action_space, observation_space, discount_factor=discount_factor, learning_rate=learning_rate, target_update_frequency=target_update_frequency, polyak_rate=polyak_rate, max_rollout_length=max_rollout_length, training_env_keys=training_env_keys, obs_scale=obs_scale, reward_scale=reward_scale, device=device, net=net, logger=logger)
        self.num_minibatches = num_minibatches
        raise NotImplementedError('This implementation doesn\'t work yet.')
    def _train(self):
        history = self.train_history_buffer
        t_max = self.max_rollout_length
        log_action_probs_old = None

        # Check if we've accumulated enough data
        num_points = [len(history[i].obs_history) for i,_ in enumerate(self.training_env_keys)]
        if min(num_points) < t_max:
            return
        if max(num_points) != min(num_points):
            raise Exception('This should not happen')
        n = num_points[0]
        for _ in range(self.num_minibatches):
            try:
                obs = {
                    ek: history[i].obs.float()*self.obs_scale
                    for i,ek in enumerate(self.training_env_keys)
                }
                action_history = {ek: history[i].action for i,ek in enumerate(self.training_env_keys)}
                reward_history = {ek: history[i].reward for i,ek in enumerate(self.training_env_keys)}
                terminal_history = {ek: history[i].terminal for i,ek in enumerate(self.training_env_keys)}
                misc = history.misc
                assert isinstance(misc, dict)
                assert 'discount' in misc
                discount_history = {ek: misc['discount'][:,i] for i,ek in enumerate(self.training_env_keys)}
                net_output = {
                    ek: self.net(o)
                    for ek,o in obs.items()
                }
                state_values = {
                    ek: out['value'].squeeze()
                    for ek,out in net_output.items()
                }
                target_state_values = {ek: 
                    self.target_net(o)['value'].squeeze().detach()
                    for ek,o in obs.items()
                }
                if isinstance(self.action_space,gym.spaces.Discrete):
                    log_action_probs = {ek: 
                        out['action'].log_softmax(1).squeeze()[range(len(a)),a]
                        for ek,(out,a) in zip_dict(net_output,action_history)
                    }
                    entropy = {
                        ek: compute_entropy(outputs['action'].log_softmax(1))
                        for ek,outputs in net_output.items()
                    }
                else:
                    log_action_probs = {
                        ek: compute_normal_log_prob_batch(
                            out['action_mean'],
                            out['action_std'],
                            torch.cat([torch.tensor(x, device=self.device) for x in a])
                        )
                        for ek,(out,a) in zip_dict(net_output,action_history)
                    }
                    entropy = {ek: 
                        compute_normal_entropy(out['action_mean'],out['action_std'])
                        for ek,out in net_output.items()
                    }
                if log_action_probs_old is None:
                    log_action_probs_old = {
                            k: v.clone().detach()
                            for k,v in log_action_probs.items()
                    }
            except:
                raise
            # Train policy
            loss_pi = [clipped_advantage_policy_gradient_loss(
                        log_action_probs = log_action_probs[ek][:n-1],
                        old_log_action_probs = log_action_probs_old[ek][:n-1],
                        state_values = state_values[ek][:n-1].detach(), # Does this need to use `state_values`? That's what I do in the A2C code
                        next_state_values = target_state_values[ek][1:n],
                        rewards = reward_history[ek][1:n],
                        terminals = terminal_history[ek][1:n],
                        prev_terminals = terminal_history[ek][:n-1],
                        discounts = discount_history[ek][1:n],
                        epsilon = 0.1,
                ) for ek in self.training_env_keys]
            loss_pi = torch.stack(loss_pi).mean()
            # Train value network
            state_value_estimate = {ek: monte_carlo_return_iterative(
                        state_values = target_state_values[ek][1:n],
                        rewards = reward_history[ek][1:n],
                        terminals = terminal_history[ek][1:n],
                        discounts = discount_history[ek][1:n],
                ) for ek in self.training_env_keys}
            loss_v = [
                    (state_values[ek][:-1]-state_value_estimate[ek])**2
                    for ek in self.training_env_keys
            ]
            loss_v = torch.stack(loss_v).mean()
            #loss_v = [compute_mc_state_value_loss(
            #            state_values = state_values[ek],
            #            last_state_target_value = float(target_state_values[ek][-1].item()),
            #            rewards = reward_history[ek],
            #            terminals = terminal_history[ek],
            #            discounts = discount_history[ek],
            #    ) for ek in self.training_env_keys]
            #loss_v = torch.stack(loss_v).mean()
            # Entropy
            loss_entropy = -torch.stack([ e.mean() for e in entropy.values() ]).mean()
            # Take a gradient step
            loss = loss_pi+loss_v+0.01*loss_entropy
            self.optimizer.zero_grad()
            loss.backward()
            if __debug__: # Check weights
                psum = 0
                for p in self.net.parameters():
                    if p.grad is None:
                        continue
                    psum += p.grad.mean()
                if psum != psum:
                    breakpoint()
            self.optimizer.step()
            # Log
            self.logger.append(
                    loss_pi=loss_pi.item(),
                    loss_v=loss_v.item(),
                    loss_entropy=loss_entropy.item(),
                    loss_total=loss.item(),
                    train_state_value=np.array([v.mean().item() for v in state_values.values()]).mean(),
                    train_state_value_target_net=np.array([v.mean().item() for v in target_state_values.values()]).mean(),
            )
        # Clear data
        history.clear()
        # Log
        self._training_steps += 1


class A2CAgentRecurrent(A2CAgent):
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
            hidden_reset_min_prob : float = 0,
            hidden_reset_max_prob : float = 0,
            device : torch.device = torch.device('cpu'),
            net : PolicyValueNetworkRecurrent = None,
            logger : Logger = None,
        ):
        super().__init__(action_space, observation_space, discount_factor=discount_factor, learning_rate=learning_rate, target_update_frequency=target_update_frequency, polyak_rate=polyak_rate, max_rollout_length=max_rollout_length, training_env_keys=training_env_keys, obs_scale=obs_scale, reward_scale=reward_scale, device=device, net=net, logger=logger)
        self.hidden_reset_min_prob = hidden_reset_min_prob
        self.hidden_reset_max_prob = hidden_reset_max_prob
        self._prev_hidden = {}
    def _init_default_net(self, observation_space, action_space,device) -> PolicyValueNetworkRecurrent:
        if isinstance(observation_space, gym.spaces.Box):
            assert observation_space.shape is not None
            if len(observation_space.shape) == 1: # Mujoco
                return PolicyValueNetworkRecurrentFCNN(
                        num_features=observation_space.shape[0],
                        num_actions=action_space.shape[0],
                        shared_std=False
                ).to(device)
            if len(observation_space.shape) == 3: # Atari
                return PolicyValueNetworkRecurrentCNN(
                        num_actions=action_space.n,
                        hidden_size=512,
                        in_channels=observation_space.shape[0],
                ).to(device)
        raise Exception('Unsupported observation space or action space.')
    def observe(self, obs, reward=None, terminal=False, testing=False, time=1, discount=None, env_key=None):
        assert isinstance(self.net, PolicyValueNetworkRecurrent)
        if env_key is None:
            env_key = testing
        if discount is None:
            discount = self.discount_factor

        if testing:
            history = self.test_history_buffer[env_key][0]
        else:
            env_index = self.training_env_keys.index(env_key)
            history = self.train_history_buffer[env_index]

        if reward is not None:
            reward *= self.reward_scale

        hidden_reset_prob = self._compute_annealed_epsilon(
                max_eps=self.hidden_reset_max_prob, min_eps=self.hidden_reset_min_prob, max_steps=1_000_000)
        hidden_reset = not testing and np.random.rand() < hidden_reset_prob
        if reward is None or hidden_reset:
            self._prev_hidden[env_key] = self.net.init_hidden(batch_size=1)

        history.append_obs(obs, reward, terminal,
                misc={
                    'discount': discount**time,
                    'hidden': tuple([x.squeeze() for x in self._prev_hidden[env_key]]),
                    'hidden_reset': hidden_reset
                })

        # Choose next action
        net_output = self.net(
                torch.tensor(obs,device=self.device).unsqueeze(0).float()*self.obs_scale,
                hidden=self._prev_hidden[env_key]
        )
        assert 'hidden' in net_output
        self._prev_hidden[env_key] = tuple([x.detach() for x in net_output['hidden']])
        if isinstance(self.action_space, gym.spaces.Discrete):
            assert 'action' in net_output
            action_probs_unnormalized = net_output['action']
            action_probs = action_probs_unnormalized.softmax(1).squeeze()
            assert torch.abs(action_probs.sum()-1) < 1e-6
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()
        elif isinstance(self.action_space, gym.spaces.Box):
            assert 'action_mean' in net_output
            assert 'action_std' in net_output
            action_dist = torch.distributions.Normal(net_output['action_mean'],net_output['action_std'])
            action = action_dist.sample().cpu().numpy()
        else:
            raise NotImplementedError('Unsupported action space of type %s' % type(self.action_space))
        self.logger.log(train_entropy=action_dist.entropy().mean().item())

        assert 'value' in net_output
        state_value = net_output['value']
        self.state_values_current.append(state_value.item())

        if not terminal:
            history.append_action(action)

        # Save training data and train
        if not testing:
            self._steps += 1
            if not isinstance(self.logger, SubLogger):
                self.logger.log(step=self._steps)

            self.logger.log(
                    debug_state_value=state_value.item(),
                    hidden_reset_prob=hidden_reset_prob,
            )

            self._train()
            self._update_target()
    def _train(self):
        history = self.train_history_buffer
        t_max = self.max_rollout_length

        # Check if we've accumulated enough data
        num_points = [len(history[i].obs_history) for i,_ in enumerate(self.training_env_keys)]
        if min(num_points) >= t_max:
            if max(num_points) != min(num_points):
                raise Exception('This should not happen')
            n = num_points[0]
            try:
                misc = history.misc
                assert isinstance(misc,dict)
                hidden = misc['hidden']
                obs = history.obs.float()*self.obs_scale
                action_history = history.action
                reward_history = history.reward
                terminal_history = history.terminal
                discount_history = misc['discount']
                hidden_reset = misc['hidden_reset']

                net_output = []
                h = (hidden[0][0,:],hidden[1][0,:])
                for o,hr in zip(obs,hidden_reset):
                    h = ( # FIXME: Should not be hard-coded. We don't know what the hidden state format is.
                            h[0]*hr.logical_not().unsqueeze(1),
                            h[1]*hr.logical_not().unsqueeze(1),
                    )
                    no = self.net(o,h)
                    h = no['hidden']
                    net_output.append(no)

                target_net_output = []
                h = (hidden[0][0,:],hidden[1][0,:])
                for o,hr in zip(obs,hidden_reset):
                    h = ( # FIXME: Should not be hard-coded. We don't know what the hidden state format is.
                            h[0]*hr.logical_not().unsqueeze(1),
                            h[1]*hr.logical_not().unsqueeze(1),
                    )
                    no = self.target_net(o,h)
                    h = no['hidden']
                    target_net_output.append(no)

                state_values = torch.stack([o['value'].squeeze() for o in net_output])
                target_state_values = torch.stack([o['value'].squeeze() for o in target_net_output])
                if isinstance(self.action_space,gym.spaces.Discrete):
                    log_action_probs = torch.cat([
                        o['action'].log_softmax(1).gather(1,a.unsqueeze(0))
                        for o,a in zip(net_output,action_history)
                    ])
                    #batch_size = obs.shape[1]
                    #log_action_probs = torch.cat([
                    #    torch.stack([torch.arange(18) for _ in range(batch_size)]).gather(1,a.unsqueeze(0))
                    #    for o,a in zip(net_output,history.action)
                    #])
                    #for s,b in itertools.product(range(n),range(batch_size)):
                    #    assert log_action_probs[s,b] == net_output[s]['action'][b,history.action[s,b]]
                    #log_action_probs = {ek: 
                    #    outputs['action'].log_softmax(1).squeeze()[indices[ek],action_history[ek]]
                    #    for ek,outputs in net_output.items()
                    #}
                    entropy = torch.stack([
                        compute_entropy(o['action'].log_softmax(1).squeeze())
                        for o in net_output
                    ])
                else:
                    raise NotImplementedError()
                    #log_action_probs = {
                    #    ek: compute_normal_log_prob_batch(
                    #            outputs['action_mean'],
                    #            outputs['action_std'],
                    #            torch.tensor(action_history[ek],device=self.device))
                    #    for ek,outputs in net_output.items()}
                    #entropy = {ek: 
                    #    compute_normal_entropy(out['action_mean'],out['action_std'])
                    #    for ek,out in net_output.items()
                    #}
            except:
                raise
            # Train policy
            loss_pi = advantage_policy_gradient_loss_batch(
                    log_action_probs = log_action_probs[:n-1,:],
                    state_values = state_values[:n-1,:].detach(),
                    next_state_values = target_state_values[1:,:],
                    rewards = reward_history[1:,:],
                    terminals = terminal_history[1:,:],
                    prev_terminals = terminal_history[:n-1,:],
                    discounts = discount_history[1:,:],
            )
            loss_pi = loss_pi.mean()
            # Train value network
            state_value_estimate = monte_carlo_return_iterative_batch(
                    state_values = target_state_values[1:,:],
                    rewards = reward_history[1:,:],
                    terminals = terminal_history[1:,:],
                    discounts = discount_history[1:,:],
            )
            loss_v = (state_values[:-1,:]-state_value_estimate)**2
            loss_v = loss_v.mean()
            # Entropy
            loss_entropy = -entropy.mean()
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
            self.train_history_buffer.clear()
            # Log
            self._training_steps += 1
            self.logger.log(
                    loss_pi=loss_pi.item(),
                    loss_v=loss_v.item(),
                    loss_entropy=loss_entropy.item(),
                    loss_total=loss.item(),
                    train_state_value=state_values.mean().item(),
                    train_state_value_target_net=target_state_values.mean().item(),
            )

    def _compute_annealed_epsilon(self, max_eps, min_eps, max_steps=1_000_000):
        if max_steps == 0:
            return min_eps
        #return (1-eps)*max(1-self._steps/max_steps,0)+eps # Linear
        return (max_eps-min_eps)*np.exp(-self._steps/max_steps)+min_eps # Exponential

    def state_dict(self):
        return {
                **super().state_dict(),
                '_prev_hidden': self._prev_hidden,
                'hidden_reset_min_prob': self.hidden_reset_min_prob,
                'hidden_reset_max_prob': self.hidden_reset_max_prob,
        }
    def load_state_dict(self, state):
        self._prev_hidden = state.pop('_prev_hidden')
        self.hidden_reset_min_prob = state.pop('hidden_reset_min_prob')
        self.hidden_reset_max_prob = state.pop('hidden_reset_max_prob')
        super().load_state_dict(state)


##################################################
# Vectorized Agents

class A2CAgentVec(DeployableAgent):
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
            num_train_envs : int = 16,
            num_test_envs : int = 4,
            obs_scale : Union[dict,float] = 1/255,
            reward_scale : float = 1,
            vf_loss_coeff : float = 1.,
            entropy_loss_coeff : float = 0.01,
            device : torch.device = torch.device('cpu'),
            optimizer : str = 'rmsprop', # adam or rmsprop
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
        self.vf_loss_coeff = vf_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.device = device
        self.num_training_envs = num_train_envs

        # Validate input
        if max_rollout_length < 2:
            raise ValueError(f'Rollout length must be >= 2. Received {max_rollout_length}. Note that a rollout length of n consists of n-1 transitions.')
        if target_update_frequency % num_train_envs != 0:
            raise ValueError(f'Target update frequency must be a multiple of the number of training environments. Received {target_update_frequency} and {num_train_envs}.')
        if isinstance(observation_space,gym.spaces.Dict) and not isinstance(obs_scale,dict):
            raise ValueError(f'If observation space is a dictionary, obs_scale must be a dictionary. Received {obs_scale}.')

        # State (training)
        self.train_history_buffer = VecHistoryBuffer(
                num_envs=num_train_envs,
                max_len=max_rollout_length,
                device=device)
        self.test_history_buffer = VecHistoryBuffer(
                num_envs=num_test_envs,
                max_len=1,
                device=device)
        self._steps = 0 # Number of steps experienced
        self._training_steps = 0 # Number of training steps experienced

        if net is None:
            self.net = self._init_default_net(observation_space,action_space,device)
        else:
            self.net = net
            self.net.to(device)
        self.target_net = copy.deepcopy(self.net)
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate)
        else:
            raise Exception(f'Unsupported optimizer: "{optimizer}". Use "adam" or "rmsprop".')

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

    def observe(self, obs, reward=None, terminal=None, testing=False, time=1, discount=None):
        if discount is None:
            discount = self.discount_factor

        if testing:
            history = self.test_history_buffer
        else:
            history = self.train_history_buffer

        if reward is not None:
            reward *= self.reward_scale

        history.append_obs(obs, reward, terminal, misc=[{'discount': discount**time}]*len(obs))

        # Choose next action
        net_output = self.net(
                torch.tensor(obs,device=self.device).float()*self.obs_scale
        )
        if isinstance(self.action_space, gym.spaces.Discrete):
            assert 'action' in net_output
            action_probs_unnormalized = net_output['action']
            action_probs = action_probs_unnormalized.softmax(1).squeeze()
            assert (torch.abs(action_probs.sum(1)-1) < 1e-6).all()
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().cpu().numpy()
        elif isinstance(self.action_space, gym.spaces.Box):
            assert 'action_mean' in net_output
            assert 'action_std' in net_output
            action_dist = torch.distributions.Normal(net_output['action_mean'],net_output['action_std'])
            action = action_dist.sample().cpu().numpy()
        else:
            raise NotImplementedError('Unsupported action space of type %s' % type(self.action_space))
        self.logger.log(train_entropy=action_dist.entropy().mean().item())

        assert 'value' in net_output
        state_value = net_output['value']
        self.state_values_current.append(state_value.mean().item())

        history.append_action(action)

        # Save training data and train
        if not testing:
            self._steps += self.num_training_envs
            if not isinstance(self.logger, SubLogger):
                self.logger.log(step=self._steps)

            self.logger.append(debug_state_value=state_value.mean().item())

            self._train()
            self._update_target()
    def _train(self):
        history = self.train_history_buffer
        t_max = self.max_rollout_length

        # Check if we've accumulated enough data
        num_points = len(history.obs_history)
        if num_points >= t_max:
            n = num_points
            try:
                obs = history.obs.float()*self.obs_scale
                action = history.action
                reward = history.reward
                terminal = history.terminal
                misc = history.misc
                assert isinstance(misc, dict)
                assert 'discount' in misc
                discount = misc['discount']
                net_output = default_collate([self.net(o) for o in obs])
                target_net_output = default_collate([self.target_net(o) for o in obs])
                state_values = net_output['value'].squeeze(2)
                target_state_values = target_net_output['value'].squeeze(2)
                if isinstance(self.action_space,gym.spaces.Discrete):
                    log_action_probs = net_output['action'].log_softmax(2).gather(2,action.unsqueeze(2)).squeeze(2)
                    entropy = -(net_output['action'].log_softmax(2)*net_output['action'].softmax(2)).sum(2).mean()
                else:
                    raise NotImplementedError()
            except:
                raise
            # Train policy
            loss_pi = advantage_policy_gradient_loss_batch(
                    log_action_probs = log_action_probs[:n-1,:],
                    state_values = state_values[:n-1,:].detach(),
                    next_state_values = target_state_values[1:,:],
                    rewards = reward[1:,:],
                    terminals = terminal[1:,:],
                    prev_terminals = terminal[:n-1,:],
                    discounts = discount[1:,:],
            )
            loss_pi = loss_pi.mean()
            # Train value network
            state_value_estimate = monte_carlo_return_iterative_batch(
                    state_values = target_state_values[1:,:].detach(),
                    rewards = reward[1:,:],
                    terminals = terminal[1:,:],
                    discounts = discount[1:,:],
            )
            loss_v = (state_values[:-1,:]-state_value_estimate)**2
            loss_v = loss_v.mean()
            # Entropy
            loss_entropy = -entropy.mean()
            # Take a gradient step
            loss = loss_pi+self.vf_loss_coeff*loss_v+self.entropy_loss_coeff*loss_entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Clear data
            self.train_history_buffer.clear()
            # Log
            self._training_steps += 1
            self.logger.log(
                    loss_pi=loss_pi.item(),
                    loss_v=loss_v.item(),
                    loss_entropy=loss_entropy.item(),
                    loss_total=loss.item(),
                    train_state_value=state_values.mean().item(),
                    train_state_value_target_net=target_state_values.mean().item(),
            )
    def _update_target(self):
        if self._steps % self.target_update_frequency != 0:
            return
        tau = self.polyak_rate
        for p1,p2 in zip(self.target_net.parameters(), self.net.parameters()):
            p1.data = (1-tau)*p1+tau*p2

    def act(self, testing=False):
        """ Return a random action according to an epsilon-greedy policy. """
        if testing:
            history = self.test_history_buffer
        else:
            history = self.train_history_buffer

        action = history.action_history[-1]

        if isinstance(self.action_space, gym.spaces.Discrete):
            return action
        elif isinstance(self.action_space, gym.spaces.Box):
            raise NotImplementedError()
        else:
            raise NotImplementedError('Unsupported action space of type %s' % type(self.action_space))

    def state_dict(self):
        return default_state_dict(self, [
            'train_history_buffer',
            'test_history_buffer',
            '_steps',
            '_training_steps',
            'net',
            'target_net',
            'optimizer',

            'logger',
            'state_values_current',
            'state_values',
            'state_values_std',
            'state_values_std_ra',
        ])
    def load_state_dict(self, state):
        default_load_state_dict(self,state)
        # The environment will be reset, so clear the buffer
        self.train_history_buffer.clear(fullclear=True)
        self.test_history_buffer.clear(fullclear=True)

    def state_dict_deploy(self):
        return default_state_dict(self, [
            'action_space',
            'observation_space',
            'net',
            'obs_scale',
        ])
    def load_state_dict_deploy(self, state):
        default_load_state_dict(self,state)


class PPOAgentVec(A2CAgentVec):
    def __init__(self,
            action_space : gym.spaces.Box,
            observation_space : gym.spaces.Box,
            discount_factor : float = 0.99,
            learning_rate : float = 1e-4,
            target_update_frequency : int = 40_000, # Mnih 2016 - section 8
            polyak_rate : float = 1.0,              # Mnih 2016 - section 8
            max_rollout_length : int = 5,           # Mnih 2016 - section 8 (t_max)
            num_train_envs : int = 16,
            num_test_envs : int = 4,
            obs_scale : Union[dict,float] = 1/255,
            reward_scale : float = 1,
            vf_loss_coeff : float = 1.,
            entropy_loss_coeff : float = 0.01,
            device : torch.device = torch.device('cpu'),
            optimizer : str = 'rmsprop', # adam or rmsprop
            net : PolicyValueNetwork = None,
            logger : Logger = None,
            # PPO-specific stuff
            num_minibatches : int = 5,
            minibatch_size : int = None,
        ):
        super().__init__(
                action_space = action_space,
                observation_space = observation_space,
                discount_factor = discount_factor,
                learning_rate = learning_rate,
                target_update_frequency = target_update_frequency,
                polyak_rate = polyak_rate,
                max_rollout_length = max_rollout_length,
                num_train_envs = num_train_envs,
                num_test_envs = num_test_envs,
                obs_scale = obs_scale,
                reward_scale = reward_scale,
                vf_loss_coeff = vf_loss_coeff,
                entropy_loss_coeff = entropy_loss_coeff,
                device = device,
                optimizer = optimizer,
                net = net,
                logger = logger,
        )
        self.num_minibatches = num_minibatches
        self.minibatch_size = minibatch_size
    def _train(self):
        history = self.train_history_buffer
        t_max = self.max_rollout_length

        # Check if we've accumulated enough data
        n = len(history.obs_history)
        if n < t_max:
            return

        obs = history.obs.float()*self.obs_scale
        action = history.action
        reward = history.reward
        terminal = history.terminal
        misc = history.misc
        assert isinstance(misc, dict)
        assert 'discount' in misc
        discount = misc['discount']
        target_net_output = default_collate([self.target_net(o) for o in obs])
        log_action_probs_old = None

        for _ in range(self.num_minibatches):
            # Sample minibatch
            if self.minibatch_size is not None:
                minibatch_indices = sample_minibatch([n-1,self.num_training_envs], self.minibatch_size)
            else:
                minibatch_indices = None

            # Re-evaluate values that depend on `self.net`
            net_output = default_collate([self.net(o) for o in obs])
            state_values = net_output['value'].squeeze(2)
            target_state_values = target_net_output['value'].squeeze(2)
            if isinstance(self.action_space,gym.spaces.Discrete):
                log_action_probs = net_output['action'].log_softmax(2).gather(2,action.unsqueeze(2)).squeeze(2)
                entropy = -(net_output['action'].log_softmax(2)*net_output['action'].softmax(2)).sum(2)[:n-1,:] # Remove the last step to match up with minibatch indices.
            else:
                raise NotImplementedError()

            if log_action_probs_old is None:
                log_action_probs_old = log_action_probs.clone().detach()

            # Train policy
            loss_pi = clipped_advantage_policy_gradient_loss_batch(
                    log_action_probs = log_action_probs[:n-1,:],
                    old_log_action_probs = log_action_probs_old[:n-1,:],
                    state_values = state_values[:n-1,:].detach(),
                    next_state_values = target_state_values[1:,:],
                    rewards = reward[1:,:],
                    terminals = terminal[1:,:],
                    prev_terminals = terminal[:n-1,:],
                    discounts = discount[1:,:],
                    epsilon=0.1
            )
            if minibatch_indices is not None:
                loss_pi = loss_pi[minibatch_indices[0],minibatch_indices[1]]
            loss_pi = loss_pi.mean()
            # Train value network
            state_value_estimate = monte_carlo_return_iterative_batch(
                    state_values = target_state_values[1:,:].detach(),
                    rewards = reward[1:,:],
                    terminals = terminal[1:,:],
                    discounts = discount[1:,:],
            )
            loss_v = (state_values[:-1,:]-state_value_estimate)**2
            if minibatch_indices is not None:
                loss_v = loss_v[minibatch_indices[0],minibatch_indices[1]]
            loss_v = loss_v.mean()
            # Entropy
            loss_entropy = -entropy
            if minibatch_indices is not None:
                loss_entropy = loss_entropy[minibatch_indices[0],minibatch_indices[1]]
            loss_entropy = loss_entropy.mean()

            # Take a gradient step
            loss = loss_pi+self.vf_loss_coeff*loss_v+self.entropy_loss_coeff*loss_entropy
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            # Log
            self.logger.append(
                    loss_pi=loss_pi.item(),
                    loss_v=loss_v.item(),
                    loss_entropy=loss_entropy.item(),
                    loss_total=loss.item(),
                    train_state_value=state_values.mean().item(),
                    train_state_value_target_net=target_state_values.mean().item(),
            )
        # Clear data
        self.train_history_buffer.clear()
        # Keep track of number of training steps so we know when to update the target network
        self._training_steps += 1


class A2CAgentRecurrentVec(A2CAgentVec):
    def __init__(self,
            action_space : gym.spaces.Box,
            observation_space : gym.spaces.Box,
            discount_factor : float = 0.99,
            learning_rate : float = 1e-4,
            target_update_frequency : int = 40_000, # Mnih 2016 - section 8
            polyak_rate : float = 1.0,              # Mnih 2016 - section 8
            max_rollout_length : int = 5,           # Mnih 2016 - section 8 (t_max)
            num_train_envs : int = 16,
            num_test_envs : int = 4,
            obs_scale : Union[dict,float] = 1/255,
            reward_scale : float = 1,
            vf_loss_coeff : float = 1.,
            entropy_loss_coeff : float = 0.01,
            hidden_reset_min_prob : float = 0,
            hidden_reset_max_prob : float = 0,
            target_net_hidden_state_forcing : bool = False,
            device : torch.device = torch.device('cpu'),
            optimizer : str = 'rmsprop', # adam or rmsprop
            net : PolicyValueNetworkRecurrent = None,
            logger : Logger = None,
        ):
        super().__init__(
                action_space = action_space,
                observation_space = observation_space,
                discount_factor = discount_factor,
                learning_rate = learning_rate,
                target_update_frequency = target_update_frequency,
                polyak_rate = polyak_rate,
                max_rollout_length = max_rollout_length,
                num_train_envs = num_train_envs,
                num_test_envs = num_test_envs,
                obs_scale = obs_scale,
                reward_scale = reward_scale,
                vf_loss_coeff = vf_loss_coeff,
                entropy_loss_coeff = entropy_loss_coeff,
                device = device,
                optimizer = optimizer,
                net = net,
                logger = logger,
        )
        self.target_net_hidden_state_forcing = target_net_hidden_state_forcing
        self.hidden_reset_min_prob = hidden_reset_min_prob
        self.hidden_reset_max_prob = hidden_reset_max_prob
        self._prev_hidden = None
    def _init_default_net(self, observation_space, action_space,device) -> PolicyValueNetworkRecurrent:
        if isinstance(observation_space, gym.spaces.Box):
            assert observation_space.shape is not None
            if len(observation_space.shape) == 1: # Mujoco
                return PolicyValueNetworkRecurrentFCNN(
                        num_features=observation_space.shape[0],
                        num_actions=action_space.shape[0],
                        shared_std=False
                ).to(device)
            if len(observation_space.shape) == 3: # Atari
                return PolicyValueNetworkRecurrentCNN(
                        num_actions=action_space.n,
                        hidden_size=512,
                        in_channels=observation_space.shape[0],
                ).to(device)
                #return PolicyValueNetworkRecurrentCNNNotRecurrent(
                #        num_actions=action_space.n,
                #        hidden_size=512,
                #        in_channels=observation_space.shape[0],
                #).to(device)
        raise Exception('Unsupported observation space or action space.')
    def observe(self, obs, reward=None, terminal=None, testing=False, time=1, discount=None):
        assert isinstance(self.net, PolicyValueNetworkRecurrent)

        if isinstance(self.observation_space,gym.spaces.Dict):
            batch_size = obs[next(iter(obs.keys()))].shape[0]
        else:
            batch_size = obs.shape[0]

        if discount is None:
            discount = self.discount_factor

        if testing:
            history = self.test_history_buffer
        else:
            history = self.train_history_buffer

        if reward is not None:
            reward *= self.reward_scale

        hidden_reset_prob = self._compute_annealed_epsilon(
                max_eps=self.hidden_reset_max_prob, min_eps=self.hidden_reset_min_prob, max_steps=1_000_000)
        #hidden_reset = not testing and np.random.rand(batch_size) < hidden_reset_prob
        hidden_reset = torch.logical_and(
                torch.tensor([not testing], device=self.device),
                torch.rand(batch_size, device=self.device) < hidden_reset_prob
        )
        if terminal is not None:
            hidden_reset = hidden_reset | torch.tensor(terminal, device=self.device)
        hidden_reset = hidden_reset.unsqueeze(1)
        if self._prev_hidden is None:
            self._prev_hidden = self.net.init_hidden(batch_size=batch_size)
        else:
            initial_hidden = self.net.init_hidden(batch_size=batch_size)
            self._prev_hidden = ( # FIXME: Hard-coded hidden state format
                hidden_reset.logical_not() * self._prev_hidden[0] + hidden_reset * initial_hidden[0],
                hidden_reset.logical_not() * self._prev_hidden[1] + hidden_reset * initial_hidden[1],
            )

        history.append_obs(obs, reward, terminal,
                misc={
                    'discount': torch.tensor([discount**time]*batch_size),
                    'hidden': self._prev_hidden,
                    'hidden_reset': hidden_reset
                })

        # Choose next action
        if isinstance(self.observation_space,gym.spaces.Dict):
            assert isinstance(self.obs_scale,Mapping)
            net_output = self.net(
                    {
                        k: torch.tensor(v, device=self.device).float() * self.obs_scale.get(k,1)
                        for k,v in obs.items()
                    },
                    hidden=self._prev_hidden
            )
        else:
            net_output = self.net(
                    torch.tensor(obs,device=self.device).float()*self.obs_scale,
                    hidden=self._prev_hidden
            )
        assert 'hidden' in net_output
        self._prev_hidden = tuple([x.detach() for x in net_output['hidden']])
        if isinstance(self.action_space, gym.spaces.Discrete):
            assert 'action' in net_output
            action_probs_unnormalized = net_output['action']
            action_probs = action_probs_unnormalized.softmax(1).squeeze()
            assert (torch.abs(action_probs.sum(1)-1) < 1e-6).all()
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().cpu().numpy()
        elif isinstance(self.action_space, gym.spaces.Box):
            assert 'action_mean' in net_output
            assert 'action_std' in net_output
            action_dist = torch.distributions.Normal(net_output['action_mean'],net_output['action_std'])
            action = action_dist.sample().cpu().numpy()
        else:
            raise NotImplementedError('Unsupported action space of type %s' % type(self.action_space))

        assert 'value' in net_output
        state_value = net_output['value']
        self.state_values_current.append(state_value.mean().item())

        history.append_action(action)

        # Save training data and train
        if not testing:
            self._steps += self.num_training_envs
            if not isinstance(self.logger, SubLogger):
                self.logger.log(step=self._steps)

            self.logger.log(
                    debug_state_value=state_value.mean().item(),
                    hidden_reset_prob=hidden_reset_prob,
                    train_entropy=action_dist.entropy().mean().item()
            )

            self._train()
            self._update_target()
    def _train(self):
        history = self.train_history_buffer
        t_max = self.max_rollout_length

        # Check if we've accumulated enough data
        num_points = len(history.obs_history)
        if num_points >= t_max:
            n = num_points
            try:
                misc = history.misc
                assert isinstance(misc,dict)
                hidden = misc['hidden']
                hidden_reset = misc['hidden_reset']
                discount = misc['discount']

                if isinstance(self.observation_space,gym.spaces.Dict):
                    assert isinstance(self.obs_scale,Mapping)
                    obs = {
                            k: v.float()*self.obs_scale.get(k,1)
                            for k,v in history.obs.items()
                    }
                else:
                    obs = history.obs.float()*self.obs_scale
                action = history.action
                reward = history.reward
                terminal = history.terminal

                net_output = []
                h = (hidden[0][0],hidden[1][0])
                for o,hr in zip2(obs,hidden_reset):
                    # FIXME: This assumes that the initial hidden state is 0.
                    h = ( # FIXME: Should not be hard-coded. We don't know what the hidden state format is.
                            h[0]*hr.logical_not(),
                            h[1]*hr.logical_not(),
                    )
                    no = self.net(o,h)
                    h = no['hidden']
                    net_output.append(no)
                net_output = default_collate(net_output)

                target_net_output = []
                if self.target_net_hidden_state_forcing:
                    h = (hidden[0][0,:,:],hidden[1][0,:,:])
                    for i,(o,hr) in enumerate(zip2(obs,hidden_reset)):
                        # FIXME: This assumes that the initial hidden state is 0.
                        h = ( # FIXME: Should not be hard-coded. We don't know what the hidden state format is.
                                hidden[0][i,:,:]*hr.logical_not(),
                                hidden[1][i,:,:]*hr.logical_not(),
                        )
                        no = self.target_net(o,h)
                        target_net_output.append(no)
                else:
                    h = (hidden[0][0,:,:],hidden[1][0,:,:])
                    for o,hr in zip2(obs,hidden_reset):
                        # FIXME: This assumes that the initial hidden state is 0.
                        h = ( # FIXME: Should not be hard-coded. We don't know what the hidden state format is.
                                h[0]*hr.logical_not(),
                                h[1]*hr.logical_not(),
                        )
                        no = self.target_net(o,h)
                        h = no['hidden']
                        target_net_output.append(no)
                target_net_output = default_collate(target_net_output)

                state_values = net_output['value'].squeeze(2)
                target_state_values = target_net_output['value'].squeeze(2)

                if isinstance(self.action_space,gym.spaces.Discrete):
                    log_action_probs = net_output['action'].log_softmax(2).gather(2,action.unsqueeze(2)).squeeze(2)
                    entropy = -(net_output['action'].log_softmax(2)*net_output['action'].softmax(2)).sum(2).mean()
                elif isinstance(self.action_space,gym.spaces.Box):
                    log_action_probs = compute_normal_log_prob_batch(
                            net_output['action_mean'],
                            net_output['action_std'],
                            action
                    )
                    entropy = compute_normal_entropy(
                            net_output['action_mean'],
                            net_output['action_std']
                    )
                    raise NotImplementedError('Untested code')
                else:
                    raise NotImplementedError()
                    #log_action_probs = {
                    #    ek: compute_normal_log_prob_batch(
                    #            outputs['action_mean'],
                    #            outputs['action_std'],
                    #            torch.tensor(action_history[ek],device=self.device))
                    #    for ek,outputs in net_output.items()}
                    #entropy = {ek: 
                    #    compute_normal_entropy(out['action_mean'],out['action_std'])
                    #    for ek,out in net_output.items()
                    #}
            except:
                raise
            # Train policy
            loss_pi = advantage_policy_gradient_loss_batch(
                    log_action_probs = log_action_probs[:n-1,:],
                    state_values = state_values[:n-1,:].detach(),
                    next_state_values = target_state_values[1:,:],
                    rewards = reward[1:,:],
                    terminals = terminal[1:,:],
                    prev_terminals = terminal[:n-1,:],
                    discounts = discount[1:,:],
            )
            loss_pi = loss_pi.mean()
            # Train value network
            state_value_estimate = monte_carlo_return_iterative_batch(
                    state_values = target_state_values[1:,:].detach(),
                    rewards = reward[1:,:],
                    terminals = terminal[1:,:],
                    discounts = discount[1:,:],
            )
            loss_v = (state_values[:-1,:]-state_value_estimate)**2
            loss_v = loss_v.mean()
            # Entropy
            loss_entropy = -entropy.mean()
            # Take a gradient step
            loss = loss_pi+self.vf_loss_coeff*loss_v+self.entropy_loss_coeff*loss_entropy
            self.optimizer.zero_grad()
            loss.backward()
            if __debug__:
                for p in self.net.parameters():
                    if p.grad is None:
                        continue
                    if p.grad.isnan().any():
                        print('NaN gradients')
                        breakpoint()
            self.optimizer.step()
            # Clear data
            self.train_history_buffer.clear()
            # Log
            self._training_steps += 1
            self.logger.log(
                    loss_pi=loss_pi.item(),
                    loss_v=loss_v.item(),
                    loss_entropy=loss_entropy.item(),
                    loss_total=loss.item(),
                    train_state_value=state_values.mean().item(),
                    train_state_value_target_net=target_state_values.mean().item(),
            )

    def _compute_annealed_epsilon(self, max_eps, min_eps, max_steps=1_000_000):
        if max_steps == 0:
            return min_eps
        #return (1-eps)*max(1-self._steps/max_steps,0)+eps # Linear
        return (max_eps-min_eps)*np.exp(-self._steps/max_steps)+min_eps # Exponential

    def state_dict(self):
        return {
                **super().state_dict(),
                '_prev_hidden': self._prev_hidden,
                'hidden_reset_min_prob': self.hidden_reset_min_prob,
                'hidden_reset_max_prob': self.hidden_reset_max_prob,
        }
    def load_state_dict(self, state):
        self._prev_hidden = state.pop('_prev_hidden')
        self.hidden_reset_min_prob = state.pop('hidden_reset_min_prob')
        self.hidden_reset_max_prob = state.pop('hidden_reset_max_prob')
        super().load_state_dict(state)


if __name__ == "__main__":
    import torch.cuda
    import numpy as np
    import gym
    import gym.envs
    #import pybullet_envs

    from rl.experiments.training.basic import TrainExperiment
    from rl.experiments.training.vectorized import TrainExperiment as TrainExperimentVec
    from experiment import make_experiment_runner

    def run_atari():
        from rl.agent.smdp.a2c import A2CAgent

        num_actors = 16
        env_name = 'ALE/Pong-v5'
        train_env_keys = list(range(num_actors))
        env_config = {
            'frameskip': 1,
            'mode': 0,
            'difficulty': 0,
            'repeat_action_probability': 0.25,
        }

        exp_runner = make_experiment_runner(
                TrainExperiment,
                config={
                    'agent': {
                        'type': A2CAgent,
                        'parameters': {
                            'training_env_keys': train_env_keys,
                        },
                    },
                    'env_test': {'env_name': env_name, 'atari': True, 'config': env_config},
                    'env_train': {'env_name': env_name, 'atari': True, 'config': env_config},
                    'train_env_keys': train_env_keys,
                    'test_frequency': None,
                    'save_model_frequency': 250_000,
                    'verbose': True,
                },
                #trial_id='checkpointtest',
                checkpoint_frequency=250_000,
                #checkpoint_frequency=None,
                max_iterations=50_000_000,
                verbose=True,
        )
        #exp_runner.exp.logger.init_wandb({
        #    'project': 'A2C-%s' % env_name.replace('/','_')
        #})
        #exp_runner.exp.logger.init_wandb({
        #    'project': 'PPO-%s' % env_name.replace('/','_')
        #})
        exp_runner.run()

    def run_mujoco():
        from rl.agent.smdp.a2c import A2CAgent

        num_actors = 16
        env_name = 'Hopper-v3'
        train_env_keys = list(range(num_actors))
        env_config = {}

        exp_runner = make_experiment_runner(
                TrainExperiment,
                config={
                    'agent': {
                        'type': A2CAgent,
                        'parameters': {
                            'training_env_keys': train_env_keys,
                            'max_rollout_length':128,
                            'reward_scale':5,
                            'obs_scale':1,
                        },
                    },
                    'env_test': {'env_name': env_name, 'config': env_config},
                    'env_train': {'env_name': env_name, 'config': env_config},
                    'train_env_keys': train_env_keys,
                    'test_frequency': None,
                    'save_model_frequency': 250_000,
                    'verbose': True,
                },
                #trial_id='checkpointtest',
                checkpoint_frequency=250_000,
                #checkpoint_frequency=None,
                max_iterations=50_000_000,
                verbose=True,
        )
        #exp_runner.exp.logger.init_wandb({
        #    'project': 'A2C-%s' % env_name.replace('/','_')
        #})
        exp_runner.exp.logger.init_wandb({
            'project': 'PPO-%s' % env_name.replace('/','_')
        })
        exp_runner.run()

    def run_atari_ppo():
        from rl.agent.smdp.a2c import PPOAgent

        num_actors = 16
        env_name = 'ALE/Pong-v5'
        train_env_keys = list(range(num_actors))
        env_config = {
            'frameskip': 1,
            'mode': 0,
            'difficulty': 0,
            'repeat_action_probability': 0.25,
        }

        exp_runner = make_experiment_runner(
                TrainExperiment,
                config={
                    'agent': {
                        'type': PPOAgent,
                        'parameters': {
                            'training_env_keys': train_env_keys,
                            'num_minibatches': 1,
                        },
                    },
                    'env_test': {'env_name': env_name, 'atari': True, 'config': env_config},
                    'env_train': {'env_name': env_name, 'atari': True, 'config': env_config},
                    'train_env_keys': train_env_keys,
                    'test_frequency': None,
                    'save_model_frequency': 250_000,
                    'verbose': True,
                },
                #trial_id='checkpointtest',
                checkpoint_frequency=250_000,
                #checkpoint_frequency=None,
                max_iterations=50_000_000,
                verbose=True,
        )
        exp_runner.exp.logger.init_wandb({
            'project': 'PPO-%s' % env_name.replace('/','_')
        })
        exp_runner.run()

    def run_atari_ppo_vec():
        from rl.agent.smdp.a2c import PPOAgentVec

        num_envs = 16
        env_name = 'Pong-v5'
        env_config = {
            'env_type': 'envpool',
            'env_configs': {
                'env_name': env_name,
                'atari': True,
                'atari_config': {
                    'num_envs': num_envs,
                    'stack_num': 4,
                    'repeat_action_probability': 0.25,
                }
            }
        }

        exp_runner = make_experiment_runner(
                TrainExperimentVec,
                config={
                    'agent': {
                        'type': PPOAgentVec,
                        'parameters': {
                            'num_train_envs': num_envs,
                            'num_test_envs': num_envs,
                            'num_minibatches': 5,
                        },
                    },
                    'env_test': env_config,
                    'env_train': env_config,
                    'test_frequency': None,
                    'save_model_frequency': None,
                    'verbose': True,
                },
                #trial_id='checkpointtest',
                #checkpoint_frequency=250_000,
                checkpoint_frequency=None,
                max_iterations=50_000_000,
                verbose=True,
        )
        exp_runner.exp.logger.init_wandb({
            'project': 'PPOAgentVec-%s' % env_name.replace('/','_')
        })
        exp_runner.run()

    def run_atari_recurrent():
        from rl.agent.smdp.a2c import A2CAgentRecurrent

        num_actors = 16
        env_name = 'ALE/Pong-v5'
        train_env_keys = list(range(num_actors))
        env_config = {
            'frameskip': 1,
            'mode': 0,
            'difficulty': 0,
            'repeat_action_probability': 0.25,
        }

        exp_runner = make_experiment_runner(
                TrainExperiment,
                config={
                    'agent': {
                        'type': A2CAgentRecurrent,
                        'parameters': {
                            'training_env_keys': train_env_keys,
                            #'max_rollout_length': 32
                        },
                    },
                    'env_test': {'env_name': env_name, 'atari': True, 'config': env_config, 'frame_stack': 4},
                    'env_train': {'env_name': env_name, 'atari': True, 'config': env_config, 'frame_stack': 4},
                    'train_env_keys': train_env_keys,
                    'test_frequency': None,
                    'save_model_frequency': 250_000,
                    'verbose': True,
                },
                #trial_id='checkpointtest',
                checkpoint_frequency=250_000,
                #checkpoint_frequency=None,
                #max_iterations=50_000_000,
                max_iterations=500_000,
                verbose=True,
        )
        #exp_runner.exp.logger.init_wandb({
        #    'project': 'A2C-recurrent-%s' % env_name.replace('/','_')
        #})
        exp_runner.run()

    def run_atari_envpool():
        import envpool
        from tqdm import tqdm
        import itertools

        device = torch.device('cuda')
        #device = torch.device('cpu')
        num_envs = 32
        env_name = 'Pong-v5'

        env = envpool.make(env_name, env_type="gym", num_envs=num_envs)
        agent = A2CAgentVec(
            observation_space=env.observation_space,
            action_space=env.action_space,
            target_update_frequency=32_000,
            num_train_envs=num_envs,
            num_test_envs=5,
            max_rollout_length=8,
            device=device,
        )
        #agent.logger.init_wandb({
        #    'project': 'A2C-envpool-%s' % env_name.replace('/','_')
        #})

        total_reward = 0
        obs = env.reset()
        agent.observe(obs)
        for _ in tqdm(itertools.count()):
            action = agent.act()
            obs, reward, done, _ = env.step(action)
            agent.observe(obs, reward, done)

            total_reward += reward

            if done.any():
                agent.logger.log(reward=total_reward[done].mean().item())
                tqdm.write(str(total_reward[done].tolist()))
                total_reward = total_reward*(1-done)

    #run_mujoco()
    #run_atari()
    #run_atari_ppo()
    run_atari_ppo_vec()
    #run_atari_recurrent()
    #run_atari_envpool()
