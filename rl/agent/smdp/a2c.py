import copy
from collections import defaultdict
from typing import Optional, Tuple, List, Union
from typing_extensions import TypedDict

import numpy as np
import torch
import torch.nn
import torch.utils.data
import torch.distributions
import torch.optim
import gym.spaces
from torchtyping import TensorType

from frankenstein.loss.policy_gradient import advantage_policy_gradient_loss, clipped_advantage_policy_gradient_loss
from frankenstein.value.monte_carlo import monte_carlo_return_iterative
from frankenstein.buffer.history import HistoryBuffer
from experiment.logger import Logger, SubLogger
from rl.utils import default_state_dict, default_load_state_dict
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

def zip_dict(*dcts):
    if not dcts:
        return
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i, tuple(d[i] for d in dcts))

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
                discount_history = {ek: history[i].misc['discount'] for i,ek in enumerate(self.training_env_keys)}
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
            self.state_values.append(np.mean(self.state_values_current))
            self.state_values_std.append(np.std(self.state_values_current))
            self.state_values_std_ra.append(np.mean(self.state_values_std) if self._training_steps < 100 else np.mean(self.state_values_std[-100:]))
            self.state_values_current = []
            self.logger.log(
                    loss_pi=loss_pi.item(),
                    loss_v=loss_v.item(),
                    loss_entropy=loss_entropy.item(),
                    loss_total=loss.item(),
                    train_state_value=np.mean([v.mean().item() for v in state_values.values()]),
                    train_state_value_target_net=np.mean([v.mean().item() for v in target_state_values.values()]),
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
        return {
            'action_space': self.action_space,
            'observation_space': self.observation_space,
            # TODO
        }
    def load_state_dict_deploy(self, state):
        state = state
        pass # TODO

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
                discount_history = {ek: history[i].misc['discount'] for i,ek in enumerate(self.training_env_keys)}
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
            loss_v = [compute_mc_state_value_loss(
                        state_values = state_values[ek],
                        last_state_target_value = float(target_state_values[ek][-1].item()),
                        rewards = reward_history[ek],
                        terminals = terminal_history[ek],
                        discounts = discount_history[ek],
                ) for ek in self.training_env_keys]
            loss_v = torch.stack(loss_v).mean()
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
                    train_state_value=np.mean([v.mean().item() for v in state_values.values()]),
                    train_state_value_target_net=np.mean([v.mean().item() for v in target_state_values.values()]),
            )
        # Clear data
        history.clear()
        # Log
        self._training_steps += 1
        self.state_values.append(np.mean(self.state_values_current))
        self.state_values_std.append(np.std(self.state_values_current))
        self.state_values_std_ra.append(np.mean(self.state_values_std) if self._training_steps < 100 else np.mean(self.state_values_std[-100:]))
        self.state_values_current = []

if __name__ == "__main__":
    import torch.cuda
    import numpy as np
    import gym
    import gym.envs
    #import pybullet_envs

    from rl.experiments.training.basic import TrainExperiment
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
        exp_runner.exp.logger.init_wandb({
            'project': 'A2C-%s' % env_name.replace('/','_')
        })
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
        #exp_runner.exp.logger.init_wandb({
        #    'project': 'PPO-%s' % env_name.replace('/','_')
        #})
        exp_runner.run()

    #run_mujoco()
    #run_atari()
    run_atari_ppo()
