import warnings
from typing import Optional, Tuple, Generic, TypeVar, Union, Mapping
from collections import defaultdict
import copy
import os

import gym
import gym.spaces
import torch
import torch.nn
import torch.utils.data
import torch.distributions
import torch.optim
import numpy as np
import dill

from experiment.logger import Logger, SubLogger

from rl.utils import default_state_dict, default_load_state_dict
from rl.agent.agent import DeployableAgent
from rl.agent.smdp.a2c import compute_advantage_policy_gradient, compute_entropy, compute_mc_state_value_loss

class QUNetworkCNN(torch.nn.Module):
    def __init__(self, num_actions, num_options):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,out_channels=32,kernel_size=8,stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=32,out_channels=64,kernel_size=4,stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,out_channels=64,kernel_size=3,stride=1),
            torch.nn.ReLU(),

            torch.nn.Flatten(),

            torch.nn.Linear(in_features=64*7*7,out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512,out_features=num_actions*num_options),

            torch.nn.Unflatten(1, (num_options,num_actions))
        )
    def forward(self, obs):
        return self.seq(obs)

class OptionCriticNetworkCNN(torch.nn.Module):
    def __init__(self, num_actions, num_options):
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,out_channels=32,kernel_size=8,stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=32,out_channels=64,kernel_size=4,stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,out_channels=64,kernel_size=3,stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=64*7*7,out_features=512),
            torch.nn.ReLU(),
        )
        # Termination
        self.beta = torch.nn.Sequential(
            torch.nn.Linear(in_features=512,out_features=num_options),
            torch.nn.Sigmoid(), # https://github.com/jeanharb/option_critic/blob/5d6c81a650a8f452bc8ad3250f1f211d317fde8c/neural_net.py#L59
        )
        # Intra-option policies
        self.iop = torch.nn.Sequential(
            torch.nn.Linear(in_features=512,out_features=num_actions*num_options),
            torch.nn.Unflatten(1, (num_options,num_actions))
        )
        # Policy over options
        self.poo = torch.nn.Sequential(
            torch.nn.Linear(in_features=512,out_features=num_options),
        )
        # Option-value
        self.q = torch.nn.Sequential(
            torch.nn.Linear(in_features=512,out_features=num_options),
        )
    def forward(self, obs):
        x = obs
        x = self.head(x)
        x = {
            'beta': self.beta(x),
            'iop': self.iop(x),
            'poo': self.poo(x),
            'q': self.q(x),
        }
        return x

class OptionCriticNetworkDiscrete(torch.nn.Module):
    def __init__(self, obs_size, num_actions, num_options):
        super().__init__()
        # Termination
        self.beta = torch.nn.Parameter(torch.zeros([obs_size,num_options]))
        # Intra-option policies
        self.iop = torch.nn.Parameter(torch.zeros([obs_size,num_options,num_actions]))
        # Policy over options
        self.poo = torch.nn.Parameter(torch.zeros([obs_size,num_options]))
        # Option-value
        self.q = torch.nn.Parameter(torch.zeros([obs_size,num_options]))
    def forward(self, obs):
        x = {
            'beta': torch.sigmoid(torch.cat([self.beta[o] for o in obs])),
            'iop': torch.cat([self.iop[o,:,:] for o in obs]),
            'poo': torch.cat([self.poo[o,:] for o in obs]),
            'q': torch.cat([self.q[o,:] for o in obs]),
        }
        return x

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

class ObservationStack2(Generic[ObsType,ActionType]):
    def __init__(self, max_len=None, default_action=None, default_reward=None) -> None:
        """
        Args:
            max_len: The maximum number of transitions to store
            default_action: This action is added to `action_history` at the end of an episode as padding.
            default_reward: The reward that is added to `reward_history` on the first step of an episode.
        """
        self.max_len = max_len
        self.default_action = default_action
        self.default_reward = default_reward
        self.obs_history = []
        self.action_history = []
        self.reward_history = []
        self.terminal_history = []
    def __len__(self):
        return len(self.obs_history)
    def append_obs(self, obs : ObsType, reward : Optional[float], terminal : bool):
        # Handle boundary between episodes
        if reward is None:
            if len(self.obs_history) != 0:
                # The last episode just finished, so we receive a new observation to start the episode without an action in between.
                # Add an action to pad the actions list.
                self.append_action(self.default_action)
            reward = self.default_reward

        # Make sure the observations and actions are in sync
        assert len(self.obs_history) == len(self.action_history)

        # Save data
        self.obs_history.append(obs)
        self.reward_history.append(reward)
        self.terminal_history.append(terminal)

        # Enforce max length
        if self.max_len is not None and len(self.obs_history) > self.max_len+1:
            self.obs_history = self.obs_history[1:]
            self.reward_history = self.reward_history[1:]
            self.terminal_history = self.terminal_history[1:]
            self.action_history = self.action_history[1:]
    def append_action(self, action : ActionType):
        assert len(self.obs_history) == len(self.action_history)+1
        self.action_history.append(action)
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
        num_obs = len(self.obs_history)
        num_actions = len(self.action_history)
        if num_obs < 2:
            return None
        if num_obs-num_actions > 1:
            raise Exception('Observations and actions out of sync.')
        i = num_obs-2
        s0 = self.obs_history[i]
        a0 = self.action_history[i]
        s1 = self.obs_history[i+1]
        r1 = self.reward_history[i+1]
        t1 = self.terminal_history[i+1]
        return s0, a0, r1, s1, t1
    def get_obs(self) -> ObsType:
        if len(self.obs_history) == 0:
            raise Exception('No observation available.')
        return self.obs_history[-1]
    def clear(self):
        i = len(self.obs_history)-1
        self.obs_history = self.obs_history[i:]
        self.reward_history = self.reward_history[i:]
        self.terminal_history = self.terminal_history[i:]
        self.action_history = self.action_history[i:]
    def state_dict(self):
        return default_state_dict(self, ['obs_history','action_history','reward_history','terminal_history'])
    def load_state_dict(self,state):
        default_load_state_dict(self,state)

class OptionCriticAgent(DeployableAgent):
    """
    On-policy implementation of Option-Critic.
    """
    def __init__(self,
            action_space : gym.spaces.Discrete,
            observation_space : gym.spaces.Space,
            discount_factor : float = 0.99,
            learning_rate : float = 2.5e-4,
            update_frequency : int = 4,
            batch_size : int = 32,
            target_update_frequency : int = 100,
            polyak_rate : float = 1.,
            device = torch.device('cpu'),
            behaviour_eps : float = 0.1,
            target_eps : float = 0.05,
            eps_annealing_steps : int = 1_000_000,
            num_options : int = 8,
            termination_reg : float = 0.01, # From paper
            entropy_reg : float = 0.01, # XXX: Arbitrary value
            net : torch.nn.Module = None,
            logger : Logger = None,
            ):
        """
        Args:
            action_space: Gym action space.
            observation_space: Gym observation space.
            behaviour_eps (float): Minimum probability of taking a uniformly random action during on each training step. The actual probability is annealed over a number of steps. In Mnih 2015, "epsilon [is] annealed linearly from 1.0 to 0.1 over the first million frames, and fixed at 0.1 thereafter."
            eps_annealing_steps (int): Number of steps over which the behaviour policy's epsilon randomness decreases from 1 to `behaviour_eps`.
            update_frequency (int): The number of actions selected by the agent between successive SGD updates. Using a value of 4 results in the agent selection 4 actions between each pair of successive updates.
            target_update_frequency (int): The frequency with which the target Q network is updated. This is measured in terms of number of training steps (i.e. it only starts counting after `warmup_steps` steps).
        """
        self.action_space = action_space
        self.observation_space = observation_space
        if isinstance(observation_space,gym.spaces.Box):
            self.obs_scale = observation_space.high.max()

        self.discount_factor = discount_factor
        self.update_frequency = update_frequency
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.polyak_rate = polyak_rate
        self.device = device
        self.eps = [behaviour_eps, target_eps]
        self.eps_annealing_steps = eps_annealing_steps
        self.num_options = num_options
        self.termination_reg = termination_reg
        self.entropy_reg = entropy_reg

        # State (training)
        self.obs_stack = defaultdict(lambda: ObservationStack2(default_action=(0,0)))
        self._steps = 0 # Number of steps experienced
        self._training_steps = 0 # Number of training steps experienced
        self._current_option = {} # Option that is currently executing
        self._current_option_duration = {} # How long the current option has been executing

        if net is not None:
            self.net = net
        else:
            if isinstance(observation_space,gym.spaces.Box):
                self.net = OptionCriticNetworkCNN( # Termination, intra-option policy, and policy over options
                        num_actions=action_space.n,
                        num_options=num_options
                ).to(self.device)
            elif isinstance(observation_space,gym.spaces.Discrete):
                self.net = OptionCriticNetworkDiscrete( # Termination, intra-option policy, and policy over options
                        obs_size=observation_space.n,
                        num_actions=self.action_space.n,
                        num_options=num_options
                ).to(self.device)
            else:
                raise NotImplementedError(f'Unsupported observation space: {type(self.observation_space)}')
        self.net_target = copy.deepcopy(self.net)

        self.policy_losses = []
        self.entropy_losses = []
        self.termination_losses = []
        self.critic_losses = []

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        self._train_env_keys = set()
        self._num_training_transitions = 0

        # Logging
        if logger is None:
            self.logger = Logger(key_name='step', allow_implicit_key=True)
            self.logger.log(step=0)
        else:
            self.logger = logger
        self._option_choice_count = defaultdict(lambda: [0]*num_options)

    def observe(self, obs, reward=None, terminal=False, testing=False, env_key=None):
        if env_key is None:
            env_key = testing
        if not testing:
            # Count the number of transitions available for training
            self._train_env_keys.add(env_key)
            if len(self.obs_stack[env_key].obs_history) != 0:
                self._num_training_transitions += 1
        else:
            # We don't need to store more than one transition on testing environments because we're not training on that data.
            self.obs_stack[env_key].max_len = 1
            # Make sure we're not reusing the same key for testing and training
            if env_key in self._train_env_keys:
                raise Exception(f'Environment key "{env_key}" was previously used in training. The same key cannot be reused for testing.')

        # Reward clipping
        if reward is not None:
            reward = np.clip(reward, -1, 1)

        self.obs_stack[env_key].append_obs(obs, reward, terminal)

        # Add to replay buffer
        transition = self.obs_stack[env_key].get_transition()
        if transition is not None and not testing:
            self._steps += 1
            if not isinstance(self.logger, SubLogger):
                self.logger.log(step=self._steps)
            
            # Train each time something is added to the buffer
            #self._train(env_key)

        if terminal:
            # Logging
            counts = self._option_choice_count[env_key]
            total = sum(counts)
            probs = [c/total for c in counts]
            entropy = sum([-p*np.log(p) for p in probs if p != 0])
            self.logger.append(option_choice_entropy=entropy)
            self.logger.log(option_choice_count=self._option_choice_count[env_key])
            self._option_choice_count[env_key] = [0]*self.num_options # Reset count
            # Reset
            self._current_option[env_key] = None

        # Train if appropriate
        if self._num_training_transitions >= self.batch_size:
            self._train2()
    def _train(self, env_key):
        self._train_actor_acc_loss(env_key)
        self._train_critic_acc_loss(env_key)
        if len(self.policy_losses) < self.batch_size:
            return
        self._train_actor_critic()
        self._training_steps += 1
    def _train2(self):
        all_losses = [
                self._compute_loss(env_key)
                for env_key in self._train_env_keys
        ]
        loss_policy = torch.stack([l['policy'] for l in all_losses]).mean()
        loss_entropy = torch.stack([l['entropy'] for l in all_losses]).mean()
        loss_termination = torch.stack([l['termination'] for l in all_losses]).mean()
        loss_q = torch.stack([l['q'] for l in all_losses]).mean()

        loss = loss_policy+loss_entropy+loss_termination+loss_q

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._training_steps += 1

        self._update_target()

        # Clear lists
        for env_key in self._train_env_keys:
            self.obs_stack[env_key].clear()
        self._num_training_transitions = 0

        # Logging
        self.logger.log(
                total_loss=loss.item(),
                termination_loss=loss_termination.item(),
                policy_loss=loss_policy.item(),
                entropy_loss=loss_entropy.item(),
                critic_loss=loss_q.item(),
        )
    def _compute_loss(self, env_key):
        obs_stack = self.obs_stack[env_key]

        obs = torch.stack([
            torch.tensor(o,device=self.device)
            for o in obs_stack.obs_history
        ]).float()/self.obs_scale
        num_obs = obs.shape[0]
        num_actions = num_obs-1
        batch0 = range(num_actions)
        batch1 = range(1,num_actions+1)
        option,action = zip(*obs_stack.action_history)

        net_output = self.net(obs)
        net_output_target = self.net_target(obs)
        q = net_output['q']
        qt = net_output_target['q']

        #target_state_values_max = qt.max(1)[0]
        beta1 = net_output['beta'][batch1,option]
        last_beta = net_output['beta'][-1,option[-1]] # Probability of terminating option n-2 at state n-1. Note that there is no distinct option associated with the last state. It is not a mistake that the indices are misaligned.
        target_state_value_last = (1-last_beta)*qt[-1,option[-1]]+last_beta*qt[-1,:].max()
        target_state_values = torch.cat([
            net_output_target['q'][batch0,option],
            target_state_value_last.unsqueeze(0) # Compute as an expectation because we don't know yet if the option will be terminated
        ])
        if isinstance(self.action_space,gym.spaces.Discrete):
            log_action_probs = net_output['iop'].log_softmax(2)[batch0,option,action]
            entropy = [
                    compute_entropy(iop.log_softmax(1).squeeze())
                    for iop in net_output['iop']
            ]
        else:
            raise NotImplementedError()
        # Policy loss
        loss_policy = compute_advantage_policy_gradient(
                log_action_probs=log_action_probs,
                target_state_values=target_state_values,
                rewards=obs_stack.reward_history,
                terminals=obs_stack.terminal_history,
                discounts=[self.discount_factor]*len(obs_stack)
        )
        # Entropy loss
        loss_entropy = -self.entropy_reg*torch.stack(entropy).mean()
        # Termination loss
        def compute_termination_loss(
                termination_prob,
                option_values_current,
                option_values_max,
                termination_reg):
            """
            Option termination gradient.

            Given a sequence of length n, we observe states/options/actions/rewards r_0,s_0,o_0,a_0,r_1,s_1,o_1,a_1,r_2,s_2,...,r_{n-1},s_{n-1},o_{n-1},a_{n-1}.

            Args:
                termination_prob: A list of n elements. The element at index i is a ???-dimensional tensor containing the predicted probability of terminating option o_{i-1} at state s_i.
                option_values_current: A list of n elements. The element at index i is the value of option o_{i-1} at state s_i.
                option_values_max: A list of n elements. The element at index i is the largest option value over all options at state s_i.
                termination_reg (float): A regularization constant to ensure that termination probabilities do not all converge on 1. Value must be greater than 0.
            """
            #option_values_current = torch.stack(option_values_current)
            #option_values_max = torch.stack(option_values_max)
            #termination_prob = torch.stack(termination_prob)
            advantage = option_values_current-option_values_max+termination_reg
            advantage = advantage.detach()
            loss = (termination_prob*advantage).mean(0)
            return loss
        loss_term = compute_termination_loss(
                termination_prob = beta1,
                option_values_current = net_output['q'][batch1,option],
                option_values_max = net_output['q'][batch1,:].max(1)[0],
                termination_reg = self.termination_reg
        )
        # Q loss
        loss_q = compute_mc_state_value_loss(
                state_values = q[batch0,option],
                #last_state_target_value = float(target_state_values_max[-1].item()),
                last_state_target_value = target_state_value_last.item(),
                rewards=obs_stack.reward_history,
                terminals=obs_stack.terminal_history,
                discounts=[self.discount_factor]*len(obs_stack)
        )
        return {
                'policy': loss_policy,
                'entropy': loss_entropy,
                'termination': loss_term,
                'q': loss_q,
        }
    def _train_critic_acc_loss(self, env_key):
        """ Using A2C state value implementation. Accumulate loss without updating. """
        transition = self.obs_stack[env_key].get_transition()
        if transition is None:
            return
        s0,(o0,a0),r1,s1,term = transition
        if isinstance(self.observation_space,gym.spaces.Discrete):
            s0 = torch.tensor(s0).to(self.device)
            s1 = torch.tensor(s1).to(self.device)
        else:
            s0 = torch.tensor(s0).to(self.device).float().unsqueeze(0)/self.obs_scale
            s1 = torch.tensor(s1).to(self.device).float().unsqueeze(0)/self.obs_scale
        a0 = a0

        net_output_0 = self.net(s0)
        net_output_target_1 = self.net_target(s1)
        q1_target = net_output_target_1['q'].squeeze(0)

        # State value loss
        loss_q = compute_mc_state_value_loss(
                state_values = [net_output_0['q'][0,o0],net_output_0['q'][0,o0]],
                last_state_target_value=q1_target.max(),
                rewards=[None,r1],
                terminals=[False,term],
                discounts=[self.discount_factor,self.discount_factor]
        )
        self.critic_losses.append(loss_q)

        ## Logging
        #self.logger.append(
        #        state_option_value=net_output_0['q'][0,o0].item(),
        #)
    def _train_actor_acc_loss(self, env_key):
        """ Using A2C policy gradient implementation. Termination is ignored. Accumulate loss without updating. """
        transition = self.obs_stack[env_key].get_transition()
        if transition is None:
            return
        s0,(o0,a0),r1,s1,term = transition
        if isinstance(self.observation_space,gym.spaces.Discrete):
            s0 = torch.tensor(s0).to(self.device)
            s1 = torch.tensor(s1).to(self.device)
        else:
            s0 = torch.tensor(s0).to(self.device).float().unsqueeze(0)/self.obs_scale
            s1 = torch.tensor(s1).to(self.device).float().unsqueeze(0)/self.obs_scale

        net_output_0 = self.net(s0)
        net_output_1 = self.net(s1)
        net_output_target_0 = self.net_target(s0)
        net_output_target_1 = self.net_target(s1)
        policy0 = net_output_0
        #policy1 = self.policy_net(s1)
        q0_target = net_output_target_0['q'].squeeze(0)
        q1_target = net_output_target_1['q'].squeeze(0)

        # Policy
        log_action_probs = [net_output_0['iop'].squeeze(0)[o0,:].log_softmax(0)[a0],None]
        target_state_values = [q0_target.max(),q1_target.max()]
        rewards = [None,r1]
        terminals = [False,term]
        discounts = [self.discount_factor, self.discount_factor]
        loss_policy = compute_advantage_policy_gradient(
                log_action_probs=log_action_probs,
                target_state_values=target_state_values,
                rewards=rewards,
                terminals=terminals,
                discounts=discounts
        )
        loss_policy = loss_policy.squeeze() # Should be a 0D tensor

        # Entropy
        action_probs = policy0['iop'].squeeze(0)[o0,:].softmax(0)
        entropy = -action_probs*torch.log(action_probs)
        loss_entropy = -0.01*entropy.sum(0) # Factor of 0.01 works in A2C

        # Termination
        termination_reg = self.termination_reg
        q_current_option = net_output_1['q'][0,o0]
        v = net_output_1['q'].max()
        beta1 = net_output_1['beta'][0,o0]
        advantage = (q_current_option-v+termination_reg).detach()
        loss_term = (beta1*advantage).mean(0)

        # Accumulate loss
        self.policy_losses.append(loss_policy)
        self.entropy_losses.append(loss_entropy)
        self.termination_losses.append(loss_term)
    def _train_actor_critic(self):
        """ Update actor using accumulated losses. """
        # Update policy network
        p_loss = torch.stack(self.policy_losses).mean(0)
        e_loss = torch.stack(self.entropy_losses).mean(0)
        t_loss = torch.stack(self.termination_losses).mean(0)
        q_loss = torch.stack(self.critic_losses).mean(0)
        loss = p_loss+e_loss+t_loss+q_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._update_target()

        # Clear lists
        self.policy_losses.clear()
        self.entropy_losses.clear()
        self.termination_losses.clear()
        self.critic_losses.clear()

        # Logging
        self.logger.log(
                policy_term_ent_loss=loss.item(),
                termination_loss=t_loss.item(),
                policy_loss=p_loss.item(),
                entropy_loss=e_loss.item(),
                critic_loss=q_loss.item(),
        )
    def _update_target(self):
        if self._training_steps % self.target_update_frequency != 0:
            return
        tau = self.polyak_rate
        for p1,p2 in zip(self.net_target.parameters(), self.net.parameters()):
            p1.data = (1-tau)*p1+tau*p2
    def act(self, testing=False, env_key=None):
        """ Return a random action according to an epsilon-greedy policy. """
        if env_key is None:
            env_key = testing

        if testing:
            eps = self.eps[testing]
        else:
            eps = self._compute_annealed_epsilon(self.eps_annealing_steps)

        obs = self.obs_stack[env_key].get_obs()
        if isinstance(self.observation_space,gym.spaces.Discrete):
            obs = torch.tensor(obs).to(self.device)
        else:
            obs = torch.tensor(obs).to(self.device).float()/self.obs_scale
        if isinstance(self.observation_space,gym.spaces.Discrete):
            obs = obs.unsqueeze(0).unsqueeze(0)
        elif isinstance(self.observation_space,gym.spaces.Box):
            obs = obs.unsqueeze(0)
        else:
            raise NotImplementedError()

        curr_option = self._current_option.get(env_key)
        pi = self.net(obs)
        # Choose an option
        terminate_option = pi['beta'][0,curr_option].item() > torch.rand(1).item() if curr_option is not None else False
        if curr_option is None or terminate_option:
            vals = pi['q']
            if torch.rand(1) >= eps:
                option = vals.max(dim=1)[1].item()
            else:
                option = torch.randint(0,self.num_options,[1]).item()
        else:
            option = curr_option
        # Choose an action
        action_probs = pi['iop'][0,option,:].softmax(dim=0)
        dist = torch.distributions.Categorical(probs=action_probs)
        action = dist.sample().item()

        self.obs_stack[env_key].append_action((option,action))

        # Save state and logging
        if curr_option is None or terminate_option:
            if curr_option is not None: # If we terminated an option
                if testing:
                    self.logger.append(
                            testing_option_duration=self._current_option_duration[env_key],
                            testing_option_choice=option
                    )
                else:
                    self.logger.append(
                            training_option_duration=self._current_option_duration[env_key],
                            training_option_choice=option
                    )
            self._current_option[env_key] = option
            self._current_option_duration[env_key] = 1
        else:
            self._current_option_duration[env_key] += 1

        assert isinstance(option,int)
        self._option_choice_count[env_key][option] += 1

        if testing:
            vals = pi['q']
            self.logger.append(
                    #step=self._steps,
                    #testing_action_value=vals[0,option,action].item(),
                    testing_option_value=vals[0,option].item(),
            )
        else:
            self.logger.log(
                state_option_value=pi['q'][0,option].item(),
                    #action_value=self.q_net(obs)[0,option,action].item(),
                    #entropy=dist.entropy().item(),
                    #action_value_diff=(pq.max()-(action_probs.squeeze() * pq.squeeze()).sum()).item()
            )

        return action
    def _compute_annealed_epsilon(self, max_steps=1_000_000):
        eps = self.eps[False]
        if max_steps == 0:
            return eps
        return (1-eps)*max(1-self._steps/max_steps,0)+eps # Linear
        #return (1-eps)*np.exp(-self._steps/max_steps)+eps # Exponential

    def state_dict(self):
        return {
                'net': self.net.state_dict(),
                #'qu_net': self.q_net.state_dict(),
                #'qu_net_target': self.q_net_target.state_dict(),
                #'policy_net': self.policy_net.state_dict(),

                'optimizer': self.optimizer.state_dict(),
                #'optimizer_qu': self.optimizer_q.state_dict(),
                #'optimizer_policy': self.optimizer_policy.state_dict(),

                'obs_stack': {k:os.state_dict() for k,os in self.obs_stack.items()},
                'steps': self._steps, # TODO: subtract length of saved losses. Those losses can't be saved because the computation graphs won't be preserved, so we'll dump those and pretend uncount those transitions from the number of steps experienced. But then it'll be inconsistent with the step number in the experiment. How can we resolve this?
                'training_steps': self._training_steps,
                'logger': self.logger.state_dict(),
        }
    def load_state_dict(self, state):
        self.net.load_state_dict(state['net'])
        self.optimizer.load_state_dict(state['optimizer'])

        for k,os_state in state['obs_stack'].items():
            self.obs_stack[k].load_state_dict(os_state)
        self._steps = state['steps']
        self._training_steps = state['training_steps']
        self.logger.load_state_dict(state['logger'])

        warnings.warn('The replay buffer is not saved. Training the agent from this point may yield unexpected results.')

    def state_dict_deploy(self):
        return {
                'action_space': self.action_space,
                'observation_space': self.observation_space,
                'num_options': self.num_options,
                'net': self.net.state_dict(),
                'net_class': self.net.__class__,
        }
    def load_state_dict_deploy(self, state):
        self.net.load_state_dict(state['net'])

def make_agent_from_deploy_state(state : Union[str,Mapping], device : torch.device = torch.device('cpu')) -> OptionCriticAgent:
    if isinstance(state, str): # If it's a string, then it's the filename to the dilled state
        filename = state
        if not os.path.isfile(filename):
            raise Exception('No file found at %s' % filename)
        with open(filename, 'rb') as f:
            state = dill.load(f)
    if not isinstance(state,Mapping):
        raise ValueError('State is expected to be a dictionary. Found a %s.' % type(state))

    cls = state['net_class']
    if cls is OptionCriticNetworkCNN:
        net = OptionCriticNetworkCNN(state['action_space'].n, state['num_options']).to(device)
    else:
        raise Exception('Unable to initialize model of type %s' % cls)

    agent = OptionCriticAgent(
            action_space=state['action_space'],
            observation_space=state['observation_space'],
            num_options=state['num_options'],
            net = net,
    )
    agent.load_state_dict_deploy(state)
    return agent

def run_atari():
    from rl.agent.option_critic import OptionCriticAgent # XXX: Doesn't work unless I import it? Why?
    #from rl.agent.option_critic import OptionCriticAgentDebug1 as OptionCriticAgent
    from rl.experiments.training.basic import TrainExperiment
    from experiment import make_experiment_runner

    #import pprint#
    #pprint.pprint(dict(os.environ))

    params_pong = {
        'discount_factor': 0.99,
        'behaviour_eps': 0.02,
        'learning_rate': 1e-4,
        'batch_size': 32,
        'update_frequency': 1,
        'target_update_frequency': 200,
        'polyak_rate': 1,
        'num_options': 1,
        'entropy_reg': 0.01,
    }

    #params_dqn = {
    #    # Use default parameters
    #    'atari': True,
    #}

    #params_option_critic = { # Jean Harb's parameters
    #    'discount_factor': 0.99,
    #    'learning_rate': 2.5e-4,
    #    'target_update_frequency': 10_000,
    #    'polyak_rate': 1,
    #    'warmup_steps': 50_000,
    #    'replay_buffer_size': 1_000_000,
    #    'atari': True,
    #    'num_options': 1, # XXX: DEBUG
    #    'termination_reg': 0.01,
    #    'entropy_reg': 1e-5,
    #}

    params_debug = {
            **params_pong,
    }

    #env_name = 'PongNoFrameskip-v4'
    env_name = 'ALE/Pong-v5'
    #env_name = 'SeaquestNoFrameskip-v4'
    #env_name = 'MsPacmanNoFrameskip-v4'
    env_config = {
            'atari': True,
            'config': {
                'frameskip': 1,
                'mode': 0,
                'difficulty': 0,
                'repeat_action_probability': 0.25,
                #'render_mode': 'human',
            }
    }
    num_actors = 16
    train_env_keys = list(range(num_actors))
    debug = False

    if debug:
        exp_runner = make_experiment_runner(
                TrainExperiment,
                config={
                    'agent': {
                        'type': OptionCriticAgent,
                        'parameters': params_debug,
                    },
                    'env_test': {'env_name': env_name, **env_config},
                    'env_train': {'env_name': env_name, **env_config},
                    'train_env_keys': train_env_keys,
                    'verbose': True,
                    'test_frequency': None,
                    'save_model_frequency': None,
                },
                #trial_id='checkpointtest',
                checkpoint_frequency=None,
                max_iterations=5_000_000,
                verbose=True,
        )
    else:
        exp_runner = make_experiment_runner(
                TrainExperiment,
                config={
                    'agent': {
                        'type': OptionCriticAgent,
                        'parameters': params_pong,
                    },
                    'env_test': {'env_name': env_name, **env_config},
                    'env_train': {'env_name': env_name, **env_config},
                    'train_env_keys': train_env_keys,
                    'save_model_frequency': 250_000,
                    'verbose': True,
                    'test_frequency': None,
                },
                #trial_id='checkpointtes128t',
                checkpoint_frequency=250_000,
                max_iterations=50_000_000,
                #slurm_split=True,
                verbose=True,
        )
        exp_runner.exp.logger.init_wandb({
            'project': 'OptionCritic-%s' % env_name.replace('/','_') # W&B doesn't support project names with the '/' character
        })
    exp_runner.run()

def run_discrete():
    from rl.agent.option_critic import OptionCriticAgent # XXX: Doesn't work unless I import it? Why?
    #from rl.agent.option_critic import OptionCriticAgentDebug1 as OptionCriticAgent
    from rl.experiments.training.basic import TrainExperiment
    from experiment import make_experiment_runner

    params = {
        'discount_factor': 0.99,
        'behaviour_eps': 0.02,
        'learning_rate': 0.01,
        'update_frequency': 1,
        'target_update_frequency': 1,
        'polyak_rate': 1,
        'replay_buffer_size': 1, # TODO: Remove
        'atari': False,
        #'num_options': 6, # XXX: DEBUG
        'num_options': 1, # XXX: DEBUG
        'entropy_reg': 0.01, # XXX: DEBUG
    }

    env_name = 'FrozenLake-v1'
    num_actors = 16
    train_env_keys = list(range(num_actors))
    debug = False

    if debug:
        exp_runner = make_experiment_runner(
                TrainExperiment,
                config={
                    'agent': {
                        'type': OptionCriticAgent,
                        'parameters': params,
                    },
                    'env_test': {'env_name': env_name, 'atari': False},
                    'env_train': {'env_name': env_name, 'atari': False},
                    'train_env_keys': train_env_keys,
                    'verbose': True,
                    'test_frequency': None,
                    'save_model_frequency': None,
                },
                #trial_id='checkpointtest',
                checkpoint_frequency=None,
                max_iterations=5_000_000,
                verbose=True,
        )
    else:
        exp_runner = make_experiment_runner(
                TrainExperiment,
                config={
                    'agent': {
                        'type': OptionCriticAgent,
                        'parameters': params,
                    },
                    'env_test': {'env_name': env_name, 'atari': False},
                    'env_train': {'env_name': env_name, 'atari': False},
                    'train_env_keys': train_env_keys,
                    'verbose': True,
                    'test_frequency': None,
                    'save_model_frequency': None,
                },
                checkpoint_frequency=None,
                max_iterations=1_000_000,
                verbose=True,
        )
        #exp_runner.exp.logger.init_wandb({
        #    'project': 'OptionCritic-%s' % env_name
        #})
    exp_runner.run()

if __name__ == "__main__":
    run_atari()
    #run_discrete()
