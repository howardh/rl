from typing import Union, Mapping
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

from frankenstein.loss.policy_gradient import advantage_policy_gradient_loss
from frankenstein.value.monte_carlo import monte_carlo_return_iterative
from frankenstein.buffer.history import HistoryBuffer
from rl.agent.agent import DeployableAgent
from rl.agent.smdp.a2c import compute_entropy

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

def compute_termination_loss(
        termination_prob,
        option_values_current,
        option_values_max,
        termination_reg,
        deliberation_cost):
    """
    Option termination gradient.

    Given a sequence of length n, we observe states/options/actions/rewards r_0,s_0,o_0,a_0,r_1,s_1,o_1,a_1,r_2,s_2,...,r_{n-1},s_{n-1},o_{n-1},a_{n-1}.

    Args:
        termination_prob: A list of n elements. The element at index i is a ???-dimensional tensor containing the predicted probability of terminating option o_{i-1} at state s_i.
        option_values_current: A list of n elements. The element at index i is the value of option o_{i-1} at state s_i.
        option_values_max: A list of n elements. The element at index i is the largest option value over all options at state s_i.
        termination_reg (float): A regularization constant to ensure that termination probabilities do not all converge on 1. Value must be greater than 0.
    """
    advantage = option_values_current-option_values_max+termination_reg
    advantage = advantage.detach()
    loss = (termination_prob*(advantage+deliberation_cost)).mean(0)
    return loss

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
            deliberation_cost : float = 0, # The paper studied a range of values between 0 and 0.03
            optimizer : str = 'adam', # adam or rmsprop
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
        self.deliberation_cost = deliberation_cost

        # State (training)
        self.train_history_buffer = HistoryBuffer(max_len=batch_size, default_action=(0,0), device=device)
        self.test_history_buffer = defaultdict(
            lambda: HistoryBuffer(max_len=1, default_action=(0,0), device=device)
        )
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

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate)
        else:
            raise Exception(f'Unsupported optimizer: "{optimizer}". Use "adam" or "rmsprop".')

        self._train_env_keys = set()
        self._num_training_transitions = 0

        # Logging
        if logger is None:
            self.logger = Logger(key_name='step', allow_implicit_key=True)
            self.logger.log(step=0)
        else:
            self.logger = logger
        self._option_choice_count = defaultdict(lambda: [0]*num_options)
        self._option_choice_history = defaultdict(lambda: [])
        self._option_term_history = defaultdict(lambda: [])

    def observe(self, obs, reward=None, terminal=False, testing=False, env_key=None):
        if env_key is None:
            env_key = testing
        if reward is None: # Reset logging stuff
            self._option_choice_count[env_key] = [0]*self.num_options
            self._option_term_history[env_key] = []
            self._option_choice_history[env_key] = []
            self._current_option[env_key] = None
        if not testing:
            # Count the number of transitions available for training
            self._train_env_keys.add(env_key)
            history = self.train_history_buffer[int(env_key)]
            if len(history.obs_history) != 0:
                self._num_training_transitions += 1
        else:
            # Make sure we're not reusing the same key for testing and training
            if env_key in self._train_env_keys:
                raise Exception(f'Environment key "{env_key}" was previously used in training. The same key cannot be reused for testing.')
            history = self.test_history_buffer[env_key]

        # Reward clipping
        if reward is not None:
            reward = np.clip(reward, -1, 1)

        history.append_obs(obs, reward, terminal)

        # Add to replay buffer
        if not testing:
            self._steps += 1
            if not isinstance(self.logger, SubLogger):
                self.logger.log(step=self._steps)

        if terminal:
            # Logging
            counts = self._option_choice_count[env_key]
            total = sum(counts)
            probs = [c/total for c in counts]
            entropy = sum([-p*np.log(p) for p in probs if p != 0])
            self.logger.append(option_choice_entropy=entropy)
            if not testing:
                self.logger.log(option_choice_count=self._option_choice_count[env_key])

        # Train if appropriate
        if self._num_training_transitions >= self.batch_size:
            self._train()
    def _train(self):
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
        self.train_history_buffer.clear()
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
        history = self.train_history_buffer[int(env_key)]
        n = len(history.obs_history)
        target_eps = self.eps[True]

        obs = history.obs.float()/self.obs_scale
        num_obs = obs.shape[0]
        num_actions = num_obs-1
        batch0 = range(num_actions)
        batch1 = range(1,num_actions+1)
        option,action = history.action
        if history.terminal[-1]:
            option = option[:-1]
            action = action[:-1]

        net_output = self.net(obs)
        net_output_target = self.net_target(obs)
        q = net_output['q']
        qt = net_output_target['q']

        #target_state_values_max = qt.max(1)[0]
        beta1 = net_output['beta'][batch1,option]
        last_beta = net_output['beta'][-1,option[-1]] # Probability of terminating option n-2 at state n-1. Note that there is no distinct option associated with the last state. It is not a mistake that the indices are misaligned.
        target_state_value_last = (1-last_beta) * qt[-1,option[-1]] \
                                +    last_beta  * (
                                        ((1-target_eps) * qt[-1,:].max()
                                         +  target_eps  * qt[-1,:].mean())
                                        -self.deliberation_cost
                                )
        target_state_values = torch.cat([
            net_output_target['q'][batch0,option],
            target_state_value_last.unsqueeze(0) # Compute as an expectation because we don't know yet if the option will be terminated
        ])
        if isinstance(self.action_space,gym.spaces.Discrete):
            log_action_probs = net_output['iop'].log_softmax(2)[batch0,option,action]
            entropy = [
                    compute_entropy(iop.log_softmax(1))
                    for iop in net_output['iop']
            ]
        else:
            raise NotImplementedError()
        # Policy loss
        loss_policy = advantage_policy_gradient_loss(
                log_action_probs = log_action_probs[:n-1],
                state_values = target_state_values[:n-1].detach(),
                next_state_values = target_state_values[1:].detach(),
                rewards = history.reward[1:],
                terminals = history.terminal[1:],
                prev_terminals = history.terminal[:n-1],
                discounts=torch.tensor([self.discount_factor]*(n-1)),
        )
        # Entropy loss
        loss_entropy = -self.entropy_reg*torch.stack(entropy).mean()
        # Termination loss
        loss_term = compute_termination_loss(
                termination_prob = beta1,
                option_values_current = net_output['q'][batch1,option],
                option_values_max = (1-target_eps)*net_output['q'][batch1,:].max(1)[0]+target_eps*net_output['q'][batch1,:].mean(1),
                termination_reg = self.termination_reg,
                deliberation_cost = self.deliberation_cost,
        )
        # Q loss
        state_value_estimates = monte_carlo_return_iterative(
                state_values = target_state_values[1:],
                rewards = history.reward[1:],
                terminals = history.terminal[1:],
                discounts=torch.tensor([self.discount_factor]*(n-1)),
        )
        loss_q = (q[batch0,option]-state_value_estimates)**2
        # Output
        return {
                'policy': loss_policy,
                'entropy': loss_entropy,
                'termination': loss_term,
                'q': loss_q,
        }
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
            history = self.test_history_buffer[env_key][0]
        else:
            eps = self._compute_annealed_epsilon(self.eps_annealing_steps)
            history = self.train_history_buffer[int(env_key)]

        obs = history.obs_history[-1]
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
            self._option_term_history[env_key].append(True)
        else:
            option = curr_option
            self._option_term_history[env_key].append(False)
        # Choose an action
        action_probs = pi['iop'][0,option,:].softmax(dim=0)
        dist = torch.distributions.Categorical(probs=action_probs)
        action = dist.sample().item()

        history.append_action((option,action))

        # Save state and logging
        if curr_option is None or terminate_option:
            if env_key in self._current_option_duration:
                if testing:
                    self.logger.append(
                            testing_option_duration=self._current_option_duration[env_key],
                            testing_option_choice=option,
                            testing_all_option_values=pi['q'].squeeze().tolist(),
                    )
                else:
                    self.logger.append(
                            training_option_duration=self._current_option_duration[env_key],
                            training_option_choice=option,
                            training_all_option_values=pi['q'].squeeze().tolist(),
                    )
            self._current_option[env_key] = option
            self._current_option_duration[env_key] = 1
        else:
            self._current_option_duration[env_key] += 1

        assert isinstance(option,int)
        self._option_choice_count[env_key][option] += 1
        self._option_choice_history[env_key].append(option)

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
                'net_target': self.net_target.state_dict(),
                'optimizer': self.optimizer.state_dict(),

                'train_history_buffer': self.train_history_buffer,
                'test_history_buffer': self.test_history_buffer,
                '_steps': self._steps,
                '_training_steps': self._training_steps,
                '_current_option': self._current_option,
                '_train_env_keys': self._train_env_keys,
                '_num_training_transitions': self._num_training_transitions,

                '_option_choice_count': self._option_choice_count,
                '_current_option_duration': self._current_option_duration,
                'logger': self.logger.state_dict(),
        }
    def load_state_dict(self, state):
        self.net.load_state_dict(state['net'])
        self.net_target.load_state_dict(state['net_target'])
        self.optimizer.load_state_dict(state['optimizer'])

        self.train_history_buffer = state['train_history_buffer']
        self.test_history_buffer = state['test_history_buffer']
        self._steps = state['_steps']
        self._training_steps = state['_training_steps']
        self._current_option = state['_current_option']
        self._train_env_keys = state['_train_env_keys']
        self._num_training_transitions = state['_num_training_transitions']

        self._option_choice_count = state['_option_choice_count']
        self._current_option_duration = state['_current_option_duration']
        self.logger.load_state_dict(state['logger'])

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
        'optimizer': 'rmsprop',
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
