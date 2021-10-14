import warnings
from typing import Optional, Tuple, Generic, TypeVar
from collections import defaultdict
import copy

import gym
import gym.spaces
import torch
import torch.nn
import torch.utils.data
import torch.distributions
import torch.optim
import numpy as np

from experiment.logger import Logger, SubLogger

from rl.agent.agent import DeployableAgent
from rl.agent.replay_buffer import ReplayBuffer, AtariReplayBuffer
import rl.debug_tools.pong
from rl.agent.smdp.a2c import compute_advantage_policy_gradient

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
    def forward(self, obs):
        x = obs
        x = self.head(x)
        x = {
            'beta': self.beta(x),
            'iop': self.iop(x),
            'poo': self.poo(x),
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

class OptionCriticAgent(DeployableAgent):
    """
    Implementation of Option-Critic
    """
    def __init__(self,
            action_space : gym.spaces.Discrete,
            observation_space : gym.spaces.Space,
            discount_factor : float = 0.99,
            learning_rate : float = 2.5e-4,
            update_frequency : int = 4,
            batch_size : int = 32,
            warmup_steps : int = 50_000,
            replay_buffer_size : int = 1_000_000,
            target_update_frequency : int = 100,
            polyak_rate : float = 1.,
            device = torch.device('cpu'),
            behaviour_eps : float = 0.1,
            target_eps : float = 0.05,
            eps_annealing_steps : int = 1_000_000,
            num_options : int = 8,
            termination_reg : float = 0.01, # From paper
            entropy_reg : float = 0, # XXX: Arbitrary value
            #q_net : torch.nn.Module = None
            logger : Logger = None,
            atari : bool = False,
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
        self.warmup_steps = warmup_steps
        self.target_update_frequency = target_update_frequency
        self.polyak_rate = polyak_rate
        self.device = device
        self.eps = [behaviour_eps, target_eps]
        self.eps_annealing_steps = eps_annealing_steps
        self.num_options = num_options
        self.termination_reg = termination_reg
        self.entropy_reg = entropy_reg

        # State (training)
        self.obs_stack = defaultdict(lambda: ObservationStack())
        self._steps = 0 # Number of steps experienced
        self._training_steps = 0 # Number of training steps experienced
        self._current_option = {} # Option that is currently executing
        self._current_option_duration = {} # How long the current option has been executing

        if atari:
            self.replay_buffer = AtariReplayBuffer(replay_buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.q_net = QNetworkCNN(
                num_actions=num_options
        ).to(self.device)
        self.q_net_target = copy.deepcopy(self.q_net)
        #self.q_net_target = QNetworkCNN(
        #        num_actions=num_options
        #).to(self.device)
        self.policy_net = OptionCriticNetworkCNN( # Termination, intra-option policy, and policy over options
                num_actions=self.action_space.n,
                num_options=num_options
        ).to(self.device)

        self.optimizer_q = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.optimizer_policy_distillation = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        #self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=learning_rate, momentum=0.95)

        if logger is None:
            self.logger = Logger(key_name='step', allow_implicit_key=True)
            self.logger.log(step=0)
        else:
            self.logger = logger

        # XXX: DEBUG
        self._pretrained_q_net = rl.debug_tools.pong.get_pretained_model()
        self._pretrained_q_net.to(self.device)

    def observe(self, obs, reward=None, terminal=False, testing=False, env_key=None):
        if env_key is None:
            env_key = testing

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
            
            self.replay_buffer.add_transition(*transition)

            # Train each time something is added to the buffer
            self._train()
    def _train_old(self, iterations=1):
        if __debug__:
            raise Exception('THIS FUNCTION IS DEPRECATED. DELETE ONCE EVERYTHING ELSE IS WORKING')
        if len(self.replay_buffer) < self.batch_size*iterations:
            return
        if len(self.replay_buffer) < self.warmup_steps:
            return
        if self._steps % self.update_frequency != 0:
            return
        dataloader = torch.utils.data.DataLoader(
                self.replay_buffer, batch_size=self.batch_size, shuffle=True)
        criterion = torch.nn.MSELoss(reduction='mean')
        for _,(s0,(o0,a0),r1,s1,term,time,gamma) in zip(range(iterations),dataloader):
            # Fix data types
            batch_size = s0.shape[0]
            s0 = s0.to(self.device).float()/self.obs_scale
            o0 = o0.to(self.device).unsqueeze(1)
            a0 = a0.to(self.device)
            r1 = r1.float().to(self.device).unsqueeze(1)
            s1 = s1.to(self.device).float()/self.obs_scale
            term = term.float().to(self.device).unsqueeze(1)
            time = time.to(self.device).unsqueeze(1)
            gamma = gamma.float().to(self.device).unsqueeze(1)
            g_t = torch.pow(gamma,time)
            if isinstance(self.observation_space,gym.spaces.Discrete):
                s0 = s0.unsqueeze(1)
                s1 = s1.unsqueeze(1)
            if isinstance(self.action_space,gym.spaces.Discrete):
                a0 = a0.unsqueeze(1)

            termination_reg = 0.01 # From paper
            entropy_reg = 1e-3 # XXX: arbitrary value

            policy0 = self.policy_net(s0)
            policy1 = self.policy_net(s1)
            q_target = self.q_net_target(s1)
            q0 = self.q_net(s0)
            q1 = self.q_net(s1)
            #beta0 = policy0['beta'].gather(1,o0)
            beta1 = policy1['beta'].gather(1,o0) # We use the same option on the next step

            # Value estimate (target)
            optimal_option_values = q_target.max(1)[0].unsqueeze(1)
            current_option_values = q_target.gather(1,o0)
            y = r1+(1-term)*g_t*(
                    (1-beta1)*current_option_values+
                    beta1*optimal_option_values
            )
            y = y.detach()

            # Value estimate (prediction)
            y_pred = self.q_net(s0).gather(1,o0)
    
            assert y.shape == y_pred.shape

            # Termination
            q_current_option = q1.gather(1,o0)
            v = q1.max(1)[0].unsqueeze(1)
            advantage = (q_current_option-v+termination_reg).detach()
            loss_term = (beta1*advantage).mean(0)

            # Policy
            action_log_probs = policy0['iop'][range(batch_size),o0.flatten(),:].log_softmax(1)
            action_probs = policy0['iop'][range(batch_size),o0.flatten(),:].softmax(1)
            entropy = -(action_probs*action_log_probs).sum(1)
            advantage = (y-q0.gather(1,o0)).detach()
            loss_policy = (action_log_probs.gather(1,a0)*advantage).mean(0)
            loss_entropy = -entropy_reg*entropy.mean(0)

            # Update Q_U network
            self.optimizer_q.zero_grad()
            loss_q = criterion(y_pred,y)
            loss_q.backward()
            #for param in self.qu_net.parameters():
            #    assert param.grad is not None
            #    param.grad.data.clamp_(-1, 1)
            self.optimizer_q.step()

            # Update policy network
            loss = loss_term+loss_policy+loss_entropy
            self.optimizer_policy.zero_grad()
            loss.backward()
            self.optimizer_policy.step()

            # Logging
            self.logger.log(
                    #step=self._steps, # XXX: Not working without this, even with implicit key. Look into this.
                    training_batch_option_value_target=current_option_values.mean().item(),
                    training_batch_option_value=y_pred.mean().item(),
                    policy_term_ent_loss=loss.item(),
                    q_loss=loss_q.item(),
                    termination_loss=loss_term.item(),
                    policy_loss=loss_policy.item(),
                    entropy_loss=loss_entropy.item(),
            )

        self._training_steps += 1

        self._update_target()
    def _train(self):
        if len(self.replay_buffer) < self.warmup_steps:
            return
        self._train_actor()
        if self._steps % self.update_frequency == 0:
            self._train_critic()
            self._training_steps += 1
    def _train_critic(self, iterations=1):
        self._train_critic_1(iterations)
    def _train_critic_1(self, iterations=1):
        if len(self.replay_buffer) < self.batch_size*iterations:
            return
        dataloader = torch.utils.data.DataLoader(
                self.replay_buffer, batch_size=self.batch_size, shuffle=True)
        criterion = torch.nn.MSELoss(reduction='mean')
        for _,(s0,(o0,a0),r1,s1,term) in zip(range(iterations),dataloader):
            # Fix data types
            s0 = s0.to(self.device).float()/self.obs_scale
            o0 = o0.to(self.device).unsqueeze(1)
            a0 = a0.to(self.device)
            r1 = r1.float().to(self.device).unsqueeze(1)
            s1 = s1.to(self.device).float()/self.obs_scale
            term = term.float().to(self.device).unsqueeze(1)
            gamma = self.discount_factor
            if isinstance(self.observation_space,gym.spaces.Discrete):
                s0 = s0.unsqueeze(1)
                s1 = s1.unsqueeze(1)
            if isinstance(self.action_space,gym.spaces.Discrete):
                a0 = a0.unsqueeze(1)

            policy1 = self.policy_net(s1)
            q_target = self.q_net_target(s1)
            #beta0 = policy0['beta'].gather(1,o0)
            beta1 = policy1['beta'].gather(1,o0) # We use the same option on the next step

            # Value estimate (target)
            optimal_option_values = q_target.max(1)[0].unsqueeze(1)
            current_option_values = q_target.gather(1,o0)
            y = r1+(1-term)*gamma*(
                    (1-beta1)*current_option_values+
                    beta1*optimal_option_values
            )
            y = y.detach()

            # Value estimate (prediction)
            y_pred = self.q_net(s0).gather(1,o0)
    
            assert y.shape == y_pred.shape

            # Update Q_U network
            self.optimizer_q.zero_grad()
            loss_q = criterion(y_pred,y)
            loss_q.backward()
            #for param in self.qu_net.parameters():
            #    assert param.grad is not None
            #    param.grad.data.clamp_(-1, 1)
            self.optimizer_q.step()

            # Logging
            self.logger.log(
                    training_batch_option_value_target=current_option_values.mean().item(),
                    training_batch_option_value=y_pred.mean().item(),
                    q_loss=loss_q.item(),
            )

        self._update_target()
    def _train_critic_2(self, iterations=1):
        if len(self.replay_buffer) < self.batch_size*iterations:
            return
        dataloader = torch.utils.data.DataLoader(
                self.replay_buffer, batch_size=self.batch_size, shuffle=True)
        criterion = torch.nn.MSELoss(reduction='mean')
        for _,(s0,(o0,a0),r1,s1,term) in zip(range(iterations),dataloader):
            # Fix data types
            s0 = s0.to(self.device).float()/self.obs_scale
            o0 = o0.to(self.device).unsqueeze(1)
            a0 = a0.to(self.device)
            r1 = r1.float().to(self.device).unsqueeze(1)
            s1 = s1.to(self.device).float()/self.obs_scale
            term = term.float().to(self.device).unsqueeze(1)
            #gamma = self.discount_factor
            if isinstance(self.observation_space,gym.spaces.Discrete):
                s0 = s0.unsqueeze(1)
                s1 = s1.unsqueeze(1)
            if isinstance(self.action_space,gym.spaces.Discrete):
                a0 = a0.unsqueeze(1)

            policy0 = self.policy_net(s0)

            # Value estimate (target)
            y = self._pretrained_q_net(s0)
            y = policy0['iop'].softmax(2) @ y.unsqueeze(2)
            y = y.squeeze(2)

            # Value estimate (prediction)
            y_pred = self.q_net(s0)
    
            assert y.shape == y_pred.shape

            # Update Q_U network
            self.optimizer_q.zero_grad()
            loss_q = criterion(y_pred,y)
            loss_q.backward()
            #for param in self.qu_net.parameters():
            #    assert param.grad is not None
            #    param.grad.data.clamp_(-1, 1)
            self.optimizer_q.step()

            # Logging
            self.logger.log(
                    training_batch_option_value=y_pred.mean().item(),
                    q_loss=loss_q.item(),
            )

        self._update_target()
    def _train_actor(self):
        #self._train_actor_6()
        if self._steps <= 300_000:
            self._train_actor_5()
        else:
            self._train_actor_6()
    def _train_actor_1(self):
        transition = self.obs_stack[False].get_transition() # FIXME: The environment key is hard-coded.
        if transition is None:
            return
        s0,(o0,a0),r1,s1,term = transition
        s0 = torch.tensor(s0).to(self.device).float().unsqueeze(0)/self.obs_scale
        s1 = torch.tensor(s1).to(self.device).float().unsqueeze(0)/self.obs_scale

        gamma = self.discount_factor
        termination_reg = self.termination_reg
        entropy_reg = self.entropy_reg

        policy0 = self.policy_net(s0)
        policy1 = self.policy_net(s1)
        q_target = self.q_net_target(s1).squeeze(0)
        q0 = self.q_net(s0).squeeze(0)
        q1 = self.q_net(s1).squeeze(0)
        #beta0 = policy0['beta'].squeeze(0)[o0]
        beta1 = policy1['beta'].squeeze(0)[o0] # We use the same option on the next step

        # Termination
        q_current_option = q1[o0]
        v = q1.max()
        advantage = (q_current_option-v+termination_reg).detach()
        loss_term = (beta1*advantage).mean(0)

        # Value estimate (target)
        optimal_option_values = q_target.max()
        current_option_values = q_target[o0]
        y = r1+(1-term)*gamma*(
                (1-beta1)*current_option_values+
                beta1*optimal_option_values
        )
        y = y.detach()

        # Policy
        action_log_probs = policy0['iop'].squeeze(0)[o0,:].log_softmax(0)
        action_probs = policy0['iop'].squeeze(0)[o0,:].softmax(0)
        entropy = -(action_probs*action_log_probs).sum(0)
        advantage = (y-q0[o0]).detach()
        loss_policy = (action_log_probs[a0]*advantage).mean(0)
        loss_entropy = -entropy_reg*entropy

        # Update policy network
        loss = loss_term+loss_policy+loss_entropy
        self.optimizer_policy.zero_grad()
        loss.backward()
        self.optimizer_policy.step()

        # Logging
        self.logger.log(
                policy_term_ent_loss=loss.item(),
                termination_loss=loss_term.item(),
                policy_loss=loss_policy.item(),
                entropy_loss=loss_entropy.item(),
        )
    def _train_actor_2(self):
        """ Copied from https://github.com/lweitkamp/option-critic-pytorch/blob/0c57da7686f8903ed2d8dded3fae832ee9defd1a/option_critic.py#L222 """
        transition = self.obs_stack[False].get_transition() # FIXME: The environment key is hard-coded.
        if transition is None:
            return
        s0,(o0,a0),r1,s1,term = transition
        s0 = torch.tensor(s0).to(self.device).float().unsqueeze(0)/self.obs_scale
        s1 = torch.tensor(s1).to(self.device).float().unsqueeze(0)/self.obs_scale

        gamma = self.discount_factor
        termination_reg = self.termination_reg
        entropy_reg = self.entropy_reg

        policy0 = self.policy_net(s0)
        policy1 = self.policy_net(s1)
        q_target = self.q_net_target(s1).squeeze(0)
        q = self.q_net(s0).squeeze(0)
        beta0 = policy0['beta'].squeeze(0)[o0]
        beta1 = policy1['beta'].squeeze(0)[o0] # We use the same option on the next step

        # Termination
        q_current_option = q[o0]
        v = q_target.max()
        advantage = (q_current_option-v+termination_reg).detach()
        loss_term = (beta0*advantage).mean(0)

        # Value estimate (target)
        optimal_option_values = q_target.max()
        current_option_values = q_target[o0]
        y = r1+(1-term)*gamma*(
                (1-beta1)*current_option_values+
                beta1*optimal_option_values
        )
        y = y.detach()

        # Policy
        action_log_probs = policy0['iop'].squeeze(0)[o0,:].log_softmax(0)
        action_probs = policy0['iop'].squeeze(0)[o0,:].softmax(0)
        entropy = -(action_probs*action_log_probs).sum(0)
        advantage = (y-q[o0]).detach()
        loss_policy = -(action_log_probs[a0]*advantage).mean(0)
        loss_entropy = -entropy_reg*entropy

        if loss_policy == 0:
            breakpoint()

        # Update policy network
        #loss = loss_term+loss_policy+loss_entropy
        loss = loss_policy+loss_entropy
        self.optimizer_policy.zero_grad()
        loss.backward()
        self.optimizer_policy.step()

        # Logging
        self.logger.log(
                policy_term_ent_loss=loss.item(),
                termination_loss=loss_term.item(),
                policy_loss=loss_policy.item(),
                entropy_loss=loss_entropy.item(),
        )
    def _train_actor_3(self):
        """ Advantage uses the same Q function for both the bootstrapped sample and the value estimate. """
        transition = self.obs_stack[False].get_transition() # FIXME: The environment key is hard-coded.
        if transition is None:
            return
        s0,(o0,a0),r1,s1,term = transition
        s0 = torch.tensor(s0).to(self.device).float().unsqueeze(0)/self.obs_scale
        s1 = torch.tensor(s1).to(self.device).float().unsqueeze(0)/self.obs_scale

        gamma = self.discount_factor
        termination_reg = self.termination_reg
        entropy_reg = self.entropy_reg

        policy0 = self.policy_net(s0)
        policy1 = self.policy_net(s1)
        q1_target = self.q_net_target(s1).squeeze(0)
        q1 = self.q_net(s1).squeeze(0)
        q0 = self.q_net(s0).squeeze(0)
        beta0 = policy0['beta'].squeeze(0)[o0]
        beta1 = policy1['beta'].squeeze(0)[o0] # We use the same option on the next step

        # Termination
        q_current_option = q0[o0]
        v = q1_target.max()
        advantage = (q_current_option-v+termination_reg).detach()
        loss_term = (beta0*advantage).mean(0)

        # Value estimate (target)
        optimal_option_values = q1.max()
        current_option_values = q1[o0]
        y = r1+(1-term)*gamma*(
                (1-beta1)*current_option_values+
                beta1*optimal_option_values
        )
        y = y.detach()

        # Policy
        action_log_probs = policy0['iop'].squeeze(0)[o0,:].log_softmax(0)
        action_probs = policy0['iop'].squeeze(0)[o0,:].softmax(0)
        entropy = -(action_probs*action_log_probs).sum(0)
        advantage = (y-q0[o0]).detach()
        loss_policy = -(action_log_probs[a0]*advantage).mean(0)
        loss_entropy = -entropy_reg*entropy

        # Update policy network
        #loss = loss_term+loss_policy+loss_entropy
        loss = loss_policy+loss_entropy
        self.optimizer_policy.zero_grad()
        loss.backward()
        self.optimizer_policy.step()

        # Logging
        self.logger.log(
                policy_term_ent_loss=loss.item(),
                termination_loss=loss_term.item(),
                policy_loss=loss_policy.item(),
                entropy_loss=loss_entropy.item(),
        )
    def _train_actor_4(self):
        """ Train the actor using distillation on the pretrained DQN. """
        transition = self.obs_stack[False].get_transition() # FIXME: The environment key is hard-coded.
        if transition is None:
            return
        s0,(o0,a0),r1,s1,term = transition
        s0 = torch.tensor(s0).to(self.device).float().unsqueeze(0)/self.obs_scale
        s1 = torch.tensor(s1).to(self.device).float().unsqueeze(0)/self.obs_scale
        a0,r1,term=a0,r1,term

        policy0 = self.policy_net(s0)

        # Policy
        criterion = torch.nn.KLDivLoss(reduction='mean')
        action_log_probs = policy0['iop'][:,o0,:].log_softmax(1)
        target_log_probs = self._pretrained_q_net(s0).log_softmax(1)
        loss_policy = criterion(action_log_probs,target_log_probs)

        # Update policy network
        loss = loss_policy
        self.optimizer_policy.zero_grad()
        loss.backward()
        self.optimizer_policy.step()

        # Logging
        self.logger.log(
                policy_term_ent_loss=loss.item(),
                policy_loss=loss_policy.item(),
        )
    def _train_actor_5(self):
        """ Train the actor using distillation on the pretrained DQN. Use data from the replay buffer. """
        iterations = 1
        if len(self.replay_buffer) < self.batch_size*iterations:
            return
        if len(self.replay_buffer) < self.warmup_steps:
            return
        if self._steps % self.update_frequency != 0:
            return
        dataloader = torch.utils.data.DataLoader(
                self.replay_buffer, batch_size=self.batch_size, shuffle=True)
        criterion = torch.nn.KLDivLoss(reduction='mean', log_target=True)
        for _,(s0,(o0,a0),r1,s1,term) in zip(range(iterations),dataloader):
            s0 = s0.to(self.device).float()/self.obs_scale
            #s0 = torch.tensor(s0).to(self.device).float().unsqueeze(0)/self.obs_scale
            #s1 = torch.tensor(s1).to(self.device).float().unsqueeze(0)/self.obs_scale
            a0,r1,s1,term = a0,r1,s1,term

            policy0 = self.policy_net(s0)

            # Policy
            action_log_probs = policy0['iop'][range(len(o0)),o0,:].log_softmax(1)
            target_log_probs = (self._pretrained_q_net(s0)*300).log_softmax(1)
            loss_policy = criterion(action_log_probs,target_log_probs)

            # Update policy network
            loss = loss_policy
            self.optimizer_policy_distillation.zero_grad()
            loss.backward()
            self.optimizer_policy_distillation.step()

            # Logging
            self.logger.log(
                    training_batch_policy_term_ent_loss=loss.item(),
                    training_batch_policy_loss=loss_policy.item(),
            )
    def _train_actor_6(self):
        """ Using A2C policy gradient implementation. Termination is ignored. """
        transition = self.obs_stack[False].get_transition() # FIXME: The environment key is hard-coded.
        if transition is None:
            return
        s0,(o0,a0),r1,s1,term = transition
        s0 = torch.tensor(s0).to(self.device).float().unsqueeze(0)/self.obs_scale
        s1 = torch.tensor(s1).to(self.device).float().unsqueeze(0)/self.obs_scale

        policy0 = self.policy_net(s0)
        #policy1 = self.policy_net(s1)
        q0_target = self.q_net_target(s0).squeeze(0)
        q1_target = self.q_net_target(s1).squeeze(0)

        # Policy
        log_action_probs = [policy0['iop'].squeeze(0)[o0,:].log_softmax(0)[a0],None]
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
        # Entropy
        action_probs = policy0['iop'].squeeze(0)[o0,:].softmax(0)
        entropy = -action_probs*torch.log(action_probs)
        loss_entropy = -0.01*entropy.sum() # Factor of 0.01 works in A2C

        # Update policy network
        #loss = loss_term+loss_policy+loss_entropy
        loss = loss_policy+loss_entropy
        self.optimizer_policy.zero_grad()
        loss.backward()
        self.optimizer_policy.step()

        # Logging
        self.logger.log(
                policy_term_ent_loss=loss.item(),
                #termination_loss=loss_term.item(),
                policy_loss=loss_policy.item(),
                entropy_loss=loss_entropy.item(),
        )
    def _update_target(self):
        if self._training_steps % self.target_update_frequency != 0:
            return
        tau = self.polyak_rate
        for p1,p2 in zip(self.q_net_target.parameters(), self.q_net.parameters()):
            p1.data = (1-tau)*p1+tau*p2
    def act(self, testing=False, env_key=None):
        if testing:
            return self._act_1(testing=testing,env_key=env_key)
        else:
            #return self._act_2(testing=testing,env_key=env_key)
            if self._steps <= 300_000:
                return self._act_2(testing=testing,env_key=env_key)
            else:
                return self._act_1(testing=testing,env_key=env_key)
    def _act_1(self, testing=False, env_key=None):
        """ Return a random action according to an epsilon-greedy policy. """
        if env_key is None:
            env_key = testing

        if testing:
            eps = self.eps[testing]
        else:
            eps = self._compute_annealed_epsilon(self.eps_annealing_steps)

        obs = self.obs_stack[env_key].get_obs()
        obs = torch.tensor(obs).to(self.device).float()/self.obs_scale
        if isinstance(self.observation_space,gym.spaces.Discrete):
            obs = obs.unsqueeze(0).unsqueeze(0)
        elif isinstance(self.observation_space,gym.spaces.Box):
            obs = obs.unsqueeze(0)
        else:
            raise NotImplementedError()

        curr_option = self._current_option.get(env_key)
        pi = self.policy_net(obs)
        # Choose an option
        terminate_option = pi['beta'][0,curr_option].item() > torch.rand(1).item() if curr_option is not None else False
        if curr_option is None or terminate_option:
            vals = self.q_net(obs)
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
                    self.logger.log(
                            training_option_duration=self._current_option_duration[env_key],
                            training_option_choice=option
                    )
            self._current_option[env_key] = option
            self._current_option_duration[env_key] = 1
        else:
            self._current_option_duration[env_key] += 1

        if testing:
            vals = self.q_net(obs)
            self.logger.append(
                    #step=self._steps,
                    #testing_action_value=vals[0,option,action].item(),
                    testing_option_value=vals[0,option].item(),
            )
        else:
            pq = self._pretrained_q_net(obs)
            self.logger.log(
                    #step=self._steps,
                    #action_value=self.q_net(obs)[0,option,action].item(),
                    entropy=dist.entropy().item(),
                    action_value_diff=(pq.max()-(action_probs.squeeze() * pq.squeeze()).sum()).item()
            )

        return action
    def _act_2(self, testing=False, env_key=None):
        """ Act according to the softmax policy with respect to the pretrained DQN. """
        if env_key is None:
            env_key = testing

        obs = self.obs_stack[env_key].get_obs()
        obs = torch.tensor(obs).to(self.device).float()/self.obs_scale
        if isinstance(self.observation_space,gym.spaces.Discrete):
            obs = obs.unsqueeze(0).unsqueeze(0)
        elif isinstance(self.observation_space,gym.spaces.Box):
            obs = obs.unsqueeze(0)
        else:
            raise NotImplementedError()

        q = self._pretrained_q_net(obs).squeeze()*300
        option = 0
        # Choose an action
        action_probs = q.softmax(dim=0)
        dist = torch.distributions.Categorical(probs=action_probs)
        action = dist.sample().item()

        self.obs_stack[env_key].append_action((option,action))

        # Save state and logging
        if not testing:
            pq = q
            self.logger.log(
                    #step=self._steps,
                    #action_value=self.q_net(obs)[0,option,action].item(),
                    entropy=dist.entropy().item(),
                    action_value_diff=(pq.max()-(action_probs.squeeze() * pq.squeeze()).sum()).item()
            )

        return action
    def _compute_annealed_epsilon(self, max_steps=1_000_000):
        eps = self.eps[False]
        return (1-eps)*max(1-self._steps/max_steps,0)+eps # Linear
        #return (1-eps)*np.exp(-self._steps/max_steps)+eps # Exponential

    def state_dict(self):
        return {
                'qu_net': self.q_net.state_dict(),
                'qu_net_target': self.q_net_target.state_dict(),
                'policy_net': self.policy_net.state_dict(),

                'optimizer_qu': self.optimizer_q.state_dict(),
                'optimizer_policy': self.optimizer_policy.state_dict(),

                'obs_stack': {k:os.state_dict() for k,os in self.obs_stack.items()},
                'steps': self._steps,
                'training_steps': self._training_steps,
                'logger': self.logger.state_dict(),
        }
    def load_state_dict(self, state):
        self.q_net.load_state_dict(state['q_net'])
        self.q_net_target.load_state_dict(state['q_net_target'])
        self.policy_net.load_state_dict(state['policy_net_target'])

        self.optimizer_q.load_state_dict(state['optimizer_q'])
        self.optimizer_policy.load_state_dict(state['optimizer_policy'])

        for k,os_state in state['obs_stack'].items():
            self.obs_stack[k].load_state_dict(os_state)
        self._steps = state['steps']
        self._training_steps = state['training_steps']
        self.logger.load_state_dict(state['logger'])

        warnings.warn('The replay buffer is not saved. Training the agent from this point may yield unexpected results.')

    def state_dict_deploy(self):
        if self._steps <= 500_000:
            breakpoint()
        return {
                'action_space': self.action_space,
                'observation_space': self.observation_space,
                'q_net': self.q_net.state_dict(),
                'q_net_class': self.q_net.__class__,
                'policy_net': self.policy_net.state_dict(),
                'policy_net_class': self.policy_net.__class__,
        }
    def load_state_dict_deploy(self, state):
        self.q_net.load_state_dict(state['qu_net'])
        self.policy_net.load_state_dict(state['policy_net'])

class OptionCriticAgentDebug1(OptionCriticAgent):
    """
    Implementation of Option-Critic
    Each option represents one primitive action
    """
    def _train(self, iterations=1):
        if len(self.replay_buffer) < self.batch_size*iterations:
            return
        if len(self.replay_buffer) < self.warmup_steps:
            return
        if self._steps % self.update_frequency != 0:
            return
        assert self.num_options == self.action_space.n
        dataloader = torch.utils.data.DataLoader(
                self.replay_buffer, batch_size=self.batch_size, shuffle=True)
        criterion = torch.nn.MSELoss(reduction='mean')
        for _,(s0,(o0,a0),r1,s1,term,time,gamma) in zip(range(iterations),dataloader):
            # Fix data types
            batch_size = s0.shape[0]
            s0 = s0.to(self.device).float()/self.obs_scale
            o0 = o0.to(self.device).unsqueeze(1)
            a0 = a0.to(self.device)
            r1 = r1.float().to(self.device).unsqueeze(1)
            s1 = s1.to(self.device).float()/self.obs_scale
            term = term.float().to(self.device).unsqueeze(1)
            time = time.to(self.device).unsqueeze(1)
            gamma = gamma.float().to(self.device).unsqueeze(1)
            g_t = torch.pow(gamma,time)
            if isinstance(self.observation_space,gym.spaces.Discrete):
                s0 = s0.unsqueeze(1)
                s1 = s1.unsqueeze(1)
            if isinstance(self.action_space,gym.spaces.Discrete):
                a0 = a0.unsqueeze(1)

            termination_reg = 0.01 # From paper
            entropy_reg = 0.1 # XXX: arbitrary value

            policy0 = self.policy_net(s0)
            policy1 = self.policy_net(s1)
            q_target = self.q_net_target(s1)
            #q0 = self.q_net(s0)
            q1 = self.q_net(s1)
            #beta0 = policy0['beta'].gather(1,o0)
            beta1 = policy1['beta'].gather(1,o0) # We use the same option on the next step

            # Value estimate (target)
            optimal_option_values = q_target.max(1)[0].unsqueeze(1)
            current_option_values = q_target.gather(1,o0)
            y = r1+(1-term)*g_t*(
                    (1-beta1)*current_option_values+
                    beta1*optimal_option_values
            )
            y = y.detach()

            # Value estimate (prediction)
            y_pred = self.q_net(s0).gather(1,o0)
    
            assert y.shape == y_pred.shape

            # Termination
            q_current_option = q1.gather(1,o0)
            v = q1.max(1)[0].unsqueeze(1)
            advantage = (q_current_option-v+termination_reg).detach()
            loss_term = (beta1*advantage).mean(0)

            # Policy
            bce = torch.nn.BCEWithLogitsLoss()
            logits = torch.cat([x for x in policy0['iop']])
            target = torch.cat([torch.eye(self.num_options) for _ in policy0['iop']]).to(self.device)
            loss_policy = bce(logits, target)
            action_log_probs = policy0['iop'][range(batch_size),o0.flatten(),:].log_softmax(1)
            action_probs = policy0['iop'][range(batch_size),o0.flatten(),:].softmax(1)
            entropy = -(action_probs*action_log_probs).sum(1)
            loss_entropy = (-entropy_reg*entropy).mean(0)
            if self._training_steps % 1000 == 0:
                print(logits.softmax(1))

            # Update Q_U network
            self.optimizer_q.zero_grad()
            loss_q = criterion(y_pred,y)
            loss_q.backward()
            #for param in self.qu_net.parameters():
            #    assert param.grad is not None
            #    param.grad.data.clamp_(-1, 1)
            self.optimizer_q.step()

            # Update policy network
            loss = loss_term+loss_policy+loss_entropy
            self.optimizer_policy.zero_grad()
            loss.backward()
            self.optimizer_policy.step()

            # Logging
            self.logger.log(
                    #step=self._steps, # XXX: Not working without this, even with implicit key. Look into this.
                    training_batch_option_value_target=current_option_values.mean().item(),
                    training_batch_option_value=y_pred.mean().item(),
                    policy_term_ent_loss=loss.item(),
                    q_loss=loss_q.item(),
                    termination_loss=loss_term.item(),
                    policy_loss=loss_policy.item(),
                    entropy_loss=loss_entropy.item(),
                    training_batch_entropy=entropy.mean().item(),
            )

        self._training_steps += 1

        self._update_target()

if __name__ == "__main__":
    import torch.cuda
    import gym

    from rl.agent.option_critic import OptionCriticAgent # XXX: Doesn't work unless I import it? Why?
    #from rl.agent.option_critic import OptionCriticAgentDebug1 as OptionCriticAgent
    from rl.experiments.training.basic import TrainExperiment
    from experiment import make_experiment_runner

    params_pong = {
        'discount_factor': 0.99,
        'behaviour_eps': 0.02,
        'learning_rate': 1e-7,
        'update_frequency': 1,
        'target_update_frequency': 1_000,
        'polyak_rate': 1,
        'warmup_steps': 10_000,
        'replay_buffer_size': 100_000,
        'atari': True,
        #'num_options': 6, # XXX: DEBUG
        'num_options': 1, # XXX: DEBUG
        'entropy_reg': 0, # XXX: DEBUG
    }

    params_dqn = {
        # Use default parameters
        'atari': True,
    }

    params_option_critic = { # Jean Harb's parameters
        'discount_factor': 0.99,
        'learning_rate': 2.5e-4,
        'target_update_frequency': 10_000,
        'polyak_rate': 1,
        'warmup_steps': 50_000,
        'replay_buffer_size': 1_000_000,
        'atari': True,
        'num_options': 1, # XXX: DEBUG
        'termination_reg': 0.01,
        'entropy_reg': 1e-5,
    }

    params_debug = {
            'discount_factor': 0.99,
            'behaviour_eps': 0.02,
            'learning_rate': 1e-4,
            'update_frequency': 1,
            'target_update_frequency': 1_000,
            'polyak_rate': 1,
            'warmup_steps': 100,
            'replay_buffer_size': 1000,
            'atari': True,
            'num_options': 6,
    }

    env_name = 'PongNoFrameskip-v4'
    #env_name = 'SeaquestNoFrameskip-v4'
    #env_name = 'MsPacmanNoFrameskip-v4'
    debug = False

    if debug:
        exp_runner = make_experiment_runner(
                TrainExperiment,
                config={
                    'agent': {
                        'type': OptionCriticAgent,
                        'parameters': params_debug,
                    },
                    'env_test': {'env_name': env_name, 'atari': True},
                    'env_train': {'env_name': env_name, 'atari': True},
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
                        #'parameters': params_option_critic,
                        'parameters': params_pong,
                        #'parameters': params_debug,
                    },
                    'env_test': {'env_name': env_name, 'atari': True},
                    'env_train': {'env_name': env_name, 'atari': True},
                    #'test_frequency': 50_000,
                    #'save_model_frequency': 250_000,
                    'test_frequency': 10_000,
                    #'save_model_frequency': 50_000,
                    'verbose': True,
                    #'test_frequency': None,
                    #'save_model_frequency': None,
                    'save_model_frequency': 300_000,
                },
                #trial_id='checkpointtest',
                checkpoint_frequency=250_000,
                #checkpoint_frequency=None,
                max_iterations=50_000_000,
                #max_iterations=5_000_000,
                verbose=True,
        )
        exp_runner.exp.logger.init_wandb({
            'project': 'OptionCritic-%s' % env_name
        })
    exp_runner.run()
