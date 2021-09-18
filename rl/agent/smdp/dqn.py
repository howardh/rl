import os
from typing import Optional, Tuple, Generic, TypeVar, Sequence, Union, Mapping
import copy
import itertools
from collections import defaultdict

import dill
import gym
import gym.spaces
import torch
import torch.nn
import torch.utils.data
import torch.distributions
import torch.optim
import numpy as np

from experiment.logger import Logger

from rl.agent.agent import DeployableAgent
from rl.agent.replay_buffer import AtariReplayBuffer, ReplayBuffer

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
class QNetworkCNN_1(torch.nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,out_channels=16,kernel_size=8,stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=16,out_channels=16,kernel_size=4,stride=2),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=16*9*9,out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256,out_features=num_actions),
        )
    def forward(self, obs):
        x = obs
        x = self.conv(x)
        x = x.view(-1,9*9*16)
        x = self.fc(x)
        return x
class QNetworkCNN_2(torch.nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,out_channels=16,kernel_size=8,stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=16,out_channels=16,kernel_size=4,stride=2),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=16*9*9,out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256,out_features=num_actions),
        )
    def forward(self, obs):
        x = obs
        x = self.conv(x)
        x = x.view(-1,9*9*16)
        x = self.fc(x)
        return x
class QNetworkCNN_3(torch.nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,out_channels=16,kernel_size=8,stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=16,out_channels=16,kernel_size=4,stride=2),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=16*9*9,out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128,out_features=num_actions),
        )
    def forward(self, obs):
        x = obs
        x = self.conv(x)
        x = x.view(-1,9*9*16)
        x = self.fc(x)
        return x

class QNetworkFCNN(torch.nn.Module):
    def __init__(self, sizes : Sequence[int] = [2,128,4]):
        super().__init__()
        self.sizes = sizes

        layers = []
        for in_size,out_size in zip(sizes,sizes[1:]):
            layers.append(torch.nn.Linear(in_features=in_size,out_features=out_size))
            layers.append(torch.nn.ReLU())
        layers.pop() # Remove last ReLU
        self.fc = torch.nn.Sequential(*layers)
    def forward(self, obs):
        return self.fc(obs)
    def state_dict(self):
        return {
                'sizes': self.sizes,
                'model_state_dict': super().state_dict(),
        }
    def load_state_dict(self, state_dict):
        return super().load_state_dict(state_dict['model_state_dict'])

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

def epsilon_greedy(vals : torch.Tensor, eps : float) -> torch.Tensor:
    max_val = vals.max()
    is_max = (vals == max_val).to(torch.float)
    not_max = 1-is_max
    if not_max.sum() == 0:
        return is_max/is_max.sum()
    return is_max/is_max.sum()*(1-eps) + not_max/not_max.sum()*eps

class DQNAgent(DeployableAgent):
    """
    Implementation of DQN based on Mnih 2015, but with added support for semi-markov decision processes (i.e. observations include the number of time steps since the last observation), and variable discount rates.
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
            target_update_frequency : int = 10_000,
            polyak_rate : float = 1.,
            device = torch.device('cpu'),
            behaviour_eps : float = 0.1,
            target_eps : float = 0,
            eps_annealing_steps : int = 1_000_000,
            q_net : torch.nn.Module = None,
            atari : bool = True
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

        # State (training)
        self.obs_stack = defaultdict(lambda: ObservationStack())
        self._steps = 0 # Number of steps experienced
        self._training_steps = 0 # Number of training steps experienced

        if atari:
            self.replay_buffer = AtariReplayBuffer(replay_buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(replay_buffer_size)

        if q_net is None:
            self.q_net = QNetworkCNN(action_space.n).to(device)
        else:
            self.q_net = q_net
        self.q_net_target = copy.deepcopy(self.q_net)
        #self.q_net_target = QNetworkCNN(action_space.n).to(device)
        #self.q_net_target.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        #self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=learning_rate, momentum=0.95)

        self.logger = Logger(key_name='step')
        self.train_action_values = []
        self.train_action_values_target = []
        self.train_action_value_diff = []
        self.train_loss_diff = []

    def observe(self, obs, reward=None, terminal=False, testing=False, time=1, discount=None, env_key=None):
        if env_key is None:
            env_key = testing
        if discount is None:
            discount = self.discount_factor

        # Reward clipping
        if reward is not None:
            reward = np.clip(reward, -1, 1)

        self.obs_stack[env_key].append_obs(obs, reward, terminal)

        # Add to replay buffer
        transition = self.obs_stack[env_key].get_transition()
        if transition is not None and not testing:
            self._steps += 1
            
            self.replay_buffer.add_transition(
                    *transition, [time, discount])

            # Train each time something is added to the buffer
            self._train()
    def _train(self, iterations=1):
        if len(self.replay_buffer) < self.batch_size*iterations:
            return
        if len(self.replay_buffer) < self.warmup_steps:
            return
        if self._steps % self.update_frequency != 0:
            return
        dataloader = torch.utils.data.DataLoader(
                self.replay_buffer, batch_size=self.batch_size, shuffle=True,
                pin_memory=self.device.type == 'cuda')
        optimizer = self.optimizer
        #criterion = torch.nn.SmoothL1Loss(reduction='mean')
        criterion = torch.nn.MSELoss(reduction='mean')
        for _,(s0,a0,r1,s1,term,time,gamma) in zip(range(iterations),dataloader):
            # Fix data types
            s0 = s0.to(self.device).float()/self.obs_scale
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

            # Value estimate (target)
            action_values_target = self.q_net_target(s1)
            optimal_values = action_values_target.max(1)[0].unsqueeze(1)
            y = r1+g_t*optimal_values*(1-term)
            y = y.detach()

            # Value estimate (prediction)
            y_pred = self.q_net(s0).gather(1,a0)
    
            self.train_action_values.append(y_pred.mean().item()) # XXX: DEBUG
            self.train_action_values_target.append(y.mean().item()) # XXX: DEBUG

            #if self._steps >= 200_000:
            #    breakpoint()

            if y.shape != y_pred.shape: # XXX: DEBUG
                raise Exception()

            # Update Q network
            optimizer.zero_grad()
            #loss = (y-y_pred).pow(2).mean()
            loss = criterion(y_pred.squeeze(),y.squeeze())
            loss.backward()
            for param in self.q_net.parameters():
                assert param.grad is not None
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

            y_pred2 = self.q_net(s0).gather(1,a0) #XXX: DEBUG
            self.train_action_value_diff.append((y_pred2-y_pred).mean().item())#XXX: DEBUG
            self.train_loss_diff.append((criterion(y_pred2,y)-loss).item()) #XXX: DEBUG

        self._training_steps += 1

        self._update_target()
    def _update_target(self):
        if self._training_steps % self.target_update_frequency != 0:
            return
        tau = self.polyak_rate
        for p1,p2 in zip(self.q_net_target.parameters(), self.q_net.parameters()):
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
        obs = torch.tensor(obs).to(self.device).float()/self.obs_scale
        if isinstance(self.observation_space,gym.spaces.Discrete):
            obs = obs.unsqueeze(0).unsqueeze(0)
        elif isinstance(self.observation_space,gym.spaces.Box):
            obs = obs.unsqueeze(0)
        else:
            raise NotImplementedError()
        vals = self.q_net(obs)
        if torch.rand(1) >= eps:
            action = vals.max(dim=1)[1].item()
        else:
            action = torch.randint(0,self.action_space.n,[1]).item()
        #probs = epsilon_greedy(vals,eps)
        #dist = torch.distributions.Categorical(probs=probs)
        #action = dist.sample().item()

        self.obs_stack[env_key].append_action(action)
        if testing:
            self.logger.append(
                    step=self._steps,
                    testing_action_value=vals.flatten()[action].item()
            )

        return action
    def _compute_annealed_epsilon(self, max_steps=1_000_000):
        eps = self.eps[False]
        return (1-eps)*max(1-self._steps/max_steps,0)+eps # Linear
        #return (1-eps)*np.exp(-self._steps/max_steps)+eps # Exponential

    def state_dict(self):
        return {
                'q_net': self.q_net.state_dict(),
                'q_net_target': self.q_net_target.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'obs_stack': {k:os.state_dict() for k,os in self.obs_stack.items()},
                'steps': self._steps,
                'training_steps': self._training_steps,
                'logger': self.logger.state_dict(),
                'replay_buffer': self.replay_buffer.state_dict()
        }
    def load_state_dict(self, state):
        self.q_net.load_state_dict(state['q_net'])
        self.q_net_target.load_state_dict(state['q_net_target'])
        self.optimizer.load_state_dict(state['optimizer'])
        for k,os_state in state['obs_stack'].items():
            self.obs_stack[k].load_state_dict(os_state)
        self._steps = state['steps']
        self._training_steps = state['training_steps']
        self.logger.load_state_dict(state['logger'])
        self.replay_buffer.load_state_dict(state['replay_buffer'])

    def state_dict_deploy(self):
        return {
                'action_space': self.action_space,
                'observation_space': self.observation_space,
                'q_net': self.q_net.state_dict(),
                'q_net_class': self.q_net.__class__,
        }
    def load_state_dict_deploy(self, state):
        self.q_net.load_state_dict(state['q_net'])

def make_agent_from_deploy_state(state : Union[str,Mapping], device : torch.device = torch.device('cpu')):
    if isinstance(state, str): # If it's a string, then it's the filename to the dilled state
        filename = state
        if not os.path.isfile(filename):
            raise Exception('No file found at %s' % filename)
        with open(filename, 'rb') as f:
            state = dill.load(f)
    if not isinstance(state,Mapping):
        raise ValueError('State is expected to be a dictionary. Found a %s.' % type(state))

    cls = state['q_net_class']
    if cls is QNetworkCNN:
        q_net = QNetworkCNN(state['action_space'].n).to(device)
    elif cls is QNetworkFCNN:
        q_net = QNetworkFCNN(state['q_net']['sizes']).to(device)
    else:
        raise Exception('Unable to initialize Q network of type %s' % cls)

    agent = DQNAgent(
            action_space=state['action_space'],
            observation_space=state['observation_space'],
            q_net = q_net,
    )
    agent.load_state_dict_deploy(state)
    return agent

if __name__ == "__main__":
    from tqdm import tqdm
    import torch.cuda
    import numpy as np
    import pprint
    import gym
    from gym.wrappers import FrameStack, AtariPreprocessing

    def make_env(env_name):
        env = gym.make(env_name)
        env = AtariPreprocessing(env)
        env = FrameStack(env, 4)
        return env
    
    def train(envs, agent, training_steps=50_000_000, test_frequency=250_000):
        test_results = {}
        env = envs[0]
        env_test = envs[1]

        rewards = []
        ep_len = 0

        done = True
        for i in tqdm(range(training_steps), desc='training'):
            if i % test_frequency == 0:
                test_results[i] = [test(env_test, agent) for _ in tqdm(range(5), desc='testing')]
                avg = np.mean([x['total_reward'] for x in test_results[i]])
                action_values = np.mean(agent.logger[-1]['testing_action_value'])
                tqdm.write('Iteration {i}\t Average reward: {avg}\t Action values: {action_values}'.format(i=i,avg=avg,action_values=action_values))
                tqdm.write(pprint.pformat(test_results[i], indent=4))
                tqdm.write('Mean weights:')
                tqdm.write(pprint.pformat([x.abs().mean().item() for x in agent.q_net.parameters()], indent=4))
            if done:
                if len(rewards) > 0:
                    tqdm.write('Iteration {i} {ep_len}\t Total training reward: {r}\t Action values: {av}\t AV diff: {diff}\t Loss diff: {ldiff}'.format(i=i,r=np.sum(rewards), av=np.mean(agent.train_action_values),diff=np.mean(agent.train_action_value_diff),ldiff=np.mean(agent.train_loss_diff),ep_len=ep_len))
                rewards = []
                ep_len = 0
                agent.train_action_values = []
                agent.train_action_value_diff = []
                agent.train_action_values_target = []
                agent.train_loss_diff = []
                obs = env.reset()
                agent.observe(obs, testing=False)
            obs, reward, done, _ = env.step(agent.act(testing=False))
            agent.observe(obs, reward, done, testing=False)
            rewards.append(reward)
            ep_len += 1
        env.close()

        return test_results

    def test(env, agent):
        total_reward = 0
        total_steps = 0

        obs = env.reset()
        agent.observe(obs, testing=True)
        for total_steps in tqdm(itertools.count(), desc='test episode'):
            #env.render()
            obs, reward, done, _ = env.step(agent.act(testing=True))
            total_reward += reward
            agent.observe(obs, reward, done, testing=True)
            if done:
                break
        env.close()

        return {
            'total_steps': total_steps,
            'total_reward': total_reward,
        }

    if torch.cuda.is_available():
        print('GPU found')
        device = torch.device('cuda')
    else:
        print('No GPU found. Running on CPU.')
        device = torch.device('cpu')

    #env_name = 'Pong-v0'
    env_name = 'PongNoFrameskip-v4'
    #env_name = 'Breakout-v0'
    env = [make_env(env_name), make_env(env_name)]

    #agent = DQNAgent(
    #    action_space=env[0].action_space,
    #    observation_space=env[0].observation_space,
    #    #discount_factor=0.99,
    #    #learning_rate=2.5e-4,
    #    #target_update_frequency=40_000,
    #    #polyak_rate=1,
    #    #warmup_steps=50_000,
    #    ##warmup_steps=50,
    #    #replay_buffer_size=1_000_000,
    #    ##replay_buffer_size=1000,
    #    q_net=QNetworkCNN(env[0].action_space.n).to(device),
    #    device=device,
    #)

    agent = DQNAgent(
        action_space=env[0].action_space,
        observation_space=env[0].observation_space,
        discount_factor=0.99,
        behaviour_eps=0.02,
        learning_rate=1e-4,
        update_frequency=1,
        target_update_frequency=1_000,
        polyak_rate=1,
        warmup_steps=10_000,
        replay_buffer_size=100_000,
        q_net=QNetworkCNN(env[0].action_space.n).to(device),
        device=device,
    )

    #results = train(env,agent)
    results = train(env,agent,test_frequency=10_000)
