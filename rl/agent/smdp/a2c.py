import copy
import itertools
from collections import defaultdict
from typing import Generic, TypeVar, Optional, Tuple, List
from typing_extensions import TypedDict

import numpy as np
import torch
import torch.nn
import torch.utils.data
import torch.distributions
import torch.optim
import gym.spaces
from torchtyping import TensorType

import rl.debug_tools.frozenlake
from rl.agent.agent import DeployableAgent

class PolicyValueNetworkOutput(TypedDict):
    value: torch.Tensor
    action: torch.Tensor
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

class PolicyValueNetworkCNN(PolicyValueNetwork):
    """ A model which acts both as a policy network and a state-value estimator. """
    def __init__(self, num_features, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.num_features = num_features
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
    def __call__(self, *args, **kwargs) -> PolicyValueNetworkOutput:
        return super().__call__(*args, **kwargs)
    def forward(self, x) -> PolicyValueNetworkOutput:
        x = self.conv(x)
        x = x.view(-1,64*7*7)
        x = self.fc(x)
        v = self.v(x)
        pi = self.pi(x)
        return {
                'value': v,
                'action': pi # Unnormalized action probabilities
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
        losses.append(loss)
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

def compute_discrete_advantage_policy_gradient(
        log_action_probs : List[torch.Tensor],
        target_state_values : List[torch.Tensor],
        rewards : List[Optional[float]],
        terminals : List[bool],
        discounts : List[float],
    ) -> torch.Tensor:
    """ Advantage policy gradient for discrete action spaces. """
    device = log_action_probs[0].device
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
            device : torch.device = torch.device('cpu'),
            net : PolicyValueNetwork = None):
        self.action_space = action_space
        self.observation_space = observation_space

        self.discount_factor = discount_factor
        self.target_update_frequency = target_update_frequency
        self.polyak_rate = polyak_rate
        self.max_rollout_length = max_rollout_length
        self.obs_scale = obs_scale
        self.device = device
        self.training_env_keys = training_env_keys

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
            raise Exception('Models must be provided. `net` is missing.')
        self.net = net
        self.target_net = copy.deepcopy(net)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        # Logging
        self.state_values_current = []
        self.state_values = []
        self.state_values_std = []
        self.state_values_std_ra = []

    def observe(self, obs, reward=None, terminal=False, testing=False, time=1, discount=None, env_key=None):
        if env_key is None:
            env_key = testing
        if discount is None:
            discount = self.discount_factor

        obs = torch.tensor(obs)*self.obs_scale

        self.obs_stack[env_key].append_obs(obs, reward, terminal)

        # Choose next action
        softmax = torch.nn.Softmax(dim=1)
        net_output = self.net(
                obs.unsqueeze(0).float().to(self.device)
        )
        action_probs_unnormalized = net_output['action']
        action_probs = softmax(action_probs_unnormalized).squeeze()
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().item()

        state_value = net_output['value']
        self.state_values_current.append(state_value.item())

        self.next_action[env_key] = action
        self.obs_stack[env_key].append_action(action)

        # Save training data and train
        if not testing:
            self._steps += 1

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
            log_action_probs = {ek: [
                log_softmax(output['action']).squeeze()
                for output in outputs
            ] for ek,outputs in net_output.items()}
            # Train policy
            loss_pi = [compute_discrete_advantage_policy_gradient(
                        log_action_probs = [
                            x[a] if a is not None else torch.tensor(0)
                            for x,a in zip(log_action_probs[ek],self.action_history[ek])
                        ],
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
                    torch.stack([
                        -compute_entropy(log_probs)
                        for log_probs in log_probs_history
                    ]).mean()
                    for log_probs_history in log_action_probs.values()
            ]
            loss_entropy = torch.stack(loss_entropy).mean()
            # Take a gradient step
            loss = loss_pi+loss_v+0.01*loss_entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
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
        testing=testing
        return self.next_action[env_key]

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

    num_actors = 16
    env_name = 'PongNoFrameskip-v4'
    train_envs = [make_env(env_name,atari=True) for _ in range(num_actors)]
    test_env = make_env(env_name,atari=True)
    #train_envs = [make_env('FrozenLake-v0',one_hot_obs=True) for _ in range(num_actors)]
    #test_env = make_env('FrozenLake-v0',one_hot_obs=True)

    action_space = test_env.action_space
    observation_space = test_env.observation_space
    agent = A2CAgent(
        action_space=action_space,
        observation_space=observation_space,
        training_env_keys=[i for i in range(num_actors)],
        net  = PolicyValueNetworkCNN(observation_space.shape[0], action_space.n).to(device),
        #net  = PolicyValueNetworkLinear(observation_space.shape[0], action_space.n).to(device),
        #discount_factor=1,
        #learning_rate=0.01,
        #obs_scale=1,
        #target_update_frequency=1,
        device=device,
    )

    results = train_onpolicy(train_envs,agent,plot_frequency=10)
    test(test_env, agent)

