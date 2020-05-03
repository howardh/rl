import numpy as np
from tqdm import tqdm
import torch
import gym
import copy
import itertools

import numpy as np

from agent.agent import Agent
from . import ReplayBuffer
from .policy import get_greedy_epsilon_policy, greedy_action

class QNetwork(torch.nn.Module):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,out_channels=16,kernel_size=8,stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=16,out_channels=32,kernel_size=4,stride=2),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=32*9*9,out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256,out_features=num_actions),
        )
    def forward(self, obs):
        x = obs
        x = self.conv(x)
        x = x.view(-1,9*9*32)
        x = self.fc(x)
        return x

class DQNAgent(Agent):
    def __init__(self, action_space, observation_space, discount_factor, learning_rate=1e-3, polyak_rate=0.001, device=torch.device('cpu'), behaviour_policy=get_greedy_epsilon_policy(0.1), target_policy=get_greedy_epsilon_policy(0), q_net=None, replay_buffer_size=50000):
        self.action_space = action_space
        self.observation_space = observation_space

        self.discount_factor = discount_factor
        self.polyak_rate = polyak_rate
        self.device = device
        self.behaviour_policy = behaviour_policy
        self.target_policy = target_policy

        # State (training)
        self.current_obs = None
        self.current_action = None
        self.current_terminal = False
        # State (testing)
        self.current_obs_testing = None

        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        if q_net is None:
            self.q_net = QNetwork(action_space.n).to(device)
        else:
            self.q_net = q_net
        self.q_net_target = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def to(self,device):
        self.device = device
        self.q_net.to(device)
        self.q_net_target.to(device)

    def observe_step(self,obs0, action0, reward1, obs1, terminal=False):
        obs0 = torch.Tensor(obs0)
        obs1 = torch.Tensor(obs1)
        self.replay_buffer.add_transition(obs0, action0, reward1, obs1, terminal)

    def observe_change(self, obs, reward=None, terminal=False, testing=False):
        if testing:
            self.current_obs_testing = obs
        else:
            if reward is not None and self.current_action is not None: # Reward is None if this is the first step of the episode
                self.observe_step(self.current_obs, self.current_action, reward, obs, terminal)
            self.current_obs = obs
            self.current_action = None

    def train(self,batch_size=2,iterations=1):
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = torch.utils.data.DataLoader(
                self.replay_buffer, batch_size=batch_size, shuffle=True)
        gamma = self.discount_factor
        tau = self.polyak_rate
        optimizer = self.optimizer
        for i,(s0,a0,r1,s1,t) in zip(range(iterations),dataloader):
            # Fix data types
            s0 = s0.to(self.device)
            a0 = a0.to(self.device)
            r1 = r1.float().to(self.device)
            s1 = s1.to(self.device)
            t = t.float().to(self.device)
            # Value estimate
            action_values = self.q_net(s1)
            optimal_actions = greedy_action(action_values)
            y = r1+gamma*self.q_net_target(s1)[range(batch_size),optimal_actions]*(1-t)
            # Update Q network
            optimizer.zero_grad()
            loss = ((y-self.q_net(s0)[range(batch_size),a0.flatten()])**2).mean()
            loss.backward()
            optimizer.step()

            # Update target weights
            for p1,p2 in zip(self.q_net_target.parameters(), self.q_net.parameters()):
                p1.data = (1-tau)*p1+tau*p2

    def act(self, testing=False):
        """Return a random action according to the current behaviour policy"""
        #observation = torch.tensor(observation, dtype=torch.float).view(-1,4,84,84).to(self.device)
        if testing:
            obs = self.current_obs_testing
            policy = self.target_policy
        else:
            obs = self.current_obs
            policy = self.behaviour_policy
        obs = torch.tensor(obs).unsqueeze(0).float()
        vals = self.q_net(obs)
        dist = policy(vals)
        action = dist.sample().item()
        self.current_action = action
        self.last_vals = vals
        return action

    def get_state_action_value(self, observation, action):
        observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0)
        vals = self.q_net(observation).squeeze()
        if vals.size() == ():
            return vals.item()
        else:
            return vals[action].item()

    def test_once(self, env, max_steps=np.inf, render=False):
        reward_sum = 0
        sa_vals = []
        obs = env.reset()
        self.observe_change(obs, testing=True)
        for steps in itertools.count():
            if steps > max_steps:
                break
            action = self.act(testing=True)
            sa_vals.append(self.get_state_action_value(obs,action))
            obs, reward, done, _ = env.step(action)
            self.observe_change(obs, reward, done, testing=True)
            reward_sum += reward
            if render:
                env.render()
            if done:
                break
        return reward_sum, np.mean(sa_vals)

    def test(self, env, iterations, max_steps=np.inf, render=False, record=True, processors=1):
        rewards = []
        sa_vals = []
        for i in range(iterations):
            r,sav = self.test_once(env, render=render, max_steps=max_steps)
            rewards.append(r)
            sa_vals.append(sav)
        return rewards, sa_vals

class HierarchicalDQNAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, batch_size, iterations, value_function):
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = torch.utils.data.DataLoader(
                self.replay_buffer, batch_size=batch_size, shuffle=True)
        gamma = self.discount_factor
        tau = self.polyak_rate
        optimizer = self.optimizer
        for i,(s0,a0,r1,s1,t) in zip(range(iterations),dataloader):
            # Fix data types
            s0 = s0.to(self.device)
            a0 = a0.to(self.device)
            r1 = r1.float().to(self.device)
            s1 = s1.to(self.device)
            t = t.float().to(self.device)
            # Value estimate
            y = r1+gamma*value_function(s1)*(1-t)
            y = y.detach()
            # Update Q network
            optimizer.zero_grad()
            loss = ((y-self.q_net(s0)[range(batch_size),a0.flatten()])**2).mean()
            loss.backward()
            optimizer.step()

            # Update target weights
            for p1,p2 in zip(self.q_net_target.parameters(), self.q_net.parameters()):
                p1.data = (1-tau)*p1+tau*p2
