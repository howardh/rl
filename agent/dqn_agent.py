import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data
import gym

from agent.agent import Agent

import numpy as np

class ReplayBuffer(torch.utils.data.Dataset):
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.index = 0
        self.prev_transition = None

    def add_transition(self, obs0, action0, reward, obs, terminal=False):
        transition = (obs0, action0, reward, obs, terminal)
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.index] = transition
            self.index = (self.index+1)%self.max_size

    def step(self, obs, reward, action, terminal=False):
        if self.prev_transition is not None:
            obs0, reward0, action0 = self.prev_transition
            self.add_transition(obs0, action0, reward, obs, terminal)
        if terminal:
            self.prev_transition = None
        else:
            self.prev_transition = (obs, reward, action)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]

class ObservationStackingWrapper(gym.Wrapper):
    def __init__(self):
        pass

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
            torch.nn.Linear(in_features=32*9*9,out_features=int(32*9*9/2)),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=int(32*9*9/2),out_features=num_actions),
        )
    def forward(self, obs):
        x = obs
        x = self.conv(x)
        x = x.view(-1,9*9*32)
        x = self.fc(x)
        return x

def get_greedy_epsilon_policy(eps):
    def foo(values):
        batch_size, num_actions = values.size()
        probs = torch.ones_like(values)*(eps/(num_actions-1))
        probs[range(batch_size),values.argmax(1)] = 1-eps
        dist = torch.distributions.Categorical(probs)
        return dist
    return foo

def greedy_action(values):
    return values.argmax(1)

class DQNAgent(Agent):
    def __init__(self, action_space, observation_space, discount_factor, learning_rate=1e-3, polyak_rate=0.001, device=torch.device('cpu'), behaviour_policy=get_greedy_epsilon_policy(0.1), target_policy=get_greedy_epsilon_policy(0)):
        self.discount_factor = discount_factor
        self.polyak_rate = polyak_rate
        self.device = device
        self.behaviour_policy = behaviour_policy
        self.target_policy = target_policy

        self.running_episode = False
        self.prev_obs = None
        self.replay_buffer = ReplayBuffer(50000)
        self.q_net = QNetwork(action_space.n).to(device)
        self.q_net_target = QNetwork(action_space.n).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def to(self,device):
        self.device = device
        self.q_net.to(device)
        self.q_net_target.to(device)

    def observe_step(self,obs0, action0, reward1, obs1, terminal=False):
        obs0 = torch.Tensor(obs0)
        obs1 = torch.Tensor(obs1)
        self.replay_buffer.add_transition(obs0, action0, reward1, obs1, terminal)

    def train(self,batch_size,iterations):
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = torch.utils.data.DataLoader(
                self.replay_buffer, batch_size=batch_size, shuffle=True)
        gamma = self.discount_factor
        tau = self.polyak_rate
        optimizer = self.optimizer
        for i,(s0,a0,r1,s1,t) in enumerate(dataloader):
            if i >= iterations:
                break
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

    def act(self, observation, testing=False):
        """Return a random action according to the current behaviour policy"""
        observation = torch.tensor(observation, dtype=torch.float).view(-1,4,84,84).to(self.device)
        vals = self.q_net(observation)
        if testing:
            policy = self.target_policy
        else:
            policy = self.behaviour_policy
        dist = policy(vals)
        return dist.sample()
