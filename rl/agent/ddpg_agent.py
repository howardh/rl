import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data
import gym
import copy

from rl.agent.agent import Agent

import numpy as np

from . import ReplayBuffer

class QNetwork(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=obs_size,out_features=400)
        self.fc2 = torch.nn.Linear(in_features=400+action_size,out_features=300)
        self.fc3 = torch.nn.Linear(in_features=300,out_features=1)
        self.relu = torch.nn.LeakyReLU()

        #self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        #self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        #self.fc3.weight.data.uniform_(-0.003,0.003)

    def forward(self, state, action):
        a = action
        x = state
        x = self.relu(self.fc1(x))
        x = torch.cat([x,a],1)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PolicyNetwork(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(PolicyNetwork, self).__init__()
        #self.fc1 = torch.nn.Linear(in_features=obs_size,out_features=400)
        #self.fc2 = torch.nn.Linear(in_features=400,out_features=300)
        #self.fc3 = torch.nn.Linear(in_features=300,out_features=action_size)
        self.fc1 = torch.nn.Linear(in_features=obs_size,out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128,out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64,out_features=action_size)
        self.relu = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()

        #self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        #self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        #self.fc3.weight.data.uniform_(-0.003,0.003)

    def forward(self, state, noise=0):
        x = state
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        #x = self.tanh(self.fc3(x)+noise)
        x = (self.fc3(x)+noise).clamp(-1,1)
        return x

class DDPGAgent(Agent):
    def __init__(self, action_space, observation_space, discount_factor, actor_lr=1e-4, critic_lr=1e-3, polyak_rate=0.001, device=torch.device('cpu'), training_noise=None, testing_noise=None, actor=None, critic=None):
        self.action_space = action_space
        self.observation_space = observation_space
        self.discount_factor = discount_factor
        self.polyak_rate = polyak_rate
        self.device = device
        self.training_noise = training_noise
        self.testing_noise = testing_noise

        self.running_episode = False
        self.prev_obs = None
        self.replay_buffer = ReplayBuffer(50000)

        if actor is None:
            self.actor = PolicyNetwork(observation_space.shape[0], action_space.shape[0])
        else:
            self.actor = actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        if critic is None:
            self.critic = QNetwork(observation_space.shape[0], action_space.shape[0])
        else:
            self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def to(self,device):
        self.device = device
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)

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
        critic_optimizer = self.critic_optimizer
        actor_optimizer = self.actor_optimizer
        obs_shape = [-1,self.observation_space.shape[0]]
        action_shape = [-1,self.action_space.shape[0]]
        for i,(s0,a0,r1,s1,t) in zip(range(iterations),dataloader):
            # Fix data types
            s0 = s0.view(*obs_shape).to(self.device)
            a0 = a0.view(*action_shape).to(self.device)
            r1 = r1.float().view(-1,1).to(self.device)
            s1 = s1.view(*obs_shape).to(self.device)
            t = t.float().view(-1,1).to(self.device)
            # Value estimate
            y = r1+gamma*self.critic_target(s1,self.actor_target(s1))*(1-t)
            # Update critic
            critic_optimizer.zero_grad()
            critic_loss = ((y-self.critic(s0,a0))**2).mean()
            critic_loss.backward()
            critic_optimizer.step()
            # Update actor
            actor_optimizer.zero_grad()
            actor_loss = -self.critic(s0,self.actor(s0)).mean()
            actor_loss.backward()
            actor_optimizer.step()

            # Update target weights
            for p1,p2 in zip(self.critic_target.parameters(), self.critic.parameters()):
                p1.data = (1-tau)*p1+tau*p2
            for p1,p2 in zip(self.actor_target.parameters(), self.actor.parameters()):
                p1.data = (1-tau)*p1+tau*p2

    def act(self, observation, testing=False):
        """Return a random action according to the current behaviour policy"""
        observation = torch.tensor(observation, dtype=torch.float).view(-1,self.observation_space.shape[0]).to(self.device)
        if testing:
            if self.testing_noise:
                noise = self.testing_noise.sample([1,self.action_space.shape[0]]).to(self.device)
        else:
            if self.training_noise:
                noise = self.training_noise.sample([1,self.action_space.shape[0]]).to(self.device)
        action = self.actor(observation, noise).detach()
        return action.cpu()[0].numpy()
