import numpy as np
from tqdm import tqdm
import torch
import gym
import copy
import itertools

import numpy as np

from agent.agent import Agent
from .replay_buffer import ReplayBufferStackedObs
from .policy import get_greedy_epsilon_policy, greedy_action

class HierarchicalQNetwork(torch.nn.Module):
    def __init__(self, controller, subpolicies):
        super().__init__()
        self.controller = controller
        self.subpolicies = subpolicies
        self.softmax = torch.nn.Softmax()

    def forward(self, obs0, obs1, obs_mask, temperature):
        # Values of each subpolicy
        controller_output = self.controller(obs0)
        # If there's no observation available, assign the same value to all
        # subpolicies
        uniform_controller = torch.ones_like(controller_output)
        mask = obs_mask[:,0].view(-1,1)
        controller_output = mask*controller_output+(1-mask)*uniform_controller
        
        # Compute value of primitive action as a weighted sum, weighted by
        # probability of taking that subpolicy
        subpolicy_probabilities = self.softmax(controller_output/temperature).unsqueeze(1)
        subpolicy_action_values = torch.stack([sp(obs1) for sp in self.subpolicies], dim=1)
        action_values = subpolicy_probabilities @ subpolicy_action_values
        action_values = action_values.squeeze(1)

        return action_values

class HierarchicalPolicyNetwork(torch.nn.Module):
    def __init__(self, controller, subpolicies):
        super().__init__()
        self.controller = controller
        self.subpolicies = subpolicies
        self.softmax = torch.nn.Softmax()

    def forward(self, obs0, obs1, obs_mask):
        # Values of each subpolicy
        controller_output = self.controller(obs0)
        # If there's no observation available, assign the same value to all
        # subpolicies
        uniform_controller = torch.ones_like(controller_output)
        mask = obs_mask[:,0].view(-1,1)
        controller_output = mask*controller_output+(1-mask)*uniform_controller
        
        # Compute overall probability of primitive action as a weighted sum, weighted by
        # probability of taking each subpolicy
        subpolicy_probabilities = self.softmax(controller_output).unsqueeze(1)
        primitive_action_probabilities = torch.stack([sp(obs1) for sp in self.subpolicies], dim=1)
        action_probs = subpolicy_probabilities @ primitive_action_probabilities
        action_probs = action_probs.squeeze(1)

        return action_probs

def compute_mask(*obs):
    mask = torch.tensor([[o is not None for o in obs]]).float()
    def to_tensor(o):
        if o is None:
            return torch.empty(obs[-1].shape).float().unsqueeze(0)
        return torch.tensor(o).float().unsqueeze(0)
    obs_tensors = [to_tensor(o) for o in obs]
    return obs_tensors, mask

class HDQNAgentWithDelay(Agent):
    def __init__(self, action_space, observation_space, discount_factor,
            learning_rate=1e-3, polyak_rate=0.001, device=torch.device('cpu'),
            behaviour_temperature=1, target_temperature=0.01,
            controller_q_net=None, subpolicy_q_nets=None):
        self.discount_factor = discount_factor
        self.polyak_rate = polyak_rate
        self.device = device
        self.behaviour_temperature = behaviour_temperature
        self.target_temperature= target_temperature

        # State (training)
        self.prev_obs = None
        self.prev_action = None
        self.current_obs = None
        self.current_action = None
        self.current_terminal = False
        # State (testing)
        self.prev_obs_testing = None
        self.current_obs_testing = None

        self.replay_buffer = ReplayBufferStackedObs(50000,num_obs=2)

        self.controller_q_net = controller_q_net
        self.controller_q_net_target = copy.deepcopy(self.controller_q_net)
        self.subpolicy_q_nets = subpolicy_q_nets
        self.subpolicy_q_net_targets = [copy.deepcopy(net) for net in subpolicy_q_nets]
        self.q_net = HierarchicalQNetwork(
                self.controller_q_net, self.subpolicy_q_nets)
        self.q_net_target = HierarchicalQNetwork(
                self.controller_q_net_target, self.subpolicy_q_net_targets)

        params = list(self.controller_q_net.parameters())
        for net in self.subpolicy_q_nets:
            params += list(net.parameters())
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

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
            self.prev_obs_testing = self.current_obs_testing
            self.current_obs_testing = obs
        else:
            if reward is not None: # Reward is None if this is the first step of the episode
                self.observe_step(self.current_obs, self.current_action, reward, obs, terminal)
            self.prev_obs = self.current_obs
            self.current_obs = obs
            self.prev_action = self.current_action
            self.current_action = None

    def train(self,batch_size,iterations):
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = torch.utils.data.DataLoader(
                self.replay_buffer, batch_size=batch_size, shuffle=True)
        gamma = self.discount_factor
        tau = self.polyak_rate
        temp = self.target_temperature
        optimizer = self.optimizer
        for i,((s0,s1),a1,r2,s2,t,m1) in zip(range(iterations),dataloader):
            # Fix data types
            s0 = s0.to(self.device)
            s1 = s1.to(self.device)
            m1 = m1.to(self.device)
            a1 = a1.to(self.device)
            r2 = r2.float().to(self.device)
            s2 = s2.to(self.device)
            m2 = torch.ones_like(m1).to(self.device)
            t = t.float().to(self.device)
            # Value estimate
            action_values = self.q_net(s1,s2,m2)
            optimal_actions = action_values.argmax(1)
            y = r2+gamma*self.q_net_target(s1,s2,m2,temp)[range(batch_size),optimal_actions]*(1-t)
            # Update Q network
            optimizer.zero_grad()
            loss = ((y-self.q_net(s0,s1,m1,temp)[range(batch_size),a1.flatten()])**2).mean()
            loss.backward()
            optimizer.step()

            # Update target weights
            params = [zip(self.controller_q_net_target.parameters(), self.q_net.parameters())]
            for net1,net2 in zip(self.subpolicy_q_net_targets,self.subpolicy_q_nets):
                params.append(zip(net1.parameters(),net2.parameters()))
            for p1,p2 in itertools.chain.from_iterable(params):
                p1.data = (1-tau)*p1+tau*p2

    def act(self, testing=False):
        """Return a random action according to the current behaviour policy"""
        #observation = torch.tensor(observation, dtype=torch.float).view(-1,4,84,84).to(self.device)
        if testing:
            (obs0,obs1),mask = compute_mask(self.prev_obs_testing,self.current_obs_testing)
            vals = self.q_net(obs0,obs1,mask,self.target_temperature).squeeze()
            policy = self.target_policy
        else:
            (obs0,obs1),mask = compute_mask(self.prev_obs,self.current_obs)
            vals = self.q_net(obs0,obs1,mask,self.behaviour_temperature).squeeze()
            policy = self.behaviour_policy
        self.last_vals = vals
        dist = policy(vals)
        action = dist.sample().item()
        self.current_action = action
        return action

    def get_state_action_value(self, obs0, obs1, action):
        (obs0,obs1),mask = compute_mask(obs0,obs1)
        vals = self.q_net(obs0,obs1,mask).squeeze()
        if vals.size() == ():
            return vals.item()
        else:
            return vals[action].item()

    def test_once(self, env, max_steps=np.inf, render=False):
        reward_sum = 0
        sa_vals = []
        obs = env.reset()
        self.observe_change(obs,testing=True)
        for steps in itertools.count():
            if steps > max_steps:
                break
            action = self.act(testing=True)
            sa_vals.append(self.get_state_action_value(self.prev_obs_testing,self.current_obs_testing,action))
            obs, reward, done, _ = env.step(action)
            self.observe_change(obs,testing=True)
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

class HDQNAgentWithDelayAC(Agent):
    def __init__(self, action_space, observation_space, discount_factor,
            behaviour_epsilon,
            learning_rate=1e-3, polyak_rate=0.001, device=torch.device('cpu'),
            controller_net=None, subpolicy_nets=None, q_net=None):
        self.discount_factor = discount_factor
        self.polyak_rate = polyak_rate
        self.behaviour_epsilon = behaviour_epsilon
        self.device = device

        # State (training)
        self.prev_obs = None
        self.prev_action = None
        self.current_obs = None
        self.current_action = None
        self.current_terminal = False
        # State (testing)
        self.prev_obs_testing = None
        self.current_obs_testing = None

        self.replay_buffer = ReplayBufferStackedObs(50000,num_obs=2)

        self.q_net = q_net
        self.q_net_target = copy.deepcopy(self.q_net)
        self.controller_net = controller_net
        self.controller_net_target = copy.deepcopy(self.controller_net)
        self.subpolicy_nets = subpolicy_nets
        self.subpolicy_net_targets = [copy.deepcopy(net) for net in subpolicy_nets]
        self.policy_net = HierarchicalPolicyNetwork(
                self.controller_net, self.subpolicy_nets)
        self.policy_net_target = HierarchicalPolicyNetwork(
                self.controller_net_target, self.subpolicy_net_targets)

        params = list(self.controller_net.parameters())
        for net in self.subpolicy_nets:
            params += list(net.parameters())
        self.actor_optimizer = torch.optim.Adam(params, lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

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
            self.prev_obs_testing = self.current_obs_testing
            self.current_obs_testing = obs
        else:
            if reward is not None: # Reward is None if this is the first step of the episode
                self.observe_step(self.current_obs, self.current_action, reward, obs, terminal)
            self.prev_obs = self.current_obs
            self.current_obs = obs
            self.prev_action = self.current_action
            self.current_action = None

    def train(self,batch_size,iterations):
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = torch.utils.data.DataLoader(
                self.replay_buffer, batch_size=batch_size, shuffle=True)
        gamma = self.discount_factor
        tau = self.polyak_rate
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        for i,((s0,s1),a1,r2,s2,t,m1) in zip(range(iterations),dataloader):
            # Fix data types
            s0 = s0.to(self.device)
            s1 = s1.to(self.device)
            m1 = m1.to(self.device)
            a1 = a1.to(self.device)
            r2 = r2.float().to(self.device)
            s2 = s2.to(self.device)
            m2 = torch.ones_like(m1).to(self.device)
            t = t.float().to(self.device)
            # Update Q function
            action_probs = self.policy_net_target(s0,s1,m1)
            next_state_vals = (action_probs * self.q_net_target(s2)).sum(1)
            val_target = r2+gamma*next_state_vals*(1-t)
            val_pred = self.q_net(s1)[range(batch_size),a1.squeeze()]
            critic_optimizer.zero_grad()
            critic_loss = ((val_target-val_pred)**2).mean()
            critic_loss.backward()
            critic_optimizer.step()
            # Update policy function
            action_values = self.q_net(s1)
            action_probs = self.policy_net(s0,s1,m1)
            actor_optimizer.zero_grad()
            actor_loss = -(action_values * action_probs).mean()
            actor_loss.backward()
            actor_optimizer.step()
            # Update target weights
            params = [zip(self.q_net_target.parameters(), self.q_net.parameters())]
            params += [zip(self.controller_net_target.parameters(), self.controller_net.parameters())]
            for net1,net2 in zip(self.subpolicy_net_targets,self.subpolicy_nets):
                params.append(zip(net1.parameters(),net2.parameters()))
            for p1,p2 in itertools.chain.from_iterable(params):
                p1.data = (1-tau)*p1+tau*p2

    def act(self, testing=False):
        """Return a random action according to the current behaviour policy"""
        #observation = torch.tensor(observation, dtype=torch.float).view(-1,4,84,84).to(self.device)
        if testing:
            (obs0,obs1),mask = compute_mask(self.prev_obs_testing,self.current_obs_testing)
        else:
            (obs0,obs1),mask = compute_mask(self.prev_obs,self.current_obs)
        action_probs = self.policy_net(obs0,obs1,mask).squeeze()

        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample().item()
        self.current_action = action
        self.last_vals = self.q_net(obs1)[0,action].item()
        return action

    def get_state_action_value(self, obs, action):
        vals = self.q_net(torch.tensor(obs).float())[action]
        if vals.size() == ():
            return vals.item()
        else:
            return vals[action].item()

    def test_once(self, env, max_steps=np.inf, render=False):
        reward_sum = 0
        sa_vals = []
        obs = env.reset()
        self.observe_change(obs,testing=True)
        for steps in itertools.count():
            if steps > max_steps:
                break
            action = self.act(testing=True)
            sa_vals.append(self.get_state_action_value(self.current_obs_testing,action))
            obs, reward, done, _ = env.step(action)
            self.observe_change(obs,testing=True)
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

