from collections import deque
import numpy as np
from tqdm import tqdm
import torch
import gym
import copy
import itertools

import numpy as np

from agent.agent import Agent
from .replay_buffer import ReplayBufferStackedObs,ReplayBufferStackedObsAction
from .policy import get_greedy_epsilon_policy, greedy_action

import utils

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

class HierarchicalPolicyNetworkAugmentedState(torch.nn.Module):
    def __init__(self, controller, subpolicies):
        super().__init__()
        self.controller = controller
        self.subpolicies = subpolicies
        self.softmax = torch.nn.Softmax()

    def forward(self, obs0, action0, obs1, obs_mask):
        # Values of each subpolicy
        controller_output = self.controller(obs0,action0)
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

class SeedableRandomSampler(torch.utils.data.sampler.RandomSampler):
    def __init__(self, data_source, replacement=False, num_samples=None, generator=None):
        super().__init__(data_source, replacement, num_samples)
        if generator is None:
            self.generator = torch.Generator()
        else:
            self.generator = generator

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64, generator=self.generator).tolist())
        return iter(torch.randperm(n,generator=self.generator).tolist())

def compute_mask(*obs):
    mask = torch.tensor([[o is not None for o in obs]]).float()
    def to_tensor(o):
        if o is None:
            return torch.zeros(obs[-1].shape).float().unsqueeze(0)
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
            self.observe_change(obs,reward,testing=True)
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
            behaviour_epsilon, delay_steps=1, replay_buffer_size=50000,
            controller_learning_rate=1e-3, subpolicy_learning_rate=1e-3,
            q_net_learning_rate=1e-3,
            polyak_rate=0.001, device=torch.device('cpu'),
            controller_net=None, subpolicy_nets=None, q_net=None, seed=None):
        self.action_space = action_space # TODO: Do I need to make a deep copy of this so changes to the random state doesn't affect other copies?
        self.observation_space = observation_space
        self.discount_factor = discount_factor
        self.polyak_rate = polyak_rate
        self.behaviour_epsilon = behaviour_epsilon
        self.device = device

        self.rand = np.random.RandomState(seed)
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

        # State (training)
        self.obs_stack = deque(maxlen=delay_steps+1)
        self.current_obs = None
        self.current_action = None
        self.current_terminal = False
        # State (testing)
        self.obs_stack_testing = deque(maxlen=delay_steps+1)
        self.current_obs_testing = None

        self.replay_buffer = ReplayBufferStackedObs(replay_buffer_size,num_obs=delay_steps+1)

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

        controller_params = list(self.controller_net.parameters())
        subpolicy_params = []
        for net in self.subpolicy_nets:
            subpolicy_params += list(net.parameters())
        self.actor_optimizer = torch.optim.Adam([
            {'params': controller_params, 'lr': controller_learning_rate},
            {'params': subpolicy_params, 'lr': subpolicy_learning_rate}])
        self.critic_optimizer = torch.optim.Adam(
                self.q_net.parameters(), lr=q_net_learning_rate)

        self.to(device)

    def to(self,device):
        self.device = device
        self.q_net.to(device)
        self.q_net_target.to(device)
        self.controller_net.to(device)
        self.controller_net_target.to(device)
        for n in self.subpolicy_nets + self.subpolicy_net_targets:
            n.to(device)
        self.policy_net.to(device)
        self.policy_net_target.to(device)

    def set_random_state(self, rand):
        self.rand = rand

    def observe_step(self,obs0, action0, reward1, obs1, terminal=False):
        obs0 = torch.Tensor(obs0)
        obs1 = torch.Tensor(obs1)
        self.replay_buffer.add_transition(obs0, action0, reward1, obs1, terminal)

    def observe_change(self, obs, reward=None, terminal=False, testing=False):
        if testing:
            if reward is None:
                self.obs_stack_testing.clear()
            self.obs_stack_testing.append(obs)
            self.current_obs_testing = obs
        else:
            if reward is None: # Reward is None if this is the first step of the episode
                self.obs_stack.clear()
            else:
                self.observe_step(self.current_obs, self.current_action, reward, obs, terminal)
            self.obs_stack.append(obs)
            self.current_obs = obs
            self.current_action = None

    def get_dataloader(self,batch_size):
        sampler = SeedableRandomSampler(self.replay_buffer, generator=self.generator)
        dataloader = torch.utils.data.DataLoader(
                self.replay_buffer, batch_size=batch_size, sampler=sampler)
        return dataloader

    def train(self,batch_size=2,iterations=1):
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = self.get_dataloader(batch_size)
        gamma = self.discount_factor
        tau = self.polyak_rate
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        for i,(s,a1,r2,s2,t,m1) in zip(range(iterations),dataloader):
            # Fix data types
            s0 = s[0].to(self.device)
            s1 = s[-1].to(self.device)
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

        obs = self.get_current_obs(testing)

        # Sample an action
        if not testing and self.rand.rand() < self.behaviour_epsilon:
            action = self.rand.randint(self.action_space.n)
        else:
            action_probs = self.policy_net(*obs).squeeze().detach().numpy()
            #dist = torch.distributions.Categorical(action_probs)
            #action = dist.sample().item()
            action = self.rand.choice(len(action_probs),p=action_probs)
        self.current_action = action

        return action

    def get_current_obs(self,testing=False):
        """ Return the observation the agent needs to act on. This can be fed
        directly to the policy network.  """
        if testing:
            (obs0,obs1),mask = compute_mask(self.obs_stack_testing[0],self.current_obs_testing)
        else:
            (obs0,obs1),mask = compute_mask(self.obs_stack[0],self.current_obs)

        # Move to appropriate device
        if obs0 is not None:
            obs0 = obs0.to(self.device)
        obs1 = obs1.to(self.device)
        mask = mask.to(self.device)

        return obs0,obs1,mask

    def get_state_action_value(self, obs, action):
        vals = self.q_net(torch.tensor(obs).to(self.device).float())[action]
        if vals.size() == ():
            return vals.item()
        else:
            return vals[action].item()

    def test_once(self, env, render=False):
        """
        :return: Results of the test.
        :rtype: Dictionary with the following values:
            - total_rewards: Sum of all rewards obtained during the episode.
            - state_action_values: An average of the values of all actions
              taken by the agent, as estimated by its internal value function.
            - steps: Length of the episode.
        """
        reward_sum = 0
        sa_vals = []
        obs = env.reset()
        self.observe_change(obs,testing=True)
        for steps in itertools.count():
            action = self.act(testing=True)
            sa_vals.append(self.get_state_action_value(self.current_obs_testing,action))
            obs, reward, done, _ = env.step(action)
            self.observe_change(obs,testing=True)
            reward_sum += reward
            if render:
                env.render()
            if done:
                break
        return {
            'total_rewards': reward_sum,
            'state_action_values': np.mean(sa_vals),
            'steps': steps
        }

    def test(self, env, iterations, render=False, record=True, processors=1):
        return [self.test_once(env, render=render) for _ in range(iterations)]

    def state_dict(self):
        return {
                'behaviour_epsilon': self.behaviour_epsilon,
                'q_net': self.q_net.state_dict(),
                'q_net_target': self.q_net_target.state_dict(),
                'controller_net': self.controller_net.state_dict(),
                'controller_net_target': self.controller_net_target.state_dict(),
                'subpolicy_nets': [net.state_dict() for net in self.subpolicy_nets],
                'subpolicy_net_targets': [net.state_dict() for net in self.subpolicy_net_targets],
                'policy_net': self.policy_net.state_dict(),
                'policy_net_target': self.policy_net_target.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'replay_buffer': self.replay_buffer.state_dict(),
                'obs_stack': self.obs_stack,
                'current_obs': self.current_obs,
                'current_action': self.current_action,
                'current_terminal': self.current_terminal,
                'rand': self.rand.get_state(),
                'generator': self.generator.get_state(),
        }

    def load_state_dict(self, state):
        self.behaviour_epsilon = state['behaviour_epsilon']
        self.q_net.load_state_dict(state['q_net'])
        self.q_net_target.load_state_dict(state['q_net_target'])
        self.controller_net.load_state_dict(state['controller_net'])
        self.controller_net_target.load_state_dict(state['controller_net_target'])
        for net,s in zip(self.subpolicy_nets,state['subpolicy_nets']):
            net.load_state_dict(s)
        for net,s in zip(self.subpolicy_net_targets,state['subpolicy_net_targets']):
            net.load_state_dict(s)
        self.policy_net.load_state_dict(state['policy_net'])
        self.policy_net_target.load_state_dict(state['policy_net_target'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        self.replay_buffer.load_state_dict(state['replay_buffer'])

        self.obs_stack = copy.deepcopy(state['obs_stack'])
        self.current_obs = state['current_obs']
        self.current_action = state['current_action']
        self.current_terminal = state['current_terminal']

        self.rand.set_state(state['rand'])
        self.generator.set_state(state['generator'])

class HDQNAgentWithDelayAC_v2(HDQNAgentWithDelayAC):
    def __init__(self, subpolicy_q_net_learning_rate=1e-3, **kwargs):
        super().__init__(**kwargs)

        learning_rate = subpolicy_q_net_learning_rate
        self.subpolicy_q_nets = [copy.deepcopy(self.q_net) for _ in self.subpolicy_nets]

        params = []
        for net in self.subpolicy_q_nets:
            params += list(net.parameters())
        self.subpolicy_critic_optimizer = torch.optim.Adam(params, lr=learning_rate)

    def train(self,batch_size=2,iterations=1):
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = self.get_dataloader(batch_size)
        gamma = self.discount_factor
        tau = self.polyak_rate
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        subpolicy_critic_optimizer = self.subpolicy_critic_optimizer
        for i,(s,a1,r2,s2,t,m1) in zip(range(iterations),dataloader):
            # Fix data types
            s0 = s[0].to(self.device)
            s1 = s[-1].to(self.device)
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

            # Update subpolicy Q functions
            subpolicy_critic_optimizer.zero_grad()
            subpolicy_loss_total = 0
            q_s2 = self.q_net_target(s2)
            for q_net,p_net in zip(self.subpolicy_q_nets,self.subpolicy_nets):
                action_probs = p_net(s1)
                next_state_vals = (action_probs * q_s2).sum(1)
                val_target = r2+gamma*next_state_vals*(1-t)
                val_pred = q_net(s1)[range(batch_size),a1.squeeze()]
                loss = ((val_target-val_pred)**2).mean()
                subpolicy_loss_total += loss
            subpolicy_loss_total.backward()
            subpolicy_critic_optimizer.step()

            # Update policy function
            num_subpolicies = len(self.subpolicy_nets)
            num_actions = self.action_space.n
            total_action_vals = 0
            sub_prob0 = self.controller_net(s0).view(-1,num_subpolicies,1) # batch * subpolicy * 1
            sub_prob1 = self.controller_net(s1).view(-1,num_subpolicies,1) # batch * subpolicy * 1
            vals = torch.empty([batch_size,num_subpolicies,num_subpolicies]) # batch * subpolicy * subpolicy
            vals = vals.to(self.device)
            for om0,om1 in itertools.product(range(num_subpolicies),range(num_subpolicies)):
                # pi(om0|s0)*pi_om0(a|s1)*Q_om1(a|s1)
                val = self.subpolicy_nets[om0](s1)*self.subpolicy_q_nets[om1](s1) # batch * actions
                val = val.sum(dim=1) # batch
                vals[:,om0,om1] = val
            actor_loss = -((sub_prob0 @ sub_prob1.permute(0,2,1))*vals).sum()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Update target weights
            params = [zip(self.q_net_target.parameters(), self.q_net.parameters())]
            params += [zip(self.controller_net_target.parameters(), self.controller_net.parameters())]
            for net1,net2 in zip(self.subpolicy_net_targets,self.subpolicy_nets):
                params.append(zip(net1.parameters(),net2.parameters()))
            for p1,p2 in itertools.chain.from_iterable(params):
                p1.data = (1-tau)*p1+tau*p2

    def state_dict(self):
        d = super().state_dict()
        d['subpolicy_q_nets'] = [net.state_dict() for net in self.subpolicy_q_nets]
        d['subpolicy_critic_optimizer'] = self.subpolicy_critic_optimizer.state_dict()
        return d

    def load_state_dict(self, state):
        super().load_state_dict(state)
        for net,s in zip(self.subpolicy_q_nets,state['subpolicy_q_nets']):
            net.load_state_dict(s)
        self.subpolicy_critic_optimizer.load_state_dict(state['subpolicy_critic_optimizer'])

def compute_mask_augmented_state(obs0,action0,obs1):
    mask = torch.tensor([[obs0 is not None, obs1 is not None]]).float()
    def o_to_t(o):
        """ obs to tensor """
        if o is None:
            return torch.zeros(obs1.shape).float().squeeze().unsqueeze(0)
        return torch.tensor(o).float().squeeze().unsqueeze(0)
    def a_to_t(a):
        """ action to tensor """
        if a is None:
            return torch.zeros([1,1]).long()
        return torch.tensor(a).long().unsqueeze(0)
    return (o_to_t(obs0),a_to_t(action0),o_to_t(obs1)), mask

class HDQNAgentWithDelayAC_v3(HDQNAgentWithDelayAC_v2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.replay_buffer = ReplayBufferStackedObsAction(kwargs['replay_buffer_size'])

        self.policy_net = HierarchicalPolicyNetworkAugmentedState(
                self.controller_net, self.subpolicy_nets)
        self.policy_net_target = HierarchicalPolicyNetworkAugmentedState(
                self.controller_net_target, self.subpolicy_net_targets)

        self.reset_obs_stack(testing=True)
        self.reset_obs_stack(testing=False)

    def reset_obs_stack(self, testing=False):
        if testing:
            self.obs_stack_testing.clear()
            self.obs_stack_testing.append((None,None))
        else:
            self.obs_stack.clear()
            self.obs_stack.append((None,None))

    def train(self,batch_size=2,iterations=1):
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = self.get_dataloader(batch_size)
        gamma = self.discount_factor
        tau = self.polyak_rate
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        subpolicy_critic_optimizer = self.subpolicy_critic_optimizer
        for i,(s0,a0,s1,a1,r2,s2,t,m1) in zip(range(iterations),dataloader):
            # Fix data types
            s0 = s0.to(self.device)
            a0 = a0.to(self.device)
            s1 = s1.to(self.device)
            a1 = a1.to(self.device)
            r2 = r2.float().to(self.device)
            s2 = s2.to(self.device)
            t = t.float().to(self.device)
            m1 = m1.to(self.device)
            m2 = torch.ones_like(m1).to(self.device)
            
            # Update Q function
            action_probs = self.policy_net_target(s0,a0,s1,m1)
            next_state_vals = (action_probs * self.q_net_target(s2)).sum(1)
            val_target = r2+gamma*next_state_vals*(1-t)

            val_pred = self.q_net(s1)[range(batch_size),a1.squeeze()]

            critic_optimizer.zero_grad()
            critic_loss = ((val_target-val_pred)**2).mean()
            critic_loss.backward()
            critic_optimizer.step()

            # Update subpolicy Q functions
            subpolicy_critic_optimizer.zero_grad()
            subpolicy_loss_total = 0
            q_s2 = self.q_net_target(s2)
            for q_net,p_net in zip(self.subpolicy_q_nets,self.subpolicy_nets):
                action_probs = p_net(s1)
                next_state_vals = (action_probs * q_s2).sum(1)
                val_target = r2+gamma*next_state_vals*(1-t)
                val_pred = q_net(s1)[range(batch_size),a1.squeeze()]
                loss = ((val_target-val_pred)**2).mean()
                subpolicy_loss_total += loss
            subpolicy_loss_total.backward()
            subpolicy_critic_optimizer.step()

            # Update policy function
            num_subpolicies = len(self.subpolicy_nets)
            num_actions = self.action_space.n
            total_action_vals = 0
            sub_prob0 = self.controller_net(s0,a0).view(-1,num_subpolicies,1) # batch * subpolicy * 1
            sub_prob1 = self.controller_net(s1,a1).view(-1,num_subpolicies,1) # batch * subpolicy * 1
            vals = torch.empty([batch_size,num_subpolicies,num_subpolicies]) # batch * subpolicy * subpolicy
            for om0,om1 in itertools.product(range(num_subpolicies),range(num_subpolicies)):
                # pi(om0|s0)*pi_om0(a|s1)*Q_om1(a|s1)
                val = self.subpolicy_nets[om0](s1)*self.subpolicy_q_nets[om1](s1) # batch * actions
                val = val.sum(dim=1) # batch
                vals[:,om0,om1] = val
            actor_loss = -((sub_prob0 @ sub_prob1.permute(0,2,1))*vals).sum()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Update target weights
            params = [zip(self.q_net_target.parameters(), self.q_net.parameters())]
            params += [zip(self.controller_net_target.parameters(), self.controller_net.parameters())]
            for net1,net2 in zip(self.subpolicy_net_targets,self.subpolicy_nets):
                params.append(zip(net1.parameters(),net2.parameters()))
            for p1,p2 in itertools.chain.from_iterable(params):
                p1.data = (1-tau)*p1+tau*p2

    def observe_step(self,obs0,action0,obs1,action1,reward1,obs2,terminal=False):
        if obs0 is None:
            obs0 = torch.zeros(obs1.shape).squeeze()
        else:
            obs0 = torch.Tensor(obs0).squeeze()
        obs1 = torch.Tensor(obs1).squeeze()
        obs2 = torch.Tensor(obs2).squeeze()
        self.replay_buffer.add_transition(obs0,action0,obs1,action1,reward1,obs2,terminal=terminal)

    def observe_change(self, obs, reward=None, terminal=False, testing=False):
        if testing:
            # Reward is None if this is the first step of the episode
            if reward is None:
                self.reset_obs_stack(testing=testing)
            self.current_obs_testing = obs
        else:
            # Reward is None if this is the first step of the episode
            if reward is None:
                self.reset_obs_stack(testing=testing)
            else:
                if self.obs_stack[0][0] is not None:
                    self.observe_step(*self.obs_stack[0], *self.obs_stack[1],
                            reward, obs, terminal)
            self.current_obs = obs
            self.current_action = None

    def act(self, testing=False):
        """Return a random action according to the current behaviour policy"""

        obs0,action0,obs1,mask = self.get_current_obs(testing)

        # Sample an action
        if not testing and self.rand.rand() < self.behaviour_epsilon:
            action = self.rand.randint(self.action_space.n)
        else:
            action_probs = self.policy_net(obs0,action0,obs1,mask).squeeze().detach().numpy()
            action = self.rand.choice(len(action_probs),p=action_probs)
        self.current_action = action

        if testing:
            self.obs_stack_testing.append((obs1,action))
        else:
            self.obs_stack.append((obs1,action))
        return action

    def get_current_obs(self,testing=False):
        """ Return the observation the agent needs to act on. This can be fed
        directly to the policy network.  """
        if testing:
            obs_stack = self.obs_stack_testing
            current_obs = self.current_obs_testing
        else:
            obs_stack = self.obs_stack
            current_obs = self.current_obs

        (obs0,action0,obs1),mask = compute_mask_augmented_state(
                *obs_stack[-1], current_obs)

        # Move to appropriate device
        if obs0 is not None:
            obs0 = obs0.to(self.device)
        obs1 = obs1.to(self.device)
        mask = mask.to(self.device)

        return obs0,action0,obs1,mask
