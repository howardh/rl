from collections import deque, defaultdict
import numpy as np
from tqdm import tqdm
import torch
import gym
import copy
import itertools

import numpy as np

from agent.agent import Agent
from .replay_buffer import ReplayBuffer,ReplayBufferStackedObs,ReplayBufferStackedObsAction
from .policy import get_greedy_epsilon_policy, greedy_action

import utils

class HierarchicalQNetwork(torch.nn.Module):
    def __init__(self, controller, subpolicies):
        super().__init__()
        self.controller = controller
        self.subpolicies = subpolicies
        self.softmax = torch.nn.Softmax(dim=1)

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
        self.softmax = torch.nn.Softmax(dim=1)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, obs0, obs1, obs_mask, controller_temperature=1, subpolicy_temperature=1, extras=False):
        # Values of each subpolicy
        controller_output = self.controller(obs0)
        # If there's no observation available, assign the same value to all
        # subpolicies
        uniform_controller = torch.ones_like(controller_output)
        mask = obs_mask[:,0].view(-1,1)
        controller_output = mask*controller_output+(1-mask)*uniform_controller
        
        # Compute overall probability of primitive action as a weighted sum, weighted by
        # probability of taking each subpolicy
        subpolicy_probabilities = self.softmax(controller_output/controller_temperature).unsqueeze(1)
        primitive_action_probabilities = torch.stack([sp(obs1,subpolicy_temperature) for sp in self.subpolicies], dim=1)
        action_probs = subpolicy_probabilities @ primitive_action_probabilities
        action_probs = action_probs.squeeze(1)

        if extras:
            return action_probs, {
                    'controller_log_probs': self.log_softmax(controller_output),
                    'subpolicy_probabilities': subpolicy_probabilities,
                    'primitive_action_probabilities': primitive_action_probabilities
            }
        return action_probs

class HierarchicalPolicyNetworkAugmentedState(torch.nn.Module):
    def __init__(self, controller, subpolicies):
        super().__init__()
        self.controller = controller
        self.subpolicies = subpolicies
        self.softmax = torch.nn.Softmax(dim=1)

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
            def foo():
                while True:
                    yield torch.randint(high=n, size=(1,), dtype=torch.int64, generator=self.generator).item()
            return foo()
        def foo():
            for x in torch.randperm(n,generator=self.generator):
                yield x
        return foo()

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
        self.subpolicy_learning_rate = subpolicy_learning_rate
        self.controller_learning_rate = controller_learning_rate

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
        #self.actor_optimizer = torch.optim.Adam([
        #    {'params': controller_params, 'lr': controller_learning_rate},
        #    {'params': subpolicy_params, 'lr': subpolicy_learning_rate}])
        #self.critic_optimizer = torch.optim.Adam(
        #        self.q_net.parameters(), lr=q_net_learning_rate)
        self.actor_optimizer = torch.optim.SGD([
            {'params': controller_params, 'lr': controller_learning_rate, 'momentum': 0},
            {'params': subpolicy_params, 'lr': subpolicy_learning_rate, 'momentum': 0}])
        self.critic_optimizer = torch.optim.SGD(
                self.q_net.parameters(), lr=q_net_learning_rate, momentum=0)

        self.to(device)

        self.controller_dropout = None

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
                for _ in range(self.obs_stack_testing.maxlen):
                    self.obs_stack_testing.append(None)
            if self.controller_dropout is not None and self.rand.random() < self.controller_dropout and len(self.obs_stack_testing) == self.obs_stack_testing.maxlen:
                self.obs_stack_testing.append(self.obs_stack_testing[-1])
            else:
                self.obs_stack_testing.append(obs)
            self.current_obs_testing = obs
        else:
            if reward is None: # Reward is None if this is the first step of the episode
                self.obs_stack.clear()
                for _ in range(self.obs_stack.maxlen):
                    self.obs_stack.append(None)
            else:
                self.observe_step(self.current_obs, self.current_action, reward, obs, terminal)
            if self.controller_dropout is not None and self.rand.random() < self.controller_dropout and len(self.obs_stack) == self.obs_stack.maxlen:
                self.obs_stack.append(self.obs_stack[-1])
            else:
                self.obs_stack.append(obs)
            self.current_obs = obs
            self.current_action = None

    def get_dataloader(self,batch_size,replacement=False):
        sampler = SeedableRandomSampler(self.replay_buffer, generator=self.generator, replacement=replacement)
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
            sa_vals.append(self.get_state_action_value(self.obs_stack_testing.get(0,0),action))
            obs, reward, done, _ = env.step(action)
            self.observe_change(obs,reward,testing=True)
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
            stack = self.obs_stack_testing
        else:
            stack = self.obs_stack
        stack.clear()
        for _ in range(stack.maxlen):
            stack.append((None,None))

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
                    self.observe_step(*self.obs_stack[0], *self.obs_stack[-1],
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
            if self.controller_dropout is not None and self.rand.random() < self.controller_dropout and len(self.obs_stack_testing) == self.obs_stack_testing.maxlen:
                self.obs_stack_testing.append(self.obs_stack_testing[-1])
            else:
                self.obs_stack_testing.append((obs1,action))
        else:
            if self.controller_dropout is not None and self.rand.random() < self.controller_dropout and len(self.obs_stack) == self.obs_stack.maxlen:
                self.obs_stack.append(self.obs_stack[-1])
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

def create_augmented_obs_transform_one_hot_action(action_space_size):
    """ Return a function that flattens the observation, encodes the actions in a one-hot vector, and concatenates everything. """
    def to_one_hot(action):
        a = np.zeros([action_space_size])
        a[action] = 1
        return a
    def transform(state, actions):
        flat_state = state.flatten()
        if len(actions) == 0:
            return flat_state
        else:
            one_hot_actions = np.concatenate([to_one_hot(a) for a in actions])
            return np.concatenate([flat_state, one_hot_actions])
    return transform

def default_augmented_obs_transform(state,actions):
    """ Return a function that flattens the observation, and concatenates the action as is. """
    flat_state = state.flatten()
    if len(actions) == 0:
        return flat_state
    else:
        actions = np.concatenate([np.array([a]) for a in actions])
        return np.concatenate([flat_state, actions])

class AugmentedObservationStack():
    """ An object which keeps track of the most recent state-action pairs in order to produce a observations consisting of an old state plus a sequence of actions taken starting from that state.
    Elements are stored in order from newest to oldest (i.e. index 0 is the most recent observation).

    The AugmentedObservationStack can be in one of two states:
    1. Observations and actions are in sync (i.e. we've seen the same number of observations as actions)
    2. Observations and actions are out of sync (i.e. we've seen the observation for the current step, but not the action)

    In both cases, the most recently added observation is treated as the current time step.
    >>> aos = AugmentedObservationStack(5)
    >>> aos.append_obs(np.array([0]))
    >>> aos.get(0,0)
    np.array([0])
    >>> aos.append_action(10)
    >>> aos.get(0,0)
    np.array([0])
    >>> aos.append_obs(np.array([1]))
    >>> aos.get(0,0)
    np.array([1])
    >>> aos.get(0,1)
    np.array([0,10])
    >>> aos.append_action(11)
    >>> aos.get(0,1)
    np.array([1,11])

    Visual examples below. The grid is a representation of the observations and actions in our trajectory, with the most recent observations and actions at the right.
    `.` = State whose representation is to be returned (Omitted if it overlaps with `x`)
    `x` = Values that are returned

    `aos.get(0,0)`
        +-+-+-+-+    +-+-+-+-+
    obs | | | |x|    | | | |x|
        +-+-+-+-+ or +-+-+-+-+
    act | | | | |    | | | | 
        +-+-+-+-+    +-+-+-+

    `aos.get(0,1)`
        +-+-+-+-+    +-+-+-+-+
    obs | | |x|.|    | | |x|.|
        +-+-+-+-+ or +-+-+-+-+
    act | | |x| |    | | |x| 
        +-+-+-+-+    +-+-+-+

    `aos.get(1,0)`
        +-+-+-+-+    +-+-+-+-+
    obs | | |x| |    | | |x| |
        +-+-+-+-+ or +-+-+-+-+
    act | | | | |    | | | | 
        +-+-+-+-+    +-+-+-+

    `aos.get(1,1)`
        +-+-+-+-+    +-+-+-+-+
    obs | |x|.| |    | |x|.| |
        +-+-+-+-+ or +-+-+-+-+
    act | |x| | |    | |x| | 
        +-+-+-+-+    +-+-+-+

    `aos.get(1,2)`
        +-+-+-+-+    +-+-+-+-+
    obs |x| |.| |    |x| |.| |
        +-+-+-+-+ or +-+-+-+-+
    act |x|x| | |    |x|x| | 
        +-+-+-+-+    +-+-+-+
    """
    def __init__(self, stack_len=1, action_len=0, transform=None):
        """
        Args:
            stack_len: Number of observation-action pairs to keep track of.
            action_len: Number of actions with which to augment the state.
        """
        self.observations = deque(maxlen=stack_len)
        self.actions = deque(maxlen=stack_len)
        self.action_len = action_len

        # Keep track of how many observations and actions were added.
        # This is used to make sure that we return matching obs-action pairs.
        self.num_observations = 0
        self.num_actions = 0

        if transform is None:
            self.transform = default_augmented_obs_transform
        else:
            self.transform = transform

    def append_obs(self, obs):
        self.observations.append(obs)
        self.num_observations += 1

    def append_action(self, action):
        self.actions.append(action)
        self.num_actions += 1

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.num_observations = 0
        self.num_actions = 0

    def get(self, delay, action_len):
        if action_len == 0:
            index = len(self.observations)-delay-1
            if index < 0:
                return None
            return self.transform(self.observations[index],[])
        else:
            tdiff = self.num_observations-self.num_actions if len(self.actions) == self.actions.maxlen else 0
            index = len(self.observations)-delay-action_len-1
            if index < 0:
                return None
            obs = self.observations[index]

            if len(self.actions) < action_len:
                return None
            else:
                # self.actions is a deque, so can't use slices. Need itertools for this.
                actions = list(itertools.islice(self.actions, index+tdiff, index+action_len+tdiff))
            return self.transform(obs,actions)

    def get_action(self, delay):
        """ Get the raw action that was taken at time `t-delay` where `t` is the current time step. """
        tdiff = self.num_observations - self.num_actions
        index = len(self.actions)-1-delay+tdiff
        if index >= len(self.actions) or index < 0:
            return None
        return self.actions[index]

    def __getitem__(self, index):
        return self.get(index, self.action_len)

class HRLAgent_v4(HDQNAgentWithDelayAC):
    def __init__(self, behaviour_temp=1, target_temp=1, action_mem=0, ac_variant='advantage', algorithm='actor-critic',
            l2_weight=1e-3, entropy_weight=1e-2,
            **kwargs):
        """
        Args:
            subpolicy_q_net_learning_rate: Learning rate for the subpolicy Q network.
                If `None`, then no separate Q network is used for the subpolicies.
            action_mem: The number of actions to use when augmenting outdated states.
                Cannot exceed the delay.
            algorithm: 'q-learning', 'actor-critic', 'actor-critic-v2'
                q-learning: Value-based learning using off-policy Q learning.
        """
        super().__init__(**kwargs) 
        self.behaviour_temp = behaviour_temp
        self.target_temp = target_temp
        self.algorithm = algorithm
        self.l2_weight = l2_weight
        self.entropy_weight = entropy_weight

        assert action_mem <= kwargs['delay_steps']
        self.action_mem = action_mem
        self.ac_variant = ac_variant
        self.delay = kwargs['delay_steps']

        self.replay_buffer = ReplayBuffer(kwargs['replay_buffer_size'])

        obs_stack_transform = create_augmented_obs_transform_one_hot_action(4)
        # Stack size: Consider delay=0. stack_len=0 can't store anything. stack_len=1 only stores the current observation. stack_len=2 can hold current obs and previous obs, which is needed to create the replay buffer.
        self.obs_stack = AugmentedObservationStack(transform=obs_stack_transform,
                stack_len=self.delay+2, action_len=self.action_mem)
        self.obs_stack_testing = AugmentedObservationStack(transform=obs_stack_transform,
                stack_len=self.delay+2, action_len=self.action_mem)

        self.controller_dropout = None
        self.controller_obs = [None,None] # Dropout obs. Index 0 = training, index 1 = testing

        num_options = len(self.subpolicy_nets)
        self.action_counts = [[0]*4,[0]*4]
        self.option_counts = [[0]*num_options,[0]*num_options]
        self.total_grad = [[] for _ in range(num_options)] # gradient of each subpolicy
        self.state_values_1 = []
        self.state_values_2 = []
        self.debug = defaultdict(lambda: [])

    def train(self,batch_size=2,iterations=1):
        if self.algorithm == 'q-learning':
            self.train_q_learning(batch_size,iterations)
        elif self.algorithm == 'actor-critic':
            self.train_actor_critic(batch_size,iterations)
        elif self.algorithm == 'actor-critic-v2':
            self.train_actor_critic_v2(batch_size,iterations)
        elif self.algorithm == 'actor-critic-v3':
            self.train_actor_critic_v3(batch_size,iterations)
        elif self.algorithm == 'actor-critic-v4':
            self.train_actor_critic_v4(batch_size,iterations)
        elif self.algorithm == 'actor-critic-v5':
            self.train_actor_critic_v5(batch_size,iterations)
        elif self.algorithm == 'actor-critic-v6':
            self.train_actor_critic_v6(batch_size,iterations)
        elif self.algorithm == 'actor-critic-v7':
            self.train_actor_critic_v7(batch_size,iterations)
        elif self.algorithm == 'actor-critic-v8':
            self.train_actor_critic_v8(batch_size,iterations)
        elif self.algorithm == 'actor-critic-v9':
            self.train_actor_critic_v9(batch_size,iterations)
        elif self.algorithm == 'actor-critic-v10':
            self.train_actor_critic_v10(batch_size,iterations)
        elif self.algorithm == 'actor-critic-v11':
            self.train_actor_critic_v11(batch_size,iterations)
    def train_actor_critic(self,batch_size=2,iterations=1):
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = self.get_dataloader(batch_size)
        gamma = self.discount_factor
        tau = self.polyak_rate
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        for i,((s0a,s0,m0),a0,r1,(s1a,s1,m1),t) in zip(range(iterations),dataloader):
            # Fix data types
            s0a = s0a.float().to(self.device)   # State at which the subpolicy was chosen (augmented)
            s0 = s0.float().to(self.device)     # State at which the subpolicy was chosen (not augmented)
            m0 = m0.float().to(self.device)     # 1 if s0a exists, and 0 otherwise
            a0 = a0.to(self.device)             # Action drawn from chosen subpolicy
            r1 = r1.float().to(self.device)     # Reward obtained for taking action a1 at state s1
            s1a = s1a.float().to(self.device)   # State on which the subpolicy was applied (augmented)
            s1 = s1.float().to(self.device)     # State on which the subpolicy was applied (not augmented)
            m1 = m1.float().to(self.device)     # 1 if s1a exists, and 0 otherwise
            t = t.float().to(self.device)       # 1 if s1 is a terminal state, 0 otherwise
            temp = self.target_temp             # Temperature
            
            # Update Q function
            a1_probs = self.policy_net_target(s1a,s1,m1,temp,temp) # Action probs at s1
            v1_pred = (a1_probs * self.q_net_target(s1)).sum(1) # Expected value of s1 under current policy
            qs0a0_target = r1+gamma*v1_pred*(1-t) # Value of (s0,a0) from sampled reward and bootstrapping

            a0_probs = self.policy_net_target(s0a,s0,m0,temp,temp) # Action probs at s0
            v0_pred = (a0_probs * self.q_net_target(s0)).sum(1) # Expected value of s0 under current policy
            q0_pred = self.q_net(s0) # Predicted state-action values at s0
            qs0a0_pred = q0_pred[range(batch_size),a0.squeeze()] # Predicted state-action value of (s0,a0)

            critic_optimizer.zero_grad()
            critic_loss = ((qs0a0_target-qs0a0_pred)**2).mean()
            critic_loss.backward()
            critic_optimizer.step()

            # Update policy
            actor_optimizer.zero_grad()

            a0_probs, extras = self.policy_net(s0a,s0,m0,temp,temp,extras=True) # Action probs at s0
            q0 = self.q_net_target(s0) # batch size * # actions
            primitive_probs = [sp(s1,temp) for sp in self.subpolicy_nets]
            # Train subpolicies to maximize expected return, weighted by the probability of the controller choosing that subpolicy
            subpolicy_log_probs = [sp(s0,temp,log=True) for sp in self.subpolicy_nets] # Log action prob for each subpolicy
            for log_prob in subpolicy_log_probs:
                # log_prob.shape = batch size * # actions
                action_probs = torch.nn.functional.softmax(log_prob, dim=1).detach() # batch size * # actions
                delta = None # batch size * # actions
                if self.ac_variant == 'advantage':
                    delta = (q0-action_probs*q0).detach()
                elif self.ac_variant == 'q':
                    delta = (action_probs*q0).detach()
                actor_loss = action_probs*log_prob*delta
                actor_loss = actor_loss.sum(1)
                actor_loss = actor_loss.mean()
                actor_loss.backward() # Accumulate gradients
            # Train controller to favour subpolicies with higher expected return at this state
            q0_subpolicies = [torch.nn.functional.softmax(splp, dim=1)*q0_pred for splp in subpolicy_log_probs] # Expected Q value of each subpolicy
            controller_log_probs = extras['controller_output'] # batch size * # options
            option_q0 = torch.stack([(torch.nn.functional.softmax(log_prob, dim=1)*q0).sum(1) for log_prob in subpolicy_log_probs],dim=1) # batch size * # options
            option_probs = torch.nn.functional.softmax(controller_log_probs, dim=1).detach()
            delta = None
            if self.ac_variant == 'advantage':
                delta = (option_q0-option_probs*option_q0).detach()
            elif self.ac_variant == 'q':
                delta = (option_probs*option_q0).detach()
            actor_loss = option_probs*controller_log_probs*delta
            actor_loss = actor_loss.sum(1)
            actor_loss = actor_loss.mean()
            actor_loss.backward() # Accumulate gradients

            #actor_optimizer.zero_grad()
            #actor_loss.backward()
            actor_optimizer.step()

            # Update target weights
            params = [zip(self.q_net_target.parameters(), self.q_net.parameters())]
            params += [zip(self.controller_net_target.parameters(), self.controller_net.parameters())]
            for p1,p2 in itertools.chain.from_iterable(params):
                p1.data = (1-tau)*p1+tau*p2
    def train_q_learning(self,batch_size=2,iterations=1):
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = self.get_dataloader(batch_size)
        gamma = self.discount_factor
        tau = self.polyak_rate
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        for i,((s0a,s0,m0),a0,r1,(s1a,s1,m1),t) in zip(range(iterations),dataloader):
            # Fix data types
            s0a = s0a.float().to(self.device)   # State at which the subpolicy was chosen (augmented)
            s0 = s0.float().to(self.device)     # State at which the subpolicy was chosen (not augmented)
            m0 = m0.float().to(self.device)     # 1 if s0a exists, and 0 otherwise
            a0 = a0.to(self.device)             # Action drawn from chosen subpolicy
            r1 = r1.float().to(self.device)     # Reward obtained for taking action a1 at state s1
            s1a = s1a.float().to(self.device)   # State on which the subpolicy was applied (augmented)
            s1 = s1.float().to(self.device)     # State on which the subpolicy was applied (not augmented)
            m1 = m1.float().to(self.device)     # 1 if s1a exists, and 0 otherwise
            t = t.float().to(self.device)       # 1 if s1 is a terminal state, 0 otherwise
            temp = self.target_temp             # Temperature
            batch_size = s0a.shape[0]
            
            # Update Q function
            qs0a0_pred = self.q_net(s0)[range(batch_size),a0.squeeze()]
            qs0a0_target = (r1+gamma*self.q_net_target(s1).max(1)[0]*(1-t)).detach()
            critic_optimizer.zero_grad()
            critic_loss = ((qs0a0_target-qs0a0_pred)**2).mean()
            critic_loss.backward()
            critic_optimizer.step()

            # Update target weights
            params = [zip(self.q_net_target.parameters(), self.q_net.parameters())]
            for p1,p2 in itertools.chain.from_iterable(params):
                p1.data = (1-tau)*p1+tau*p2
    def train_actor_critic_v2(self,batch_size=2,iterations=1):
        # Actor critic, with Q function learned off-policy
        # No hierarchy, and the policy is learned by supervised learning to match the greedy policy
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = self.get_dataloader(batch_size)
        gamma = self.discount_factor
        tau = self.polyak_rate
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        criterion = torch.nn.NLLLoss()
        for i,((s0a,s0,m0),a0,r1,(s1a,s1,m1),t) in zip(range(iterations),dataloader):
            # Fix data types
            s0a = s0a.float().to(self.device)   # State at which the subpolicy was chosen (augmented)
            s0 = s0.float().to(self.device)     # State at which the subpolicy was chosen (not augmented)
            m0 = m0.float().to(self.device)     # 1 if s0a exists, and 0 otherwise
            a0 = a0.to(self.device)             # Action drawn from chosen subpolicy
            r1 = r1.float().to(self.device)     # Reward obtained for taking action a1 at state s1
            s1a = s1a.float().to(self.device)   # State on which the subpolicy was applied (augmented)
            s1 = s1.float().to(self.device)     # State on which the subpolicy was applied (not augmented)
            m1 = m1.float().to(self.device)     # 1 if s1a exists, and 0 otherwise
            t = t.float().to(self.device)       # 1 if s1 is a terminal state, 0 otherwise
            temp = self.target_temp             # Temperature
            batch_size = s0a.shape[0]
            
            # Update Q function
            qs0a0_pred = self.q_net(s0)[range(batch_size),a0.squeeze()]
            qs0a0_target = (r1+gamma*self.q_net_target(s1).max(1)[0]*(1-t)).detach()
            critic_optimizer.zero_grad()
            critic_loss = ((qs0a0_target-qs0a0_pred)**2).mean()
            critic_loss.backward()
            critic_optimizer.step()

            # Update policy
            q0 = self.q_net(s0) # batch size * # actions
            self.state_values_1.append((q0*self.subpolicy_nets[0](s0,temp,log=False)).sum(1).mean().item())
            ideal_policy = torch.zeros_like(q0)
            ideal_policy[range(batch_size),q0.max(1)[1]]=1
            pol_before = self.subpolicy_nets[0](s0,temp,log=False)

            self.debug['policy_diff_1'].append(((self.subpolicy_nets[0](s0,temp,log=False)-ideal_policy)**2).sum(1).mean(0).item())

            actor_optimizer.zero_grad()
            actor_loss = criterion(self.subpolicy_nets[0](s0,temp,log=True),q0.max(1)[1])
            actor_loss.backward()
            actor_optimizer.step()
            actor_loss_after = criterion(self.subpolicy_nets[0](s0,temp,log=True),q0.max(1)[1])

            self.state_values_2.append((q0*self.subpolicy_nets[0](s0,temp,log=False)).sum(1).mean().item())
            self.debug['ideal_q'].append(q0.max(1)[0].mean(0).item())
            self.debug['q_diff_1'].append(self.debug['ideal_q'][-1]-self.state_values_1[-1])
            self.debug['q_diff_2'].append(self.debug['ideal_q'][-1]-self.state_values_2[-1])
            self.debug['q_diff_diff'].append(self.debug['q_diff_1'][-1]-self.debug['q_diff_2'][-1])

            # Update target weights
            params = [zip(self.q_net_target.parameters(), self.q_net.parameters())]
            for p1,p2 in itertools.chain.from_iterable(params):
                p1.data = (1-tau)*p1+tau*p2
    def train_actor_critic_v3(self,batch_size=2,iterations=1):
        # Actor critic, with Q function learned off-policy
        # No hierarchy, and the policy is learned via RL
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = self.get_dataloader(batch_size)
        gamma = self.discount_factor
        tau = self.polyak_rate
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        for i,((s0a,s0,m0),a0,r1,(s1a,s1,m1),t) in zip(range(iterations),dataloader):
            # Fix data types
            s0a = s0a.float().to(self.device)   # State at which the subpolicy was chosen (augmented)
            s0 = s0.float().to(self.device)     # State at which the subpolicy was chosen (not augmented)
            m0 = m0.float().to(self.device)     # 1 if s0a exists, and 0 otherwise
            a0 = a0.to(self.device)             # Action drawn from chosen subpolicy
            r1 = r1.float().to(self.device)     # Reward obtained for taking action a1 at state s1
            s1a = s1a.float().to(self.device)   # State on which the subpolicy was applied (augmented)
            s1 = s1.float().to(self.device)     # State on which the subpolicy was applied (not augmented)
            m1 = m1.float().to(self.device)     # 1 if s1a exists, and 0 otherwise
            t = t.float().to(self.device)       # 1 if s1 is a terminal state, 0 otherwise
            temp = self.target_temp             # Temperature
            batch_size = s0a.shape[0]
            
            # Update Q function
            qs0a0_pred = self.q_net(s0)[range(batch_size),a0.squeeze()]
            qs0a0_target = (r1+gamma*self.q_net_target(s1).max(1)[0]*(1-t)).detach()
            critic_optimizer.zero_grad()
            critic_loss = ((qs0a0_target-qs0a0_pred)**2).mean()
            critic_loss.backward()
            critic_optimizer.step()

            # Update policy
            actor_optimizer.zero_grad()

            q0 = self.q_net(s0) # batch size * # actions
            self.state_values_1.append((q0*self.subpolicy_nets[0](s0,temp,log=False)).sum(1).mean().item())
            log_probs = self.subpolicy_nets[0](s0,temp,log=True) # log_prob.shape = batch size * # actions
            action_probs = torch.exp(log_probs) # batch size * # actions
            delta = None # batch size * # actions
            if self.ac_variant == 'advantage':
                #delta = (q0-(action_probs*q0).sum(1).view(-1,1)).detach()
                delta = (qs0a0_pred-(action_probs*q0).sum(1)).detach()
            elif self.ac_variant == 'q':
                #delta = q0.detach()
                delta = qs0a0_pred.detach()
            #actor_loss = -log_probs*delta*(action_probs.detach())
            log_probs = log_probs[range(batch_size),a0.squeeze()]
            actor_loss = -log_probs*delta
            #actor_loss = actor_loss.sum(1)
            actor_loss = actor_loss.mean(0)
            actor_loss.backward()

            self.state_values_2.append((q0*self.subpolicy_nets[0](s0,temp,log=False)).sum(1).mean().item())

            actor_optimizer.step()

            # Update target weights
            params = [zip(self.q_net_target.parameters(), self.q_net.parameters())]
            for p1,p2 in itertools.chain.from_iterable(params):
                p1.data = (1-tau)*p1+tau*p2
    def train_actor_critic_v4(self,batch_size=2,iterations=1):
        # Actor critic, with Q function learned off-policy
        # With hierarchy
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = self.get_dataloader(batch_size)
        gamma = self.discount_factor
        tau = self.polyak_rate
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        for i,((s0a,s0,m0),a0,r1,(s1a,s1,m1),t) in zip(range(iterations),dataloader):
            # Fix data types
            s0a = s0a.float().to(self.device)   # State at which the subpolicy was chosen (augmented)
            s0 = s0.float().to(self.device)     # State at which the subpolicy was chosen (not augmented)
            m0 = m0.float().to(self.device)     # 1 if s0a exists, and 0 otherwise
            a0 = a0.to(self.device)             # Action drawn from chosen subpolicy
            r1 = r1.float().to(self.device)     # Reward obtained for taking action a1 at state s1
            s1a = s1a.float().to(self.device)   # State on which the subpolicy was applied (augmented)
            s1 = s1.float().to(self.device)     # State on which the subpolicy was applied (not augmented)
            m1 = m1.float().to(self.device)     # 1 if s1a exists, and 0 otherwise
            t = t.float().to(self.device)       # 1 if s1 is a terminal state, 0 otherwise
            temp = self.target_temp             # Temperature
            batch_size = s0a.shape[0]
            
            # Update Q function
            qs0a0_pred = self.q_net(s0)[range(batch_size),a0.squeeze()]
            qs0a0_target = (r1+gamma*self.q_net_target(s1).max(1)[0]*(1-t)).detach()
            critic_optimizer.zero_grad()
            critic_loss = ((qs0a0_target-qs0a0_pred)**2).mean()
            critic_loss.backward()
            critic_optimizer.step()

            # Update policy
            actor_optimizer.zero_grad()

            a0_probs, extras = self.policy_net(s0a,s0,m0,temp,temp,extras=True) # Action probs at s0
            q0 = self.q_net(s0) # batch size * # actions
            primitive_probs = [sp(s1,temp) for sp in self.subpolicy_nets]
            # Train subpolicies to maximize expected return, weighted by the probability of the controller choosing that subpolicy
            subpolicy_log_probs = [sp(s0,temp,log=True) for sp in self.subpolicy_nets] # Log action prob for each subpolicy
            controller_log_probs = extras['controller_log_probs'] # batch size * # options
            option_probs = torch.exp(controller_log_probs).detach()
            for sp_idx,log_probs in enumerate(subpolicy_log_probs):
                # log_prob.shape = batch size * # actions
                action_probs = torch.exp(log_probs) # batch size * # actions
                delta = None # batch size * # actions
                if self.ac_variant == 'advantage':
                    delta = (q0-(action_probs*q0).sum(1).view(-1,1)).detach()
                elif self.ac_variant == 'q':
                    delta = q0.detach()
                actor_loss = -log_probs*delta*(action_probs.detach())*option_probs[:,sp_idx].view(-1,1)
                actor_loss = actor_loss.sum(1)
                actor_loss = actor_loss.mean()
                actor_loss.backward() # Accumulate gradients
            # Train controller to favour subpolicies with higher expected return at this state
            option_q0 = torch.stack([(torch.exp(log_prob)*q0).sum(1) for log_prob in subpolicy_log_probs],dim=1) # batch size * # options
            delta = None
            if self.ac_variant == 'advantage':
                delta = (option_q0-(option_probs*option_q0).sum(1).view(-1,1)).detach()
            elif self.ac_variant == 'q':
                delta = option_q0.detach()
            actor_loss = -controller_log_probs*delta*(option_probs.detach())
            actor_loss = actor_loss.sum(1)
            actor_loss = actor_loss.mean()
            actor_loss.backward() # Accumulate gradients

            actor_optimizer.step()

            # Update target weights
            params = [zip(self.q_net_target.parameters(), self.q_net.parameters())]
            for p1,p2 in itertools.chain.from_iterable(params):
                p1.data = (1-tau)*p1+tau*p2
    def train_actor_critic_v5(self,batch_size=2,iterations=1):
        # Actor critic, with Q function learned off-policy
        # Optimize for expected return
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = self.get_dataloader(batch_size)
        gamma = self.discount_factor
        tau = self.polyak_rate
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        for i,((s0a,s0,m0),a0,r1,(s1a,s1,m1),t) in zip(range(iterations),dataloader):
            # Fix data types
            s0a = s0a.float().to(self.device)   # State at which the subpolicy was chosen (augmented)
            s0 = s0.float().to(self.device)     # State at which the subpolicy was chosen (not augmented)
            m0 = m0.float().to(self.device)     # 1 if s0a exists, and 0 otherwise
            a0 = a0.to(self.device)             # Action drawn from chosen subpolicy
            r1 = r1.float().to(self.device)     # Reward obtained for taking action a1 at state s1
            s1a = s1a.float().to(self.device)   # State on which the subpolicy was applied (augmented)
            s1 = s1.float().to(self.device)     # State on which the subpolicy was applied (not augmented)
            m1 = m1.float().to(self.device)     # 1 if s1a exists, and 0 otherwise
            t = t.float().to(self.device)       # 1 if s1 is a terminal state, 0 otherwise
            temp = self.target_temp             # Temperature
            batch_size = s0a.shape[0]
            
            # Update Q function
            qs0a0_pred = self.q_net(s0)[range(batch_size),a0.squeeze()]
            qs0a0_target = (r1+gamma*self.q_net_target(s1).max(1)[0]*(1-t)).detach()
            critic_optimizer.zero_grad()
            critic_loss = ((qs0a0_target-qs0a0_pred)**2).mean()
            critic_loss.backward()
            critic_optimizer.step()

            # Update policy
            actor_optimizer.zero_grad()

            q0 = self.q_net(s0) # batch size * # actions
            ideal_policy = torch.zeros_like(q0)
            ideal_policy[range(batch_size),q0.max(1)[1]]=1

            self.state_values_1.append((q0*self.subpolicy_nets[0](s0,temp,log=False)).sum(1).mean().item())
            log_probs = self.subpolicy_nets[0](s0,temp,log=True) # log_prob.shape = batch size * # actions
            action_probs = torch.exp(log_probs) # batch size * # actions
            self.debug['policy_diff_1'].append(((action_probs-ideal_policy)**2).sum(1).mean(0).item())
            actor_loss = -action_probs*(q0.detach())
            actor_loss = actor_loss.sum(1)
            actor_loss = actor_loss.mean(0)
            actor_loss.backward()

            self.state_values_2.append((q0*self.subpolicy_nets[0](s0,temp,log=False)).sum(1).mean().item())
            self.debug['ideal_q'].append(q0.max(1)[0].mean(0).item())
            self.debug['q_diff_1'].append(self.debug['ideal_q'][-1]-self.state_values_1[-1])
            self.debug['q_diff_2'].append(self.debug['ideal_q'][-1]-self.state_values_2[-1])
            self.debug['q_diff_diff'].append(self.debug['q_diff_1'][-1]-self.debug['q_diff_2'][-1])

            action_probs = self.subpolicy_nets[0](s0,temp,log=False) # log_prob.shape = batch size * # actions
            self.debug['policy_diff_2'].append(((action_probs-ideal_policy)**2).sum(1).mean(0).item())

            actor_optimizer.step()

            # Update target weights
            params = [zip(self.q_net_target.parameters(), self.q_net.parameters())]
            for p1,p2 in itertools.chain.from_iterable(params):
                p1.data = (1-tau)*p1+tau*p2
    def train_actor_critic_v6(self,batch_size=2,iterations=1):
        # Actor critic, with Q function learned off-policy
        # Optimize for expected return using advantage AC
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = self.get_dataloader(batch_size)
        gamma = self.discount_factor
        tau = self.polyak_rate
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        for i,((s0a,s0,m0),a0,r1,(s1a,s1,m1),t) in zip(range(iterations),dataloader):
            # Fix data types
            s0a = s0a.float().to(self.device)   # State at which the subpolicy was chosen (augmented)
            s0 = s0.float().to(self.device)     # State at which the subpolicy was chosen (not augmented)
            m0 = m0.float().to(self.device)     # 1 if s0a exists, and 0 otherwise
            a0 = a0.to(self.device)             # Action drawn from chosen subpolicy
            r1 = r1.float().to(self.device)     # Reward obtained for taking action a1 at state s1
            s1a = s1a.float().to(self.device)   # State on which the subpolicy was applied (augmented)
            s1 = s1.float().to(self.device)     # State on which the subpolicy was applied (not augmented)
            m1 = m1.float().to(self.device)     # 1 if s1a exists, and 0 otherwise
            t = t.float().to(self.device)       # 1 if s1 is a terminal state, 0 otherwise
            temp = self.target_temp             # Temperature
            batch_size = s0a.shape[0]
            
            # Update Q function
            qs0a0_pred = self.q_net(s0)[range(batch_size),a0.squeeze()]
            qs0a0_target = (r1+gamma*self.q_net_target(s1).max(1)[0]*(1-t)).detach()
            critic_optimizer.zero_grad()
            critic_loss = ((qs0a0_target-qs0a0_pred)**2).mean()
            critic_loss.backward()
            critic_optimizer.step()

            # Update policy
            actor_optimizer.zero_grad()

            q0 = self.q_net(s0).detach() # batch size * # actions
            ideal_policy = torch.zeros_like(q0)
            ideal_policy[range(batch_size),q0.max(1)[1]]=1
            self.debug['ideal_policy_entropy'].append(
                -(torch.log(ideal_policy.sum(0)/ideal_policy.sum())*ideal_policy.sum(0)/ideal_policy.sum()).sum().item()
            )

            self.state_values_1.append((q0*self.subpolicy_nets[0](s0,temp,log=False)).sum(1).mean().item())
            log_probs = self.subpolicy_nets[0](s0,temp,log=True) # log_prob.shape = batch size * # actions
            action_probs = torch.exp(log_probs) # batch size * # actions
            expected_q0 = (action_probs*q0).sum(1).view(-1,1).detach()

            self.debug['policy_diff_1'].append(((action_probs-ideal_policy)**2).sum(1).mean(0).item())
            self.debug['policy_entropy'].append(
                -(torch.log(action_probs.sum(0)/action_probs.sum())*action_probs.sum(0)/action_probs.sum()).nansum().item()
            )
            self.debug['policy_is_constant'].append((action_probs.sum(0)==0).sum().item()==3)

            actor_loss = -log_probs*(q0-expected_q0)
            #actor_loss = actor_loss.sum(1)
            actor_loss = actor_loss[range(batch_size),a0.squeeze()]
            actor_loss = actor_loss.mean(0)

            l2_loss = 0
            l2_count = 0
            l2_weight = self.l2_weight
            for p in self.subpolicy_nets[0].parameters():
                l2_loss += (p**2).sum()
                l2_count += p.flatten().shape[0]
            l2_loss /= l2_count
            actor_loss += l2_weight*l2_loss
            self.debug['l2_loss'].append(l2_loss.item())

            actor_loss.backward()
            actor_optimizer.step()

            g = 0
            for p in self.subpolicy_nets[0].parameters():
                g += p.grad.abs().sum().item()
                #with torch.no_grad():
                #    p -= self.subpolicy_learning_rate*p.grad
            self.debug['grad_snet0'].append(g)

            #if g > 10000:
            #    breakpoint()

            self.state_values_2.append((q0*self.subpolicy_nets[0](s0,temp,log=False)).sum(1).mean().item())
            self.debug['ideal_q'].append(q0.max(1)[0].mean(0).item())
            self.debug['q_diff_1'].append(self.debug['ideal_q'][-1]-self.state_values_1[-1])
            self.debug['q_diff_2'].append(self.debug['ideal_q'][-1]-self.state_values_2[-1])
            self.debug['q_diff_diff'].append(self.debug['q_diff_1'][-1]-self.debug['q_diff_2'][-1])

            action_probs = self.subpolicy_nets[0](s0,temp,log=False) # log_prob.shape = batch size * # actions
            self.debug['policy_diff_2'].append(((action_probs-ideal_policy)**2).sum(1).mean(0).item())

            # Update target weights
            params = [zip(self.q_net_target.parameters(), self.q_net.parameters())]
            for p1,p2 in itertools.chain.from_iterable(params):
                p1.data = (1-tau)*p1+tau*p2
    def train_actor_critic_v7(self,batch_size=2,iterations=1):
        # Actor critic, with Q function learned off-policy
        # Optimize for expected return using advantage AC
        # The advantage is computed using the sample instead of using the Q function
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = self.get_dataloader(batch_size)
        gamma = self.discount_factor
        tau = self.polyak_rate
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        for i,((s0a,s0,m0),a0,r1,(s1a,s1,m1),t) in zip(range(iterations),dataloader):
            # Fix data types
            s0a = s0a.float().to(self.device)   # State at which the subpolicy was chosen (augmented)
            s0 = s0.float().to(self.device)     # State at which the subpolicy was chosen (not augmented)
            m0 = m0.float().to(self.device)     # 1 if s0a exists, and 0 otherwise
            a0 = a0.to(self.device)             # Action drawn from chosen subpolicy
            r1 = r1.float().to(self.device)     # Reward obtained for taking action a1 at state s1
            s1a = s1a.float().to(self.device)   # State on which the subpolicy was applied (augmented)
            s1 = s1.float().to(self.device)     # State on which the subpolicy was applied (not augmented)
            m1 = m1.float().to(self.device)     # 1 if s1a exists, and 0 otherwise
            t = t.float().to(self.device)       # 1 if s1 is a terminal state, 0 otherwise
            temp = self.target_temp             # Temperature
            batch_size = s0a.shape[0]
            
            # Update Q function
            qs0a0_pred = self.q_net(s0)[range(batch_size),a0.squeeze()]
            qs0a0_target = (r1+gamma*self.q_net_target(s1).max(1)[0]*(1-t)).detach()
            critic_optimizer.zero_grad()
            critic_loss = ((qs0a0_target-qs0a0_pred)**2).mean()
            critic_loss.backward()
            critic_optimizer.step()

            # Update policy
            actor_optimizer.zero_grad()

            q0 = self.q_net(s0).detach() # batch size * # actions
            ideal_policy = torch.zeros_like(q0)
            ideal_policy[range(batch_size),q0.max(1)[1]]=1
            self.debug['ideal_policy_entropy'].append(
                -(torch.log(ideal_policy.sum(0)/ideal_policy.sum())*ideal_policy.sum(0)/ideal_policy.sum()).sum().item()
            )

            self.state_values_1.append((q0*self.subpolicy_nets[0](s0,temp,log=False)).sum(1).mean().item())
            log_probs = self.subpolicy_nets[0](s0,temp,log=True) # log_prob.shape = batch size * # actions
            action_probs = torch.exp(log_probs) # batch size * # actions
            expected_q0 = (action_probs*q0).sum(1).view(-1,1).detach()

            self.debug['policy_diff_1'].append(((action_probs-ideal_policy)**2).sum(1).mean(0).item())
            self.debug['policy_entropy'].append(
                -(torch.log(action_probs.sum(0)/action_probs.sum())*action_probs.sum(0)/action_probs.sum()).nansum().item()
            )
            self.debug['policy_is_constant'].append((action_probs.sum(0)==0).sum().item()==3)

            actor_loss = -log_probs[range(batch_size),a0.squeeze()]*(qs0a0_target-expected_q0.squeeze())
            actor_loss = actor_loss.mean(0)

            l2_loss = 0
            l2_count = 0
            l2_weight = self.l2_weight
            for p in self.subpolicy_nets[0].parameters():
                l2_loss += (p**2).sum()
                l2_count += p.flatten().shape[0]
            l2_loss /= l2_count
            actor_loss += l2_weight*l2_loss
            self.debug['l2_loss'].append(l2_loss.item())

            actor_loss.backward()
            actor_optimizer.step()

            g = 0
            for p in self.subpolicy_nets[0].parameters():
                g += p.grad.abs().sum().item()
                #with torch.no_grad():
                #    p -= self.subpolicy_learning_rate*p.grad
            self.debug['grad_snet0'].append(g)

            #if g > 10000:
            #    breakpoint()

            self.state_values_2.append((q0*self.subpolicy_nets[0](s0,temp,log=False)).sum(1).mean().item())
            self.debug['ideal_q'].append(q0.max(1)[0].mean(0).item())
            self.debug['q_diff_1'].append(self.debug['ideal_q'][-1]-self.state_values_1[-1])
            self.debug['q_diff_2'].append(self.debug['ideal_q'][-1]-self.state_values_2[-1])
            self.debug['q_diff_diff'].append(self.debug['q_diff_1'][-1]-self.debug['q_diff_2'][-1])

            action_probs = self.subpolicy_nets[0](s0,temp,log=False) # log_prob.shape = batch size * # actions
            self.debug['policy_diff_2'].append(((action_probs-ideal_policy)**2).sum(1).mean(0).item())

            # Update target weights
            params = [zip(self.q_net_target.parameters(), self.q_net.parameters())]
            for p1,p2 in itertools.chain.from_iterable(params):
                p1.data = (1-tau)*p1+tau*p2
    def train_actor_critic_v8(self,batch_size=2,iterations=1):
        # Actor critic, with Q function learned off-policy
        # Optimize for expected return using advantage AC
        # The advantage is computed using the sample instead of using the Q function, and is on-policy
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = self.get_dataloader(batch_size)
        gamma = self.discount_factor
        tau = self.polyak_rate
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        for i,((s0a,s0,m0),a0,r1,(s1a,s1,m1),t) in zip(range(iterations),dataloader):
            # Fix data types
            s0a = s0a.float().to(self.device)   # State at which the subpolicy was chosen (augmented)
            s0 = s0.float().to(self.device)     # State at which the subpolicy was chosen (not augmented)
            m0 = m0.float().to(self.device)     # 1 if s0a exists, and 0 otherwise
            a0 = a0.to(self.device)             # Action drawn from chosen subpolicy
            r1 = r1.float().to(self.device)     # Reward obtained for taking action a1 at state s1
            s1a = s1a.float().to(self.device)   # State on which the subpolicy was applied (augmented)
            s1 = s1.float().to(self.device)     # State on which the subpolicy was applied (not augmented)
            m1 = m1.float().to(self.device)     # 1 if s1a exists, and 0 otherwise
            t = t.float().to(self.device)       # 1 if s1 is a terminal state, 0 otherwise
            temp = self.target_temp             # Temperature
            batch_size = s0a.shape[0]
            
            # Update Q function
            qs0a0_pred = self.q_net(s0)[range(batch_size),a0.squeeze()]
            qs0a0_target = (r1+gamma*self.q_net_target(s1).max(1)[0]*(1-t)).detach()
            critic_optimizer.zero_grad()
            critic_loss = ((qs0a0_target-qs0a0_pred)**2).mean()
            critic_loss.backward()
            critic_optimizer.step()

            # Update policy
            actor_optimizer.zero_grad()

            q0 = self.q_net(s0).detach() # batch size * # actions
            ideal_policy = torch.zeros_like(q0)
            ideal_policy[range(batch_size),q0.max(1)[1]]=1
            self.debug['ideal_policy_entropy'].append(
                -(torch.log(ideal_policy.sum(0)/ideal_policy.sum())*ideal_policy.sum(0)/ideal_policy.sum()).sum().item()
            )

            self.state_values_1.append((q0*self.subpolicy_nets[0](s0,temp,log=False)).sum(1).mean().item())
            log_probs = self.subpolicy_nets[0](s0,temp,log=True) # log_prob.shape = batch size * # actions
            action_probs = torch.exp(log_probs) # batch size * # actions
            action_probs1 = self.subpolicy_nets[0](s1,temp,log=False)
            qs0a0 = (r1+gamma*(action_probs1*self.q_net_target(s1)).sum(1)*(1-t)).detach()
            expected_q0 = (action_probs*q0).sum(1).view(-1,1).detach()

            self.debug['policy_diff_1'].append(((action_probs-ideal_policy)**2).sum(1).mean(0).item())
            self.debug['policy_entropy'].append(
                -(torch.log(action_probs.sum(0)/action_probs.sum())*action_probs.sum(0)/action_probs.sum()).nansum().item()
            )
            self.debug['policy_is_constant'].append((action_probs.sum(0)==0).sum().item()==3)

            actor_loss = -log_probs[range(batch_size),a0.squeeze()]*(qs0a0-expected_q0.squeeze())
            actor_loss = actor_loss.mean(0)

            l2_loss = 0
            l2_count = 0
            l2_weight = self.l2_weight
            for p in self.subpolicy_nets[0].parameters():
                l2_loss += (p**2).sum()
                l2_count += p.flatten().shape[0]
            l2_loss /= l2_count
            actor_loss += l2_weight*l2_loss
            self.debug['l2_loss'].append(l2_loss.item())

            actor_loss.backward()
            actor_optimizer.step()

            g = 0
            for p in self.subpolicy_nets[0].parameters():
                g += p.grad.abs().sum().item()
                #with torch.no_grad():
                #    p -= self.subpolicy_learning_rate*p.grad
            self.debug['grad_snet0'].append(g)

            #if g > 10000:
            #    breakpoint()

            self.state_values_2.append((q0*self.subpolicy_nets[0](s0,temp,log=False)).sum(1).mean().item())
            self.debug['ideal_q'].append(q0.max(1)[0].mean(0).item())
            self.debug['q_diff_1'].append(self.debug['ideal_q'][-1]-self.state_values_1[-1])
            self.debug['q_diff_2'].append(self.debug['ideal_q'][-1]-self.state_values_2[-1])
            self.debug['q_diff_diff'].append(self.debug['q_diff_1'][-1]-self.debug['q_diff_2'][-1])

            action_probs = self.subpolicy_nets[0](s0,temp,log=False) # log_prob.shape = batch size * # actions
            self.debug['policy_diff_2'].append(((action_probs-ideal_policy)**2).sum(1).mean(0).item())

            # Update target weights
            params = [zip(self.q_net_target.parameters(), self.q_net.parameters())]
            for p1,p2 in itertools.chain.from_iterable(params):
                p1.data = (1-tau)*p1+tau*p2
    def train_actor_critic_v9(self,batch_size=2,iterations=1):
        # Actor critic, with Q function learned off-policy
        # Optimize for expected return using advantage AC
        # The advantage is computed using the sample instead of using the Q function, and is on-policy
        # Includes entropy loss
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = self.get_dataloader(batch_size, replacement=True)
        gamma = self.discount_factor
        tau = self.polyak_rate
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        for i,((s0a,s0,m0),a0,r1,(s1a,s1,m1),t) in zip(range(iterations),dataloader):
            # Fix data types
            s0a = s0a.float().to(self.device)   # State at which the subpolicy was chosen (augmented)
            s0 = s0.float().to(self.device)     # State at which the subpolicy was chosen (not augmented)
            m0 = m0.float().to(self.device)     # 1 if s0a exists, and 0 otherwise
            a0 = a0.to(self.device)             # Action drawn from chosen subpolicy
            r1 = r1.float().to(self.device)     # Reward obtained for taking action a1 at state s1
            s1a = s1a.float().to(self.device)   # State on which the subpolicy was applied (augmented)
            s1 = s1.float().to(self.device)     # State on which the subpolicy was applied (not augmented)
            m1 = m1.float().to(self.device)     # 1 if s1a exists, and 0 otherwise
            t = t.float().to(self.device)       # 1 if s1 is a terminal state, 0 otherwise
            temp = self.target_temp             # Temperature
            batch_size = s0a.shape[0]
            
            # Update Q function
            qs0a0_pred = self.q_net(s0)[range(batch_size),a0.squeeze()]
            qs0a0_target = (r1+gamma*self.q_net_target(s1).max(1)[0]*(1-t)).detach()
            critic_optimizer.zero_grad()
            critic_loss = ((qs0a0_target-qs0a0_pred)**2).mean()
            critic_loss.backward()
            critic_optimizer.step()

            # Update policy
            actor_optimizer.zero_grad()

            q0 = self.q_net(s0).detach() # batch size * # actions
            ideal_policy = torch.zeros_like(q0)
            ideal_policy[range(batch_size),q0.max(1)[1]]=1
            self.debug['ideal_policy_entropy'].append(
                -(torch.log(ideal_policy.sum(0)/ideal_policy.sum())*ideal_policy.sum(0)/ideal_policy.sum()).sum().item()
            )

            self.state_values_1.append((q0*self.subpolicy_nets[0](s0,temp,log=False)).sum(1).mean().item())
            log_probs = self.subpolicy_nets[0](s0,temp,log=True) # log_prob.shape = batch size * # actions
            action_probs = torch.exp(log_probs) # batch size * # actions
            action_probs1 = self.subpolicy_nets[0](s1,temp,log=False)
            qs0a0 = (r1+gamma*(action_probs1*self.q_net_target(s1)).sum(1)*(1-t)).detach()
            expected_q0 = (action_probs*q0).sum(1)#.detach()
            entropy = -(log_probs*action_probs).sum(1).mean(0)

            self.debug['policy_diff_1'].append(((action_probs-ideal_policy)**2).sum(1).mean(0).item())
            self.debug['policy_entropy'].append(
                -(torch.log(action_probs.sum(0)/action_probs.sum())*action_probs.sum(0)/action_probs.sum()).nansum().item()
            )
            self.debug['policy_is_constant'].append((action_probs.sum(0)==0).sum().item()==3)

            ## v1
            #advantage = qs0a0-expected_q0
            ## v2
            #advantage = ((q0.max(1)[1] == a0.squeeze())*1).detach()
            # v3
            advantage = q0[range(batch_size),a0.squeeze()]-expected_q0
            #breakpoint()
            #advantage = advantage.clip(min=-0.05)+0.05
            advantage = advantage - advantage.min()
            advantage = advantage/advantage.max()
            actor_loss = -log_probs[range(batch_size),a0.squeeze()]*advantage
            actor_loss = actor_loss.mean(0)
            #actor_loss -= self.entropy_weight*entropy

            l2_loss = 0
            l2_count = 0
            l2_weight = self.l2_weight
            for p in self.subpolicy_nets[0].parameters():
                l2_loss += (p**2).sum()
                l2_count += p.flatten().shape[0]
            l2_loss /= l2_count
            #actor_loss += l2_weight*l2_loss
            self.debug['l2_loss'].append(l2_loss.item())

            actor_loss.backward()
            actor_optimizer.step()

            g = 0
            for p in self.subpolicy_nets[0].parameters():
                g += p.grad.abs().sum().item()
                #with torch.no_grad():
                #    p -= self.subpolicy_learning_rate*p.grad
            self.debug['grad_snet0'].append(g)

            #if g > 10000:
            #    breakpoint()

            self.state_values_2.append((q0*self.subpolicy_nets[0](s0,temp,log=False)).sum(1).mean().item())
            self.debug['ideal_q'].append(q0.max(1)[0].mean(0).item())
            self.debug['q_diff_1'].append(self.debug['ideal_q'][-1]-self.state_values_1[-1])
            self.debug['q_diff_2'].append(self.debug['ideal_q'][-1]-self.state_values_2[-1])
            self.debug['q_diff_diff'].append(self.debug['q_diff_1'][-1]-self.debug['q_diff_2'][-1])

            action_probs = self.subpolicy_nets[0](s0,temp,log=False) # log_prob.shape = batch size * # actions
            self.debug['policy_diff_2'].append(((action_probs-ideal_policy)**2).sum(1).mean(0).item())

            # Update target weights
            params = [zip(self.q_net_target.parameters(), self.q_net.parameters())]
            for p1,p2 in itertools.chain.from_iterable(params):
                p1.data = (1-tau)*p1+tau*p2
    def train_actor_critic_v10(self,batch_size=2,iterations=1):
        # Actor critic, with Q function learned off-policy
        # With hierarchy
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = self.get_dataloader(batch_size, replacement=True)
        gamma = self.discount_factor
        tau = self.polyak_rate
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        for i,((s0a,s0,m0),a0,r1,(s1a,s1,m1),t) in zip(range(iterations),dataloader):
            # Fix data types
            s0a = s0a.float().to(self.device)   # State at which the subpolicy was chosen (augmented)
            s0 = s0.float().to(self.device)     # State at which the subpolicy was chosen (not augmented)
            m0 = m0.float().to(self.device)     # 1 if s0a exists, and 0 otherwise
            a0 = a0.to(self.device)             # Action drawn from chosen subpolicy
            r1 = r1.float().to(self.device)     # Reward obtained for taking action a1 at state s1
            s1a = s1a.float().to(self.device)   # State on which the subpolicy was applied (augmented)
            s1 = s1.float().to(self.device)     # State on which the subpolicy was applied (not augmented)
            m1 = m1.float().to(self.device)     # 1 if s1a exists, and 0 otherwise
            t = t.float().to(self.device)       # 1 if s1 is a terminal state, 0 otherwise
            temp = self.target_temp             # Temperature
            batch_size = s0a.shape[0]
            
            # Update Q function
            qs0a0_pred = self.q_net(s0)[range(batch_size),a0.squeeze()]
            qs0a0_target = (r1+gamma*self.q_net_target(s1).max(1)[0]*(1-t)).detach()
            critic_optimizer.zero_grad()
            critic_loss = ((qs0a0_target-qs0a0_pred)**2).mean()
            critic_loss.backward()
            critic_optimizer.step()

            # Update policy
            actor_optimizer.zero_grad()

            a0_probs, extras = self.policy_net(s0a,s0,m0,temp,temp,extras=True) # Action probs at s0
            q0 = self.q_net(s0) # batch size * # actions
            # Train subpolicies to maximize expected return, weighted by the probability of the controller choosing that subpolicy
            subpolicy_log_probs = [sp(s0,temp,log=True) for sp in self.subpolicy_nets] # Log action prob for each subpolicy
            controller_log_probs = extras['controller_log_probs'] # batch size * # options
            option_probs = torch.exp(controller_log_probs).detach()
            for sp_idx,log_probs in enumerate(subpolicy_log_probs):
                # log_prob.shape = batch size * # actions
                action_probs = torch.exp(log_probs) # batch size * # actions
                delta = None # batch size * # actions
                if self.ac_variant == 'advantage':
                    expected_q0 = (action_probs*q0).sum(1)#.detach()
                    delta = q0[range(batch_size),a0.squeeze()]-expected_q0
                    delta = delta.clip(min=0)
                elif self.ac_variant == 'q':
                    delta = q0[range(batch_size),a0.squeeze()].detach()
                actor_loss = -log_probs[range(batch_size),a0.squeeze()]*delta*option_probs[:,sp_idx]
                actor_loss = actor_loss.mean(0)
                actor_loss.backward(retain_graph=True) # Accumulate gradients
            # Train controller to favour subpolicies with higher expected return at this state
            option_q0 = torch.stack([(torch.exp(log_prob)*q0).sum(1) for log_prob in subpolicy_log_probs],dim=1) # batch size * # options
            delta = None
            if self.ac_variant == 'advantage':
                delta = (option_q0-(option_probs*option_q0).sum(1).view(-1,1)).detach()
                delta = delta.clip(min=0)
            elif self.ac_variant == 'q':
                delta = option_q0.detach()
            actor_loss = -controller_log_probs*delta*(option_probs.detach())
            actor_loss = actor_loss.sum(1)
            actor_loss = actor_loss.mean(0)
            actor_loss.backward() # Accumulate gradients

            actor_optimizer.step()

            # Update target weights
            params = [zip(self.q_net_target.parameters(), self.q_net.parameters())]
            for p1,p2 in itertools.chain.from_iterable(params):
                p1.data = (1-tau)*p1+tau*p2
    def train_actor_critic_v11(self,batch_size=2,iterations=1):
        # Actor critic, with Q function learned off-policy
        # With hierarchy
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = self.get_dataloader(batch_size, replacement=True)
        gamma = self.discount_factor
        tau = self.polyak_rate
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        for i,((s0a,s0,m0),a0,r1,(s1a,s1,m1),t) in zip(range(iterations),dataloader):
            # Fix data types
            s0a = s0a.float().to(self.device)   # State at which the subpolicy was chosen (augmented)
            s0 = s0.float().to(self.device)     # State at which the subpolicy was chosen (not augmented)
            m0 = m0.float().to(self.device)     # 1 if s0a exists, and 0 otherwise
            a0 = a0.to(self.device)             # Action drawn from chosen subpolicy
            r1 = r1.float().to(self.device)     # Reward obtained for taking action a1 at state s1
            s1a = s1a.float().to(self.device)   # State on which the subpolicy was applied (augmented)
            s1 = s1.float().to(self.device)     # State on which the subpolicy was applied (not augmented)
            m1 = m1.float().to(self.device)     # 1 if s1a exists, and 0 otherwise
            t = t.float().to(self.device)       # 1 if s1 is a terminal state, 0 otherwise
            temp = self.target_temp             # Temperature
            batch_size = s0a.shape[0]
            
            # Update Q function
            qs0a0_pred = self.q_net(s0)[range(batch_size),a0.squeeze()]
            qs0a0_target = (r1+gamma*self.q_net_target(s1).max(1)[0]*(1-t)).detach()
            critic_optimizer.zero_grad()
            critic_loss = ((qs0a0_target-qs0a0_pred)**2).mean()
            critic_loss.backward()
            critic_optimizer.step()

            # Update policy
            actor_optimizer.zero_grad()
            a0_probs, extras = self.policy_net(s0a,s0,m0,temp,temp,extras=True) # Action probs at s0
            q0 = self.q_net(s0) # batch size * # actions
            # Train subpolicies to maximize expected return, weighted by the probability of the controller choosing that subpolicy
            subpolicy_log_probs = [sp(s0,temp,log=True) for sp in self.subpolicy_nets] # Log action prob for each subpolicy
            controller_log_probs = extras['controller_log_probs'] # batch size * # options
            option_probs = torch.exp(controller_log_probs).detach()
            for sp_idx,log_probs in enumerate(subpolicy_log_probs):
                # log_prob.shape = batch size * # actions
                action_probs = torch.exp(log_probs) # batch size * # actions
                delta = None # batch size * # actions
                if self.ac_variant == 'advantage':
                    expected_q0 = (action_probs*q0).sum(1)#.detach()
                    delta = q0[range(batch_size),a0.squeeze()]-expected_q0
                    delta = delta.clip(min=0)
                elif self.ac_variant == 'q':
                    delta = q0[range(batch_size),a0.squeeze()].detach()
                actor_loss = -log_probs[range(batch_size),a0.squeeze()]*delta*option_probs[:,sp_idx]
                actor_loss = actor_loss.mean(0)
                actor_loss.backward(retain_graph=True) # Accumulate gradients
            # Train controller to favour subpolicies with higher expected return at this state
            option_q0 = torch.stack([(torch.exp(log_prob)*q0).sum(1) for log_prob in subpolicy_log_probs],dim=1) # batch size * # options
            o0 = torch.stack(subpolicy_log_probs)[:,range(batch_size),a0].max(0)[1] # Most probable options given the primitive action choice
            delta = None
            if self.ac_variant == 'advantage':
                expected_q0 = (option_probs*option_q0).sum(1)#.detach()
                delta = option_q0[range(batch_size),o0]-expected_q0
                delta = delta.clip(min=0)
            elif self.ac_variant == 'q':
                delta = option_q0.detach()
            actor_loss = -controller_log_probs[range(batch_size),o0]*delta
            actor_loss = actor_loss.mean(0)
            actor_loss.backward() # Accumulate gradients

            actor_optimizer.step()

            # Update target weights
            params = [zip(self.q_net_target.parameters(), self.q_net.parameters())]
            for p1,p2 in itertools.chain.from_iterable(params):
                p1.data = (1-tau)*p1+tau*p2

    def state_dict(self):
        d = super().state_dict()
        d['action_mem'] = self.action_mem
        d['ac_variant'] = self.ac_variant
        d['delay'] = self.delay
        return d

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.action_mem = state['action_mem']
        self.ac_variant = state['ac_variant']
        self.delay = state['delay']

    def get_obs_sizes(self):
        """ Return an array with the size of the inputs for policies of each level of the hierarchy (highest to lowest). """
        if len(self.observation_space.high.shape) > 1:
            raise NotImplementedError('Cannot handle multidimensional observations')
        return [
            self.observation_space.high.shape[0]+self.action_space.n*self.action_mem,
            self.observation_space.high.shape[0]
        ]

    def observe_change(self, obs, reward=None, terminal=False, testing=False):
        if testing:
            obs_stack = self.obs_stack_testing
        else:
            obs_stack = self.obs_stack
            
        if reward is None: # reward is None if it is the first observation of the episode
            obs_stack.clear()
        obs_stack.append_obs(obs)

        if not testing and reward is not None:
            # Add to the replay buffer if we have a long enough trajectory to add
            s0_aug = obs_stack.get(1+self.delay-self.action_mem,self.action_mem)
            s0 = obs_stack.get(1,0)
            a0 = obs_stack.get_action(1)
            s1_aug = obs_stack.get(0+self.delay-self.action_mem,self.action_mem)
            s1 = obs_stack.get(0,0)
            s0_mask = torch.tensor([s0_aug is not None, s0 is not None]).float().to(self.device)
            s1_mask = torch.tensor([s1_aug is not None, s1 is not None]).float().to(self.device)

            obs_sizes = self.get_obs_sizes()
            if s0 is None:
                s0 = np.zeros([obs_sizes[1]])
            if s1 is None:
                s1 = np.zeros([obs_sizes[1]])
            if s0_aug is None:
                s0_aug = np.zeros([obs_sizes[0]])
            if s1_aug is None:
                s1_aug = np.zeros([obs_sizes[0]])

            s0 = torch.tensor(s0).squeeze().float()
            s0_aug = torch.tensor(s0_aug).squeeze().float()
            s1 = torch.tensor(s1).squeeze().float()
            s1_aug = torch.tensor(s1_aug).squeeze().float()

            self.replay_buffer.add_transition((s0_aug,s0,s0_mask),a0,reward,(s1_aug,s1,s1_mask),terminal)

        # Dropout
        if self.controller_dropout is not None and \
                self.rand.random() < self.controller_dropout and \
                self.controller_obs[testing] is None:
            s0_aug = obs_stack.get(1+self.delay-self.action_mem,self.action_mem)
            if s0_aug is not None:
                s0_aug = torch.tensor(s0_aug).squeeze()
            self.controller_obs[testing] = s0_aug # If s0_aug is None, it'll just be ignored.
        else:
            self.controller_obs[testing] = None

    def act(self, testing=False):
        if self.algorithm == 'q-learning':
            return self.act_q_learning(testing)
        elif self.algorithm == 'actor-critic':
            return self.act_actor_critic(testing)
        elif self.algorithm == 'actor-critic-v2':
            return self.act_actor_critic_v2(testing)
        elif self.algorithm == 'actor-critic-v3':
            return self.act_actor_critic_v2(testing) # Save as v2
        #elif self.algorithm == 'actor-critic-v4':
        else:
            return self.act_actor_critic_v2(testing) # Save as v2
    def act_actor_critic(self, testing=False):
        """
        Return a random action according to the current behaviour policy

        Args:
            testing: True if this action was taken during testing. False otherwise.
        """

        obs0,obs1,mask = self.get_current_obs(testing)

        action = self.q_net(obs1).squeeze().argmax().item()
        if testing:
            self.obs_stack_testing.append_action(action)
        else:
            if self.rand.random() < 0.1:
                action = self.action_space.sample()
            self.obs_stack.append_action(action)
        self.current_action = action
        return self.current_action

        # Dropout
        if self.controller_obs[testing] is not None:
            #print('drop',obs0-self.controller_obs[testing].float().squeeze().unsqueeze(0))
            obs0 = self.controller_obs[testing]
            obs0 = obs0.float().squeeze().unsqueeze(0).to(self.device)

        # Sample an action
        temp = self.target_temp if testing else self.behaviour_temp
        action_probs,policy_net_output = self.policy_net(obs0,obs1,mask,temp,temp,extras=True)
        #action_probs = action_probs.squeeze().detach().numpy()
        subpolicy_probs = policy_net_output['subpolicy_probabilities'].squeeze().detach().numpy()
        subpolicy_probs = subpolicy_probs.reshape(-1) # In case there's only one subpolicy
        subpolicy_choice = self.rand.choice(len(subpolicy_probs),p=subpolicy_probs)
        action_probs = policy_net_output['primitive_action_probabilities'][0,subpolicy_choice,:].detach().numpy() # policy_net_output['primitive_action_probabilities'] has shape [1 * # options * # primitive actions]
        #if self.controller_obs[testing] is not None:
        #    o0 = self.controller_obs[testing].float().squeeze().unsqueeze(0)
        #    action_probs2 = self.policy_net(o0,obs1,mask,temp,temp).squeeze().detach().numpy()
        #    print('drop',action_probs-action_probs2)
        #    #print('drop',action_probs,action_probs2)
        action = self.rand.choice(len(action_probs),p=action_probs)
        self.current_action = action

        self.action_counts[testing][action] += 1
        self.option_counts[testing][subpolicy_choice] += 1

        # Save action
        if testing:
            self.obs_stack_testing.append_action(action)
        else:
            self.obs_stack.append_action(action)

        return action
    def act_q_learning(self, testing=False):
        """
        Return a random action according to the current behaviour policy

        Args:
            testing: True if this action was taken during testing. False otherwise.
        """

        obs0,obs1,mask = self.get_current_obs(testing)

        action = self.q_net(obs1).squeeze().argmax().item()
        if testing:
            self.obs_stack_testing.append_action(action)
        else:
            if self.rand.random() < 0.1:
                action = self.action_space.sample()
            self.obs_stack.append_action(action)
        self.current_action = action
        return self.current_action
    def act_actor_critic_v2(self, testing=False):
        obs_aug,obs,mask = self.get_current_obs(testing)

        # Dropout
        if self.controller_obs[testing] is not None:
            #print('drop',obs0-self.controller_obs[testing].float().squeeze().unsqueeze(0))
            obs_aug = self.controller_obs[testing]
            obs_aug = obs_aug.float().squeeze().unsqueeze(0).to(self.device)

        # Sample an action
        temp = self.target_temp if testing else self.behaviour_temp
        _,policy_net_output = self.policy_net(obs_aug,obs,mask,temp,temp,extras=True)

        # Sample a subpolicy
        subpolicy_probs = policy_net_output['subpolicy_probabilities'].squeeze().detach().numpy()
        subpolicy_probs = subpolicy_probs.reshape(-1) # In case there's only one subpolicy
        subpolicy_choice = self.rand.choice(len(subpolicy_probs),p=subpolicy_probs)

        #self.current_action = subpolicy_choice # DEBUG: One action per subpolicy
        #action = subpolicy_choice

        # Sample action from chosen subpolicy
        action_probs = policy_net_output['primitive_action_probabilities'][0,subpolicy_choice,:].detach().numpy() # policy_net_output['primitive_action_probabilities'] has shape [1 * # options * # primitive actions]
        action = self.rand.choice(len(action_probs),p=action_probs)
        self.current_action = action

        self.action_counts[testing][action] += 1
        self.option_counts[testing][subpolicy_choice] += 1

        # Save action
        if testing:
            self.obs_stack_testing.append_action(action)
        else:
            self.obs_stack.append_action(action)

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

        obs_aug = obs_stack.get(self.delay-self.action_mem,self.action_mem)
        obs = obs_stack.get(0,0)
        mask = torch.tensor([[obs_aug is not None, obs is not None]]).float().to(self.device)

        def to_tensor(x):
            if x is None:
                size = self.get_obs_sizes()[0] # Only obs0 can be None
                return torch.zeros([1,size]).float().to(self.device)
            return torch.tensor(x).float().squeeze().unsqueeze(0).to(self.device)

        return to_tensor(obs_aug),to_tensor(obs),mask
