import numpy as np
from tqdm import tqdm
import torch
import gym
import copy
import itertools
from sklearn.gaussian_process import GaussianProcessRegressor

import numpy as np

from .agent import Agent
from agent.dqn_agent import DQNAgent
from . import ReplayBuffer
from .policy import get_greedy_epsilon_policy, greedy_action

"""
TODO:
- Don't call the oracle on a state if it's already been called on that state
- Set the beta/lambda/zeta parameters properly (As described by the paper)
- Store values exactly instead of with a functoin approximator to make sure the algorithm works first
"""

class MultiFidelityDQNAgent(DQNAgent):
    def __init__(self, action_space, observation_space, discount_factor, learning_rate=1e-3, polyak_rate=0.001, device=torch.device('cpu'), behaviour_policy=get_greedy_epsilon_policy(0.1), target_policy=get_greedy_epsilon_policy(0), q_net=None, replay_buffer_size=50000, transition_function=None, oracles=[], oracle_costs=[]):
        super().__init__(action_space=action_space, observation_space=observation_space, discount_factor=discount_factor, learning_rate=learning_rate, polyak_rate=1, device=device, behaviour_policy=behaviour_policy, target_policy=target_policy, q_net=q_net, replay_buffer_size=replay_buffer_size)

        self.oracles = oracles
        self.transition_function = transition_function

        self.zeta = 0.1 # Max difference between levels of fidelity
        self.gamma = [0.1]*len(oracles) # Threshold for deciding between fidelity levels
        self.beta = 1 # Threshold for standard deviation
        self.estimates = [GaussianProcessRegressor() for _ in oracles]
        self.oracle_data = None # Values from oracle calls
        self.oracle_costs = oracle_costs # Runtime costs for each oracle

    def observe_change(self, obs, testing=False):
        if testing:
            self.current_obs_testing = obs
        else:
            self.replay_buffer._add_to_buffer(obs)
            self.current_obs = obs
            self.current_action = None

    def neighbours(self, state):
        """
        state - a single state (np.array)
        """
        output = [state]
        if state.sum() >= 25: # TODO: Not sure if this is right. Check for off-by-one error.
            return output
        for action in range(self.action_space.n):
            s = self.transition_function(state,action)
            if s is not None:
                output.append(s)
        return output

    def compute_state_value_ucb(self,state,fidelity):
        gp = self.estimates[fidelity]
        mu,std = gp.predict(state.reshape(1,-1),return_std=True)
        beta = self.beta
        zeta = self.zeta
        phi = mu + (beta**(1/2))*std+zeta
        return phi

    def compute_state_value(self, state):
        """
        state - a single state (np.array)
        """
        max_val = -float('inf')
        for s in self.neighbours(state):
            min_phi = float('inf')
            for fidelity in range(len(self.oracles)):
                phi = self.compute_state_value_ucb(s,fidelity)
                if phi < min_phi:
                    min_phi = phi
            if min_phi > max_val:
                max_val = max_val
        return max_val

    def train(self,batch_size=2,iterations=1):
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = torch.utils.data.DataLoader(
                self.replay_buffer, batch_size=batch_size, shuffle=True)
        gamma = self.discount_factor
        tau = self.polyak_rate
        optimizer = self.optimizer
        for i,s in zip(range(iterations),dataloader):
            # Fix data types
            s = s.to(self.device)
            # Value estimate
            y = torch.tensor([self.compute_state_value(x.numpy()) for x in s])
            # Update Q network
            optimizer.zero_grad()
            loss = (y-self.q_net(s.float())).mean()
            loss.backward()
            optimizer.step()

            # Update target weights
            for p1,p2 in zip(self.q_net_target.parameters(), self.q_net.parameters()):
                p1.data = (1-tau)*p1+tau*p2

    def act(self, testing=False):
        """ Check if we need more information on current state. """
        #observation = torch.tensor(observation, dtype=torch.float).view(-1,4,84,84).to(self.device)
        if testing:
            obs = self.current_obs_testing
            policy = self.target_policy
        else:
            # TODO
            # Check if we need more info on current state
            # If so, check which fidelity we need to evaluate at
            # If not, transition to next best state
            obs = self.current_obs
            policy = self.behaviour_policy
        next_obs = [self.transition_function(obs,a) for a in range(self.action_space.n)]
        vals = torch.tensor([self.q_net(torch.tensor(o).unsqueeze(0).float()) for o in next_obs])
        dist = policy(vals)
        action = dist.sample().item()
        self.current_action = action
        self.last_vals = vals
        return action

    def init_oracle_data(self):
        if self.oracle_data is not None:
            return # Don't initialize again
        if len(self.replay_buffer) < 100:
            #raise Exception('Not enough data in replay buffer to initialize Gaussian processes.')
            return
        self.oracle_data = [{} for _ in self.oracles] # Values from oracle calls
        for i in range(len(self.oracles)):
            for x in np.random.choice(list(range(len(self.replay_buffer))),size=5,replace=False):
                obs = self.replay_buffer[x]
                if obs is None: # We'll just start with one less. nbd.
                    continue
                val = self.oracles[i](torch.tensor(obs).float())
                self.oracle_data[i][tuple(obs.tolist())] = val
            x = list(self.oracle_data[i].keys())
            y = [self.oracle_data[i][k] for k in x]
            self.estimates[i].fit(x,y)

    def evaluate_obs(self):
        # Check if obs needs evaluating
        obs = self.current_obs
        #if self.compute_state_value(obs) >= self.q_net(torch.tensor(obs).float()):
        if self.oracle_data is not None:
            # evaluate and return runtime
            for i in range(len(self.oracles)):
                _,std = self.estimates[i].predict(obs.reshape(1,-1),return_std=True)
                if np.sqrt(self.beta)*std < self.gamma[i] and i < len(self.oracles)-1:
                    # If we're reasonably certain of this estimate, move on to a higher fidelity
                    continue
                # Evaluate at fidelity i
                self.oracle_data[i][tuple(obs.tolist())] = self.oracles[i](obs)
                # Update Gaussian processes
                x = list(self.oracle_data[i].keys())
                y = [self.oracle_data[i][k] for k in x]
                self.estimates[i].fit(x,y)
                # Return runtime
                return self.oracle_costs[i]
            raise Exception('Something went wrong. No oracles could be called.')
        else:
            # Nothing to do, so runtime is 0
            return 0

    def test_once(self, env, max_steps=np.inf, render=False):
        reward_sum = 0
        sa_vals = []
        # Sample highest-scoring path through the graph
        obs = env.reset()
        all_obs = [obs]
        self.observe_change(obs, testing=True)
        for steps in itertools.count():
            if steps > max_steps:
                break
            action = self.act(testing=True)
            sa_vals.append(self.get_state_action_value(obs,action))
            obs, _, done, _ = env.step(action)
            all_obs.append(obs)
            self.observe_change(obs, testing=True)
            if render:
                env.render()
            if done:
                break
        # Choose highest-scoring entity in the graph according to estimates
        fidelity = len(self.oracles)-1
        scores = [self.compute_state_value_ucb(obs,fidelity) for obs in all_obs]
        i = np.argmax(scores)
        reward = self.oracles[-1](all_obs[i])
        return reward, np.mean(sa_vals)

    def test(self, env, iterations, max_steps=np.inf, render=False, record=True, processors=1):
        rewards = []
        sa_vals = []
        for i in range(iterations):
            r,sav = self.test_once(env, render=render, max_steps=max_steps)
            rewards.append(r)
            sa_vals.append(sav)
        return rewards, sa_vals

class MultiFidelityDiscreteAgent(Agent):
    def __init__(self, action_space, observation_space, behaviour_policy=get_greedy_epsilon_policy(0.1), target_policy=get_greedy_epsilon_policy(0), transition_function=None, oracles=[], oracle_costs=[], warmup_steps=100, max_depth=5):
        self.action_space = action_space
        self.observation_space = observation_space
        self.behaviour_policy = behaviour_policy
        self.target_policy = target_policy
        self.max_depth = max_depth

        self.state_values_explore = {}
        self.state_values_exploit = {}

        self.oracles = oracles
        self.transition_function = transition_function

        self.zeta = 0.1 # Max difference between levels of fidelity
        self.gamma = [0.1]*len(oracles) # Threshold for deciding between fidelity levels
        self.beta = 1 # Threshold for standard deviation
        self.estimates = [GaussianProcessRegressor() for _ in oracles]
        self.oracle_data = None # Values from oracle calls
        self.oracle_costs = oracle_costs # Runtime costs for each oracle

        self.warmup_steps = warmup_steps
        self.warmup_states = []

    def observe_change(self, obs, testing=False):
        if testing:
            self.current_obs_testing = obs
        else:
            self.current_obs = obs
            self.current_action = None

    def neighbours(self, state):
        """
        state - a single state (np.array)
        """
        output = []
        if state.sum() >= self.max_depth: # TODO: Not sure if this is right. Check for off-by-one error.
            return output
        for action in range(self.action_space.n):
            s = self.transition_function(state,action)
            if s is not None:
                output.append(s)
        return output

    def compute_state_value_ucb(self,state,fidelity):
        gp = self.estimates[fidelity]
        mu,std = gp.predict(state.reshape(1,-1),return_std=True)
        beta = self.beta
        zeta = self.zeta
        phi = mu + (beta**(1/2))*std+zeta
        return phi.item()

    def compute_state_value(self, state):
        """
        state - a single state (np.array)
        """
        n = list(self.neighbours(state))
        if len(n) > 0:
            max_val = max(*[self.state_values_explore[tuple(s.astype(np.int).tolist())] for s in n])
        else:
            max_val = -float('inf')
        max_phi = max(*[self.compute_state_value_ucb(state,fidelity) for fidelity in range(len(self.oracles))])
        return max(max_val,max_phi)

    def train(self):
        if self.oracle_data is None:
            return # Don't train until we're done with warmup
        def all_states(starting_state=np.zeros(self.observation_space.low.shape),max_depth=self.max_depth):
            if max_depth > 0:
                for a in range(self.action_space.n):
                    diff = np.zeros(self.observation_space.low.shape)
                    diff[a] = 1
                    yield from all_states(starting_state+diff,max_depth-1)
            yield starting_state

        #for s in tqdm(list(all_states()),desc='training'):
        for s in all_states():
            self.state_values_explore[tuple(s.tolist())] = self.compute_state_value(s)

    def act(self, testing=False):
        """ Check if we need more information on current state. """
        #observation = torch.tensor(observation, dtype=torch.float).view(-1,4,84,84).to(self.device)
        if testing:
            obs = self.current_obs_testing
            policy = self.target_policy
        else:
            # TODO
            # Check if we need more info on current state
            # If so, check which fidelity we need to evaluate at
            # If not, transition to next best state
            obs = self.current_obs
            policy = self.behaviour_policy
        if len(self.state_values_explore) == 0:
            self.warmup_states.append(obs)
            return self.action_space.sample()
        next_obs = [self.transition_function(obs,a) for a in range(self.action_space.n)]
        vals = torch.tensor([self.state_values_explore[tuple(o.astype(np.int).tolist())] for o in next_obs])
        dist = policy(vals)
        action = dist.sample().item()
        self.current_action = action
        self.last_vals = vals
        return action

    def init_oracle_data(self):
        if self.oracle_data is not None:
            return # Don't initialize again
        if len(self.warmup_states) < self.warmup_steps:
            return
        self.oracle_data = [{} for _ in self.oracles] # Values from oracle calls
        for i in range(len(self.oracles)):
            for x in np.random.choice(list(range(len(self.warmup_states))),size=5,replace=False):
                obs = self.warmup_states[x]
                if obs is None: # We'll just start with one less. nbd.
                    continue
                val = self.oracles[i](torch.tensor(obs).float())
                self.oracle_data[i][tuple(obs.tolist())] = val
            x = list(self.oracle_data[i].keys())
            y = [self.oracle_data[i][k] for k in x]
            self.estimates[i].fit(x,y)

    def evaluate_obs(self):
        # Check if obs needs evaluating
        obs = self.current_obs
        #if self.oracle_data is None:
        if len(self.state_values_explore) == 0:
            return 0
        #if self.compute_state_value(obs) >= self.state_values_explore[tuple(obs.tolist())]:
        if True:
            # evaluate and return runtime
            for i in range(len(self.oracles)):
                _,std = self.estimates[i].predict(obs.reshape(1,-1),return_std=True)
                if np.sqrt(self.beta)*std < self.gamma[i] and i < len(self.oracles)-1:
                    # If we're reasonably certain of this estimate, move on to a higher fidelity
                    continue
                # Evaluate at fidelity i
                self.oracle_data[i][tuple(obs.tolist())] = self.oracles[i](obs)
                # Update Gaussian processes
                x = list(self.oracle_data[i].keys())
                y = [self.oracle_data[i][k] for k in x]
                self.estimates[i].fit(x,y)
                # Return runtime
                return self.oracle_costs[i]
            raise Exception('Something went wrong. No oracles could be called.')
        else:
            # Nothing to do, so runtime is 0
            return 0

    def test_once(self, env, max_steps=np.inf, render=False):
        reward_sum = 0
        # Sample highest-scoring path through the graph
        obs = env.reset()
        all_obs = [obs]
        self.observe_change(obs, testing=True)
        for steps in itertools.count():
            if steps > max_steps:
                break
            action = self.act(testing=True)
            obs, _, done, _ = env.step(action)
            all_obs.append(obs)
            self.observe_change(obs, testing=True)
            if render:
                env.render()
            if done:
                break
        # Choose highest-scoring entity in the graph according to estimates
        fidelity = len(self.oracles)-1
        scores = [self.compute_state_value_ucb(obs,fidelity) for obs in all_obs]
        i = np.argmax(scores)
        reward = self.oracles[-1](all_obs[i])
        return reward, None

    def test(self, env, iterations, max_steps=np.inf, render=False, record=True, processors=1):
        rewards = []
        sa_vals = []
        for i in range(iterations):
            r,sav = self.test_once(env, render=render, max_steps=max_steps)
            rewards.append(r)
            sa_vals.append(sav)
        return rewards, sa_vals
