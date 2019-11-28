import pytest
import gym
import numpy as np

import agent
from experiments.hrl.model import QFunction, PolicyFunction
from agent.hdqn_agent import HDQNAgentWithDelayAC
from agent.dqn_agent import DQNAgent

def create_agent(agent_name, env):
    num_actions = env.action_space.n
    state_size = env.observation_space.low.shape[0]
    if agent_name == 'HDQNAgentWithDelayAC':
        num_options = 3
        return HDQNAgentWithDelayAC(
                action_space=env.action_space,
                observation_space=env.observation_space,
                controller_learning_rate=0.01,
                subpolicy_learning_rate=0.01,
                q_net_learning_rate=0.01,
                discount_factor=0.9,
                polyak_rate=0.001,
                behaviour_epsilon=0.01,
                controller_net=PolicyFunction(
                    layer_sizes=[],input_size=state_size,output_size=num_options),
                subpolicy_nets=[PolicyFunction(layer_sizes=[],input_size=state_size,output_size=num_actions) 
                    for _ in range(num_options)],
                q_net=QFunction(layer_sizes=[],input_size=state_size,output_size=num_actions),
        )
    if agent_name == 'DQNAgent':
        return DQNAgent(
                action_space=env.action_space,
                observation_space=env.observation_space,
                learning_rate=0.01,
                discount_factor=0.9,
                polyak_rate=0.001,
                q_net=QFunction(layer_sizes=[],input_size=state_size,output_size=num_actions),
        )

class DummyEnv(gym.Env):
    """ Create a simple Markov chain environment"""
    def __init__(self, actions, observations):
        action_spaces = {
            'discrete': gym.spaces.Discrete(2),
            'box': gym.spaces.Box(high=np.array([1]),low=np.array([0]))
        }
        observation_spaces = {
            'discrete': gym.spaces.Discrete(3),
            'box': gym.spaces.Box(high=np.array([1,1,1]),low=np.array([0,0,0]))
        }
        self.action_space = action_spaces[actions]
        self.observation_space = observation_spaces[observations]

        self.state = 0

    def step(self, action):
        done = False
        reward = 0
        if type(self.action_space) is gym.spaces.Box:
            action = 0 if np.random.rand() < action else 1
        if action == 0:
            self.state -= 1
            if self.state < 0:
                self.state = 0
        else:
            self.state += 1
            if self.state >= 2:
                done = True
                reward = 1

        obs = self.state
        if type(self.observation_space) is gym.spaces.Box:
            obs = np.zeros([3])
            obs[self.state] = 1

        info = {}
        return obs, reward, done, info

    def reset(self):
        self.state = 0
        if type(self.observation_space) is gym.spaces.Box:
            obs = np.zeros([3])
            obs[self.state] = 1
            return obs
        else:
            return self.state

    def render(self):
        pass

    def close(self):
        self.state = None

@pytest.mark.parametrize('agent_name', ['DQNAgent','HDQNAgentWithDelayAC'])
def test_agents_train_without_error(agent_name):
    env = DummyEnv(actions='discrete',observations='box')
    agent = create_agent(agent_name,env)

    obs = env.reset()
    agent.observe_change(obs)

    for _ in range(5):
        obs, reward, done, _ = env.step(agent.act())
        agent.observe_change(obs, reward, terminal=done)
        agent.train()
        if done:
            break
