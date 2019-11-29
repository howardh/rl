import pytest
import gym
import numpy as np
import torch

import agent
from experiments.hrl.model import QFunction, PolicyFunction
from agent.hdqn_agent import HDQNAgentWithDelayAC, HDQNAgentWithDelayAC_v3
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
    elif agent_name == 'DQNAgent':
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

def test_HDQNAC_v3():
    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Box(
            high=np.array([1,1,1]),low=np.array([0,0,0]))

    class DummyPolicy(torch.nn.Module):
        def __init__(self,ret_val):
            super().__init__()
            self.fc = torch.nn.Linear(1,1)
            self.call_stack = []
            self.ret_val = ret_val
        def forward(self,*params):
            self.call_stack.append(params)
            return self.ret_val.float()

    agent = HDQNAgentWithDelayAC_v3(
                action_space=action_space,
                observation_space=observation_space,
                discount_factor=1,
                behaviour_epsilon=0,
                replay_buffer_size=10,
                controller_net=DummyPolicy(torch.tensor([[0.5,0.5]])),
                subpolicy_nets=[DummyPolicy(torch.tensor([[0.5,0.5]])),DummyPolicy(torch.tensor([[0.5,0.5]]))],
                q_net=DummyPolicy(torch.tensor([[0,0]]))
            )
    
    for testing in [False,True]:
        agent.observe_change(np.array([0,0,0]),testing=testing)
        obs0,action0,obs1,mask = agent.get_current_obs(testing=testing)
        assert (obs1 == torch.tensor([[0,0,0]]).float()).all()
        assert (mask == torch.tensor([[0,1]]).float()).all()

        a = agent.act(testing=testing)
        agent.observe_change(np.array([0,0,0]),0,False,testing=testing)
        obs0,action0,obs1,mask = agent.get_current_obs(testing=testing)
        assert (obs0 == torch.tensor([[0,0,0]]).float()).all()
        assert action0 == a
        assert (obs1 == torch.tensor([[0,0,0]]).float()).all()
        assert (mask == torch.tensor([[1,1]]).float()).all()

        a = agent.act(testing=testing)
        agent.observe_change(np.array([0,0,1]),0,False,testing=testing)
        obs0,action0,obs1,mask = agent.get_current_obs(testing=testing)
        assert (obs0 == torch.tensor([[0,0,0]]).float()).all()
        assert action0 == a
        assert (obs1 == torch.tensor([[0,0,1]]).float()).all()
        assert (mask == torch.tensor([[1,1]]).float()).all()

        a = agent.act(testing=testing)
        agent.observe_change(np.array([0,1,0]),0,False,testing=testing)
        obs0,action0,obs1,mask = agent.get_current_obs(testing=testing)
        assert (obs0 == torch.tensor([[0,0,1]]).float()).all()
        assert action0 == a
        assert (obs1 == torch.tensor([[0,1,0]]).float()).all()
        assert (mask == torch.tensor([[1,1]]).float()).all()

    s0,a0,s1,a1,r2,s2,t,m1 = agent.replay_buffer.buffer[0]
    assert (s0 == torch.tensor([[0,0,0]]).float()).all()
    assert a0 is not None
    assert (s1 == torch.tensor([[0,0,0]]).float()).all()
    assert (s2 == torch.tensor([[0,0,1]]).float()).all()

    s0,a0,s1,a1,r2,s2,t,m1 = agent.replay_buffer.buffer[1]
    assert (s0 == torch.tensor([[0,0,0]]).float()).all()
    assert a0 is not None
    assert (s1 == torch.tensor([[0,0,1]]).float()).all()
    assert (s2 == torch.tensor([[0,1,0]]).float()).all()
