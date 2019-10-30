import pytest
import gym

import agent
from experiments.hrl.model import QFunction, PolicyFunction
from agent.hdqn_agent import HDQNAgentWithDelayAC
from agent.dqn_agent import DQNAgent

def create_agent(agent_name, env):
    if agent_name == 'HDQNAgentWithDelayAC':
        num_options = 3
        return HDQNAgentWithDelayAC(
                action_space=env.action_space,
                observation_space=env.observation_space,
                learning_rate=0.01,
                discount_factor=0.9,
                polyak_rate=0.001,
                behaviour_epsilon=0.01,
                controller_net=PolicyFunction(
                    layer_sizes=[],input_size=4,output_size=num_options),
                subpolicy_nets=[PolicyFunction(layer_sizes=[],input_size=4) for _ in range(num_options)],
                q_net=QFunction(layer_sizes=[],input_size=4,output_size=4),
        )
    if agent_name == 'DQNAgent':
        return DQNAgent(
                action_space=env.action_space,
                observation_space=env.observation_space,
                learning_rate=0.01,
                discount_factor=0.9,
                polyak_rate=0.001,
                q_net=QFunction(layer_sizes=[],input_size=4,output_size=4),
        )

@pytest.mark.parametrize('agent_name', ['DQNAgent','HDQNAgentWithDelayAC'])
def test_agents_train_without_error(agent_name):
    #env = DummyEnv(actions='discrete',observations='box')
    env = gym.make('gym_fourrooms:fourrooms-v0')
    agent = create_agent(agent_name,env)

    obs = env.reset()
    agent.observe_change(obs)

    for _ in range(10):
        obs, reward, done, _ = env.step(agent.act())
        agent.observe_change(obs, reward, terminal=done)
        agent.train()
        if done:
            break
