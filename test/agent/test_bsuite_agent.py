import pytest

import numpy as np
import haiku as hk
import gym.spaces
from rl.agent.bsuite_agent import A2CAgent, DQNAgent

def test_A2C_load_state_dict():
    action_space = gym.spaces.Discrete(10)
    obs_space = gym.spaces.Box(low=np.array([0,0]),high=np.array([1,1]))
    agent1 = A2CAgent(action_space, obs_space, 0.9, 1e-4, hk.PRNGSequence(0))
    agent2 = A2CAgent(action_space, obs_space, 0.9, 1e-4, hk.PRNGSequence(0))

    # Do stuff with agent1
    obs_sequence = [obs_space.sample() for _ in range(10)]
    agent1.observe_change(obs_sequence[0], None, False, False)
    for obs in obs_sequence:
        agent1.act()
        agent1.observe_change(obs, 1, False, False)

    # Set agent2 to the same state
    agent2.load_state_dict(agent1.state_dict())

    # Run the same code on both agents and check that the outcome is the same
    obs_sequence = [obs_space.sample() for _ in range(10)]
    action_sequence = [[],[]]
    param1 = [[],[]]
    for i,agent in enumerate([agent1,agent2]):
        agent.observe_change(obs_sequence[0], None, False, False)
        for obs in obs_sequence:
            a = agent.act()
            action_sequence[i].append(a)
            param1[i].append(agent.agent._state.params['mlp/~/linear_0']['w'][0,0].item())
            agent.observe_change(obs, 1, False, False)

    assert param1[0] == param1[1]
    assert action_sequence[0] == action_sequence[1]
    # assert agent1.state_dict() == agent2.state_dict()

@pytest.mark.skip(reason='bsuite implementation is not deterministic')
def test_DQN_load_state_dict():
    action_space = gym.spaces.Discrete(10)
    obs_space = gym.spaces.Box(low=np.array([0,0]),high=np.array([1,1]))
    agent1 = DQNAgent(action_space, obs_space, 0.9, warmup_steps=5, batch_size=2, learning_rate=1e-4, rng=hk.PRNGSequence(0))
    agent2 = DQNAgent(action_space, obs_space, 0.9, warmup_steps=5, batch_size=2, learning_rate=1e-4, rng=hk.PRNGSequence(0))
    agent2.load_state_dict(agent1.state_dict())

    obs_sequence = [obs_space.sample() for _ in range(10)]
    action_sequence = [[],[]]
    param1 = [[],[]]
    for i,agent in enumerate([agent1,agent2]):
        agent.observe_change(obs_sequence[0], None, False, False)
        for obs in obs_sequence:
            a = agent.act()
            action_sequence[i].append(a)
            param1[i].append(agent.agent._state.params['mlp/~/linear_0']['w'][0,0].item())
            agent.observe_change(obs, 1, False, False)

    assert param1[0] == param1[1]
    # assert action_sequence[0] == action_sequence[1]
    # assert agent1.state_dict() == agent2.state_dict()


def test_A2C_seeded():
    action_space = gym.spaces.Discrete(10)
    obs_space = gym.spaces.Box(low=np.array([0,0]),high=np.array([1,1]))
    agent1 = A2CAgent(action_space, obs_space, 0.9, 1e-4, hk.PRNGSequence(0))
    agent2 = A2CAgent(action_space, obs_space, 0.9, 1e-4, hk.PRNGSequence(0))

    obs_sequence = [obs_space.sample() for _ in range(10)]
    action_sequence = [[],[]]
    param1 = [[],[]]
    for i,agent in enumerate([agent1,agent2]):
        agent.observe_change(obs_sequence[0], None, False, False)
        for obs in obs_sequence:
            a = agent.act()
            action_sequence[i].append(a)
            param1[i].append(agent.agent._state.params['mlp/~/linear_0']['w'][0,0].item())
            agent.observe_change(obs, 1, False, False)

    assert param1[0] == param1[1]
    assert action_sequence[0] == action_sequence[1]
    # assert agent1.state_dict() == agent2.state_dict()

@pytest.mark.skip(reason='bsuite implementation is not deterministic')
def test_DQN_seeded():
    action_space = gym.spaces.Discrete(10)
    obs_space = gym.spaces.Box(low=np.array([0,0]),high=np.array([1,1]))
    agent1 = DQNAgent(action_space, obs_space, 0.9, warmup_steps=5, batch_size=2, learning_rate=1e-4, rng=hk.PRNGSequence(0))
    agent2 = DQNAgent(action_space, obs_space, 0.9, warmup_steps=5, batch_size=2, learning_rate=1e-4, rng=hk.PRNGSequence(0))
    agent2.load_state_dict(agent1.state_dict())

    obs_sequence = [obs_space.sample() for _ in range(10)]
    action_sequence = [[],[]]
    param1 = [[],[]]
    for i,agent in enumerate([agent1,agent2]):
        agent.observe_change(obs_sequence[0], None, False, False)
        for obs in obs_sequence:
            a = agent.act()
            action_sequence[i].append(a)
            param1[i].append(agent.agent._state.params['mlp/~/linear_0']['w'][0,0].item())
            agent.observe_change(obs, 1, False, False)

    assert param1[0] == param1[1]
    assert action_sequence[0] == action_sequence[1]
    # assert agent1.state_dict() == agent2.state_dict()
