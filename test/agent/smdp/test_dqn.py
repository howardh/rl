import os
import dill
import gym
from gym.wrappers import FrameStack, AtariPreprocessing
import torch
from rl.agent.smdp.dqn import DQNAgent, epsilon_greedy, make_agent_from_deploy_state

def test_epsilon_greedy_one_value():
    for val in [0,1,-10]:
        for eps in [0,0.5,1]:
            assert epsilon_greedy(torch.Tensor([val]), eps=eps) == torch.tensor([1.])

def test_epsilon_greedy_two_values_different():
    assert (epsilon_greedy(torch.Tensor([1,2]), eps=0) == torch.tensor([0., 1.])).all()
    assert (epsilon_greedy(torch.Tensor([3,2]), eps=0) == torch.tensor([1., 0.])).all()

    assert (epsilon_greedy(torch.Tensor([1,2]), eps=0.1) == torch.tensor([.1, .9])).all()
    assert (epsilon_greedy(torch.Tensor([3,2]), eps=0.1) == torch.tensor([.9, .1])).all()

def test_epsilon_greedy_two_values_same():
    assert (epsilon_greedy(torch.Tensor([2,2]), eps=0) == torch.tensor([.5,.5])).all()
    assert (epsilon_greedy(torch.Tensor([0,0]), eps=0.1) == torch.tensor([.5,.5])).all()

def test_epsilon_greedy_three_values():
    assert (epsilon_greedy(torch.Tensor([2,1,1]), eps=0) == torch.tensor([1.,0.,0.])).all()
    assert (epsilon_greedy(torch.Tensor([2,1,1]), eps=0.1) == torch.tensor([.9,0.05,0.05])).all()
    assert (epsilon_greedy(torch.Tensor([2,2,1]), eps=0) == torch.tensor([.5,.5,0.])).all()
    assert (epsilon_greedy(torch.Tensor([0,0,0]), eps=0.1) == torch.tensor([1.])/3).all()

def test_deploy_state():
    env = gym.make('PongNoFrameskip-v4')
    env = AtariPreprocessing(env)
    env = FrameStack(env, 4)
    agent1 = DQNAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            target_eps=0,
    )
    state = agent1.state_dict_deploy()
    agent2 = make_agent_from_deploy_state(state)

    obs = env.reset()
    agent1.observe(obs, testing=True)
    agent2.observe(obs, testing=True)

    a1 = agent1.act(testing=True)
    a2 = agent2.act(testing=True)

    assert a1 == a2

def test_deploy_state_from_file(tmpdir):
    env = gym.make('PongNoFrameskip-v4')
    env = AtariPreprocessing(env)
    env = FrameStack(env, 4)
    agent1 = DQNAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            target_eps=0,
    )
    state = agent1.state_dict_deploy()

    filename = os.path.join(tmpdir,'file.pkl')
    with open(filename, 'wb') as f:
        dill.dump(state, f)

    agent2 = make_agent_from_deploy_state(filename)

    obs = env.reset()
    agent1.observe(obs, testing=True)
    agent2.observe(obs, testing=True)

    a1 = agent1.act(testing=True)
    a2 = agent2.act(testing=True)

    assert a1 == a2
