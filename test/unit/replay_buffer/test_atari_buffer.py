import pytest

import numpy as np
import gym
from gym.wrappers import FrameStack, AtariPreprocessing, TimeLimit

from rl.agent.replay_buffer import AtariReplayBuffer, ReplayBuffer

@pytest.mark.parametrize(['num_episodes'],[[0],[1],[2]])
def test_no_overwrite(num_episodes):
    """ Check that the AtariReplayBuffer behaves the same as the ReplayBuffer.
    The replay buffer is large, so all data will be preserved.
    """

    atari_buffer = AtariReplayBuffer(max_size=1000)
    baseline_buffer = ReplayBuffer(max_size=1000)

    env = gym.make('PongNoFrameskip-v4')
    env = AtariPreprocessing(env)
    env = FrameStack(env, 4)
    env = TimeLimit(env, 10)

    for _ in range(num_episodes):
        done = False
        obs0 = env.reset()
        while not done:
            action = env.action_space.sample()
            obs1, reward, done, _ = env.step(action)

            atari_buffer.add_transition(
                    obs0, action, reward, obs1, done)
            baseline_buffer.add_transition(
                    np.array(obs0), action, reward, np.array(obs1), done)
            
            obs0 = obs1

    assert len(atari_buffer) == len(baseline_buffer)

    for i in range(len(atari_buffer)):
        a_obs, a_action, a_reward, _, a_term = atari_buffer[i]
        b_obs, b_action, b_reward, _, b_term = baseline_buffer[i]
        assert (a_obs.numpy() == b_obs).all()
        assert a_action == b_action
        assert a_reward == b_reward
        assert a_term == b_term

@pytest.mark.parametrize(['num_episodes'],[[0],[1],[2]])
def test_with_overwrite(num_episodes):
    """ Check that the AtariReplayBuffer behaves the same as the ReplayBuffer.
    The replay buffer is smaller than all the data that needs to be recorded, so old data will need to be overwritten by newer data.
    """

    atari_buffer = AtariReplayBuffer(max_size=15)
    baseline_buffer = ReplayBuffer(max_size=15)

    env = gym.make('PongNoFrameskip-v4')
    env = AtariPreprocessing(env)
    env = FrameStack(env, 4)
    env = TimeLimit(env, 10)

    for _ in range(num_episodes):
        done = False
        obs0 = env.reset()
        while not done:
            action = env.action_space.sample()
            obs1, reward, done, _ = env.step(action)

            atari_buffer.add_transition(
                    obs0, action, reward, obs1, done)
            baseline_buffer.add_transition(
                    np.array(obs0), action, reward, np.array(obs1), done)
            
            obs0 = obs1

    assert len(atari_buffer) == len(baseline_buffer)

    for i in range(len(atari_buffer)):
        a_obs, a_action, a_reward, _, a_term = atari_buffer[i]
        b_obs, b_action, b_reward, _, b_term = baseline_buffer[i]
        assert (a_obs.numpy() == b_obs).all()
        assert a_action == b_action
        assert a_reward == b_reward
        assert a_term == b_term

@pytest.mark.parametrize(['num_episodes'],[[0],[1],[2]])
def test_smaller_than_one_episode(num_episodes):
    """ Check that the AtariReplayBuffer behaves the same as the ReplayBuffer.
    The replay buffer size is less than that of one episode.
    """

    atari_buffer = AtariReplayBuffer(max_size=5)
    baseline_buffer = ReplayBuffer(max_size=5)

    env = gym.make('PongNoFrameskip-v4')
    env = AtariPreprocessing(env)
    env = FrameStack(env, 4)
    env = TimeLimit(env, 10)

    for _ in range(num_episodes):
        done = False
        obs0 = env.reset()
        while not done:
            action = env.action_space.sample()
            obs1, reward, done, _ = env.step(action)

            atari_buffer.add_transition(
                    obs0, action, reward, obs1, done)
            baseline_buffer.add_transition(
                    np.array(obs0), action, reward, np.array(obs1), done)
            
            obs0 = obs1

    assert len(atari_buffer) == len(baseline_buffer)

    for i in range(len(atari_buffer)):
        a_obs, a_action, a_reward, _, a_term = atari_buffer[i]
        b_obs, b_action, b_reward, _, b_term = baseline_buffer[i]
        assert (a_obs.numpy() == b_obs).all()
        assert a_action == b_action
        assert a_reward == b_reward
        assert a_term == b_term
