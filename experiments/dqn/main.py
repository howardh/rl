import numpy as np
import gym
import torch
from tqdm import tqdm
import dill

import utils
from agent.dqn_agent import DQNAgent
from agent.policy import get_greedy_epsilon_policy

from environment.wrappers import AtariPreprocessing, FrameStack

def run_trial(gamma, alpha, eps_b, eps_t, directory=None,
        max_iters=5000, epoch=50, test_iters=1):
    args = locals()
    env_name = 'Breakout-v0'
    env = gym.make(env_name)
    env = AtariPreprocessing(env)
    env = FrameStack(env,4)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    agent = DQNAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            learning_rate=alpha,
            discount_factor=gamma,
            device=device,
            behaviour_policy=get_greedy_epsilon_policy(eps_b),
            target_policy=get_greedy_epsilon_policy(eps_t)
    )

    rewards = []
    try:
        for iters in range(0,max_iters+1):
            # Run tests
            if iters % epoch == 0:
                r = agent.test(env, test_iters, render=False, processors=1)
                rewards.append(r)
                print('iter %d \t Reward: %f' % (iters, np.mean(r)))
            # Run an episode
            obs = env.reset()
            action = agent.act(obs)
            obs2 = None
            done = False
            step_count = 0
            reward_sum = 0
            while not done:
                step_count += 1

                obs2, reward2, done, _ = env.step(action)
                action2 = agent.act(obs2)

                agent.observe_step(obs, action, reward2, obs2, terminal=done)
                agent.train(batch_size=32,iterations=1)

                # Next time step
                obs = obs2
                action = action2
    except ValueError as e:
        tqdm.write(str(e))
        tqdm.write("Diverged")

    while len(rewards) < (max_iters/epoch)+1: # Means it diverged at some point
        rewards.append([0]*test_iters)

    data = (args, rewards, steps_to_learn)
    file_name, file_num = utils.find_next_free_file("results", "pkl", directory)
    with open(file_name, "wb") as f:
        dill.dump(data, f)

def run_trial_steps(gamma, alpha, eps_b, eps_t, tau, directory=None,
        env_name = 'Breakout-v0', batch_size=32, min_replay_buffer_size=10000,
        max_steps=5000, epoch=50, test_iters=1, verbose=False):
    args = locals()
    env = gym.make(env_name)
    env = AtariPreprocessing(env)
    env = FrameStack(env,4)
    test_env = gym.make(env_name)
    test_env = AtariPreprocessing(test_env)
    test_env = FrameStack(test_env,4)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    agent = DQNAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            learning_rate=alpha,
            discount_factor=gamma,
            polyak_rate=tau,
            device=device,
            behaviour_policy=get_greedy_epsilon_policy(eps_b),
            target_policy=get_greedy_epsilon_policy(eps_t)
    )

    rewards = []
    done = True
    step_range = range(0,max_steps+1)
    if verbose:
        step_range = tqdm(step_range)
    try:
        for steps in step_range:
            # Run tests
            if steps % epoch == 0:
                r = agent.test(test_env, test_iters, render=False, processors=1)
                rewards.append(r)
                if verbose:
                    tqdm.write('steps %d \t Reward: %f' % (steps, np.mean(r)))

            # Linearly Anneal epsilon
            agent.behaviour_policy = get_greedy_epsilon_policy((1-min(steps/1000000,1))*(1-eps_b)+eps_b)

            # Run step
            if done:
                obs = env.reset()
            action = agent.act(obs)

            obs2, reward2, done, _ = env.step(action)
            agent.observe_step(obs, action, reward2, obs2, terminal=done)

            # Update weights
            if steps >= min_replay_buffer_size:
                agent.train(batch_size=batch_size,iterations=1)

            # Next time step
            obs = obs2
    except ValueError as e:
        if verbose:
            tqdm.write(str(e))
            tqdm.write("Diverged")

    while len(rewards) < (max_steps/epoch)+1: # Means it diverged at some point
        rewards.append([0]*test_iters)

    data = (args, rewards)
    file_name, file_num = utils.find_next_free_file("results", "pkl", directory)
    with open(file_name, "wb") as f:
        dill.dump(data, f)
