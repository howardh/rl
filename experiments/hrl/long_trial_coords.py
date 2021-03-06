import numpy as np
import gym
import torch
from tqdm import tqdm
import dill
import os
import itertools

from agent.dqn_agent import DQNAgent
from agent.policy import get_greedy_epsilon_policy

from environment.wrappers import FrozenLakeToCoords

from .model import QFunction
from .long_trial import plot

import utils

def run_trial(gamma, alpha, eps_b, eps_t, tau, directory=None,
        net_structure=[2,3,4],
        env_name='FrozenLake-v0', batch_size=32, min_replay_buffer_size=1000,
        epoch=50, test_iters=1, verbose=False):
    args = locals()
    env = gym.make(env_name)
    env = FrozenLakeToCoords(env)
    test_env = gym.make(env_name)
    test_env = FrozenLakeToCoords(test_env)

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
            target_policy=get_greedy_epsilon_policy(eps_t),
            q_net=QFunction(layer_sizes=net_structure,input_size=env.observation_space.shape[0])
    )

    # Create file to save results
    results_file_path = utils.save_results(args,
            {'rewards': [], 'state_action_values': []},
            directory=directory)

    rewards = []
    state_action_values = []
    done = True
    step_range = itertools.count()
    if verbose:
        step_range = tqdm(step_range)
    try:
        for steps in step_range:
            # Run tests
            if steps % epoch == 0:
                r,sa_vals = agent.test(test_env, test_iters, render=False, processors=1)
                rewards.append(r)
                state_action_values.append(sa_vals)
                if verbose:
                    tqdm.write('steps %d \t Reward: %f' % (steps, np.mean(r)))
                utils.save_results(args,
                        {'rewards': rewards, 'state_action_values': state_action_values},
                        file_path=results_file_path)

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
            raise e
    except KeyboardInterrupt:
        tqdm.write('Keyboard Interrupt')

    return (args, rewards, state_action_values)

def run():
    utils.set_results_directory(
            os.path.join(utils.get_results_root_directory(),'hrl'))
    directory = os.path.join(utils.get_results_directory(),__name__)
    plot_directory = os.path.join(utils.get_results_directory(),'plots',__name__)

    run_trial(gamma=1,alpha=0.01,eps_b=0,eps_t=0,tau=0.01,net_structure=(10,10),batch_size=256,epoch=1000,test_iters=10,verbose=True,directory=directory)
    plot(results_directory=directory,plot_directory=plot_directory)
