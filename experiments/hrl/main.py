import numpy as np
import gym
import torch
from tqdm import tqdm
import dill
import os

from agent.dqn_agent import DQNAgent
from agent.policy import get_greedy_epsilon_policy

from environment.wrappers import DiscreteObservationToBox

import utils

class QFunction(torch.nn.Module):
    def __init__(self, layer_sizes = [2,3,4]):
        super().__init__()
        layers = []
        in_f = 1
        for out_f in layer_sizes:
            layers.append(torch.nn.Linear(in_features=in_f,out_features=out_f))
            layers.append(torch.nn.LeakyReLU())
            in_f = out_f
        layers.append(torch.nn.Linear(in_features=in_f,out_features=4))
        self.seq = torch.nn.Sequential(*layers)
        print(self.seq)
    def forward(self, x):
        return self.seq(x)

def run_trial_steps(gamma, alpha, eps_b, eps_t, tau, directory=None,
        net_structure=[2,3,4],
        env_name='FrozenLake-v0', batch_size=32, min_replay_buffer_size=1000,
        max_steps=5000, epoch=50, test_iters=1, verbose=False):
    args = locals()
    env = gym.make(env_name)
    env = DiscreteObservationToBox(env)
    test_env = gym.make(env_name)
    test_env = DiscreteObservationToBox(test_env)

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
            q_net=QFunction(layer_sizes=net_structure)
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
            raise e

    while len(rewards) < (max_steps/epoch)+1: # Means it diverged at some point
        rewards.append([0]*test_iters)

    data = (args, rewards)

    if directory is not None:
        file_name, file_num = utils.find_next_free_file("results", "pkl", directory)
        with open(file_name, "wb") as f:
            dill.dump(data, f)

    return data

def run_gridsearch():
    directory = os.path.join(utils.get_results_directory(),__name__)
    params = {
            'gamma': [1],
            'alpha': np.logspace(np.log10(10),np.log10(.0001),num=16,endpoint=True,base=10).tolist(),
            'eps_b': [0, 0.1],
            'eps_t': [0],
            'tau': [0.001, 1],
            'env_name': ['FrozenLake-v0'],
            'batch_size': [32],
            'min_replay_buffer_size': [10000],
            'max_steps': [1000000],
            'epoch': [1000],
            'test_iters': [5],
            'verbose': [False],
            'net_structure': [[2,3,4]],
            'directory': [directory]
    }
    params = {
            'gamma': [1],
            'alpha': [0.1],
            'eps_b': [0, 0.1],
            'eps_t': [0],
            'tau': [0.001, 1],
            'env_name': ['FrozenLake-v0'],
            'batch_size': [32],
            'min_replay_buffer_size': [100],
            'max_steps': [1000],
            'epoch': [100],
            'test_iters': [5],
            'verbose': [True],
            'net_structure': [[]],
            'directory': [directory]
    }
    funcs = utils.gridsearch(params, run_trial_steps)
    utils.cc(funcs)
    return utils.get_all_results(directory)
