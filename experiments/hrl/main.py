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

    utils.save_results(args, rewards, directory=directory)
    return (args, rewards)

def run_gridsearch(proc=1):
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
            'net_structure': [(2,3,4)],
            'directory': [directory]
    }
    params = { # For testing purposes. Remove later.
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
            'verbose': [False],
            'net_structure': [()],
            'directory': [directory]
    }
    funcs = utils.gridsearch(params, run_trial_steps)
    utils.cc(funcs,proc=proc)
    return utils.get_all_results(directory)

def run(proc=2):
    utils.set_results_directory(
            os.path.join(utils.get_results_root_directory(),'hrl'))
    # Run gridsearch
    run_gridsearch(proc=proc)
    # Look through params for best performance
    def reduce(results,s=[]):
        return s + [results]
    directory = os.path.join(utils.get_results_directory(),__name__)
    results = utils.get_all_results_reduce(directory, reduce, [])

    def compute_performance_max_cum_mean(results):
        performance = {}
        for k,v in results.items():
            """
            v =
            - Trial1
                - Epoch 1
                    - Test iter 1
                    - Test iter 2
                    - ...
                - Epoch 2
                    - ...
                - ...
            - Trial 2
                - etc.
            """
            # Average all iterations under an epoch
            # Do a cumulative mean over all epochs
            # Take a max over the cumulative means for each trial
            # Take a mean over all max cum means over all trials
            max_means = []
            for trial in v:
                mean_rewards = [np.mean(epoch) for epoch in trial]
                cum_mean = np.cumsum(mean_rewards)/np.arange(1,len(mean_rewards)+1)
                max_mean = np.max(cum_mean)
                max_means.append(max_mean)
            mean_max_mean = np.mean(max_means)
            performance[k] = mean_max_mean
        return performance
    def compute_performance_abs_max(results):
        performance = {}
        for k,v in results.items():
            trial_rewards = []
            for trial in v:
                trial_rewards.append([np.mean(epoch) for epoch in trial])
            performance[k] = np.max(np.mean(trial_rewards,axis=0))
        return performance

    print('-'*80)

    performance = compute_performance_abs_max(results)
    best_param, best_performance = max(performance.items(), key=lambda x: x[1])
    print('Performance by best performance reached at any point')
    print('Best parameter set:', best_param)
    print('Best performance:', best_performance)

    print('-'*80)

    performance = compute_performance_max_cum_mean(results)
    best_param, best_performance = max(performance.items(), key=lambda x: x[1])
    print('Performance by sample complexity')
    print('Best parameter set:', best_param)
    print('Best performance:', best_performance)

    print('-'*80)

    # Plot average performance over time
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plot_dir = os.path.join(utils.get_results_root_directory(),'plots',__name__)
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    file_mapping = {}
    for i,(k,v) in enumerate(results.items()):
        trial_rewards = []
        for trial in v:
            trial_rewards.append([np.mean(epoch) for epoch in trial])
        params = dict(k)
        y = np.mean(trial_rewards,axis=0)
        x = list(range(0,params['max_steps']+1,params['epoch']))

        plt.figure()
        plt.plot(x,y)
        file_name = os.path.join(plot_dir,'%d.png'%i)
        plt.savefig(file_name)
        plt.close()

        print('Saved', file_name)
        file_mapping[k] = file_name
