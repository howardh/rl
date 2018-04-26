import numpy as np
import gym
import itertools
import pandas
import multiprocessing
import dill
import csv
import os
from tqdm import tqdm
import time
import operator
import pprint
import sys

from agent.linear_agent import LinearAgent

import frozenlake
import frozenlake.features
import frozenlake.utils
from frozenlake.experiments import get_mean_rewards
from frozenlake.experiments import get_final_rewards
from frozenlake.experiments import get_params_best

from frozenlake import ENV_NAME
from frozenlake import MAX_REWARD
from frozenlake import MIN_REWARD
from frozenlake import LEARNED_REWARD

import graph
import utils

def run_trial(alpha, gamma, eps_b, eps_t, sigma, lam,
        directory=None, max_iters=5000, epoch=50, test_iters=1):
    """
    Run the learning algorithm on FrozenLake and return the number of
    iterations needed to learn the task.
    """
    args = locals()
    env_name = ENV_NAME
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = LinearAgent(
            action_space=action_space,
            num_features=frozenlake.features.ONE_HOT_NUM_FEATURES,
            discount_factor=gamma,
            learning_rate=alpha,
            features=frozenlake.features.one_hot,
            trace_factor=lam,
            sigma=sigma,
    )
    agent.set_behaviour_policy("%.3f-epsilon"%eps_b)
    agent.set_target_policy("%.3f-epsilon"%eps_t)

    rewards = []
    steps_to_learn = None
    try:
        for iters in range(0,max_iters+1):
            if epoch is not None and iters % epoch == 0:
                r = agent.test(e, test_iters, render=False, processors=1)
                rewards.append(r)
            agent.run_episode(e)
    except ValueError as e:
        tqdm.write(str(e))
        tqdm.write("Diverged")

    while len(rewards) < (max_iters/epoch)+1: 
        rewards.append([0]*test_iters)

    data = (args, rewards, steps_to_learn)
    file_name, file_num = utils.find_next_free_file("results", "pkl", directory)
    with open(file_name, "wb") as f:
        dill.dump(data, f)

def get_directory():
    return os.path.join(utils.get_results_directory(),__name__,"part1")

def get_params_gridsearch():
    behaviour_eps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    target_eps = [0, 0.1, 0.2, 0.3, 0.4]
    trace_factors = [0, 0.25, 0.5, 0.75, 1]
    sigmas = [0, 0.25, 0.5, 0.75, 1]
    #learning_rate = np.logspace(np.log10(10),np.log10(.001),num=13,endpoint=True,base=10).tolist()
    learning_rate = np.logspace(np.log10(1),np.log10(.001),num=10,endpoint=True,base=10).tolist()

    keys = ['eps_b', 'eps_t', 'sigma','lam', 'alpha']
    params = []
    for vals in itertools.product(behaviour_eps, target_eps, sigmas,
            trace_factors, learning_rate):
        d = dict(zip(keys,vals))
        d['gamma'] = 1
        d['epoch'] = 10
        d['max_iters'] = 2000
        d['test_iters'] = 50
        params.append(d)
    return params

def plot_final_rewards(directory=None):
    if directory is None:
        directory=get_directory()
    # Check that the experiment has been run and that results are present
    if not os.path.isdir(directory):
        print("No results to parse in %s" % directory)
        return None

    data = utils.parse_results(directory, LEARNED_REWARD)
    data = data.apply(lambda row: row.MRS/row.Count, axis=1)
    keys = data.index.names
    all_params = dict([(k, set(data.index.get_level_values(k))) for k in keys])

    # Graph stuff
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x_axis = 'eps_b'
    best_of = []
    average = []
    each_curve = ['eps_t']
    each_plot = ['sigma', 'lam', 'alpha']
    file_name_template = 'graph-s{sigma}-l{lam}-a{alpha}.png'
    label_template = 'epsilon={eps_t}'

    print(all_params)

    p_dict = dict([(k,next(iter(v))) for k,v in all_params.items()])
    # Loop over plots
    for pp in itertools.product(*[all_params[k] for k in each_plot]):
        for k,v in zip(each_plot,pp):
            p_dict[k] = v
        fig, ax = plt.subplots(1,1)
        ax.set_ylim([MIN_REWARD,MAX_REWARD])
        ax.set_xlabel('Behaviour epsilon')
        ax.set_ylabel('Cumulative reward')
        # Loop over curves in a plot
        for pc in itertools.product(*[sorted(all_params[k]) for k in each_curve]):
            for k,v in zip(each_curve,pc):
                p_dict[k] = v
            x = []
            y = []
            for px in sorted(all_params[x_axis]):
                p_dict[x_axis] = px
                vals = []
                for pb in itertools.product(*[sorted(all_params[k]) for k in best_of]):
                    for k,v in zip(each_curve,pb):
                        p_dict[k] = v
                    param_vals = tuple([p_dict[k] for k in keys])
                    vals.append(data.loc[param_vals])
                x.append(float(px))
                y.append(np.max(vals))
            ax.plot(x,y,label=label_template.format(**p_dict))
        ax.legend(loc='best')
        file_name = os.path.join(directory, file_name_template.format(**p_dict))
        print("Saving file %s" % file_name)
        plt.savefig(file_name)
        plt.close(fig)

    return data

def plot_best(directory=None):
    if directory is None:
        directory=get_directory()

    data = []
    for score_function in [get_mean_rewards, get_final_rewards]:
        params = get_params_best(directory, score_function, 1)[0]
        print("Plotting params: ", params)
        data.append(graph.get_data(params, directory, label='SGD'))
    graph.graph_data(data, 'graph-best.png', directory)

    return data

def plot_custom(directory=None):
    if directory is None:
        directory=get_directory()

    all_params = [
            {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.0, 'lam': 0.25},
            {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.1, 'lam': 0.25},
            {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.2, 'lam': 0.25},
            {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.3, 'lam': 0.25}
    ]
    all_params = [
            {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.0, 'lam': 0.0},
            {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.0, 'lam': 0.25},
            {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.0, 'lam': 0.5},
            {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.0, 'lam': 0.75},
            {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.0, 'lam': 1.0}
    ]
    data = []
    for params in all_params:
        print("Plotting params: ", params)
        #data.append(graph.get_data(params, directory,
        #        label='SGD eps_t=%f'%params['eps_t']))
        data.append(graph.get_data(params, directory,
                label='SGD lam=%f'%params['lam']))
    graph.graph_data(data, 'graph-custom.png', directory)

    return data
