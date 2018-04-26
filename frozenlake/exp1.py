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

from agent.discrete_agent import TabularAgent

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
from frozenlake import exp2

import utils

def run_trial(alpha, gamma, eps_b, eps_t, sigma, lam,
        directory=None, max_iters=5000, epoch=50, test_iters=1):
    args = locals()
    env_name = ENV_NAME
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = TabularAgent(
            action_space=action_space,
            discount_factor=gamma,
            learning_rate=alpha,
            trace_factor=lam,
            sigma=sigma)
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

get_params_gridsearch = exp2.get_params_gridsearch

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

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = []

    fig, ax = plt.subplots(1,1)
    ax.set_ylim([MIN_REWARD,MAX_REWARD])
    ax.set_xlabel('Behaviour epsilon')
    ax.set_ylabel('Cumulative reward')
    for score_function in [get_mean_rewards, get_final_rewards]:
        params = get_params_best(directory, score_function, 1)[0]
        print("Plotting params: ", params)

        series = utils.get_series_with_params_pkl(directory, params)
        mean = np.mean(series, axis=0)
        std = np.std(series, axis=0)
        epoch = params['epoch']
        x = [i*epoch for i in range(len(mean))]
        data.append((x, mean, std, 'SGD'))
        ax.plot(x,mean,label='SGD')
    ax.legend(loc='best')
    file_name = os.path.join(directory, 'graph-best.png')
    print("Saving file %s" % file_name)
    plt.savefig(file_name)
    plt.close(fig)

    return data

