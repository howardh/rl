import numpy as np
import gym
import itertools
import pandas
import dill
import csv
import os
from tqdm import tqdm
import time
import operator
import pprint
import sys
import traceback
import random

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent

import cartpole 
from cartpole import ENV_NAME
from cartpole import MAX_REWARD
from cartpole import MIN_REWARD
from cartpole import LEARNED_REWARD

from cartpole import features
from cartpole import utils
from cartpole import experiments
from cartpole import exp3

from cartpole.experiments import get_mean_rewards
from cartpole.experiments import get_final_rewards
from cartpole.experiments import get_params_best

import utils

run_trial = exp3.run_trial

get_directory = exp3.get_directory

def get_params_best(directory, score_function, n=1):
    out0 = cartpole.experiments.get_params_best(
            directory, score_function, n, {'sigma': 0.0})
    out1 = cartpole.experiments.get_params_best(
            directory, score_function, n, {'sigma': 1.0})
    return out0+out1

def get_params_gridsearch():
    update_frequencies = [1,50,200]
    behaviour_eps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    target_eps = [0, 0.1, 0.2, 0.3, 0.4]
    trace_factors = [0, 0.25, 0.5, 0.75, 1]
    sigmas = [0, 1]

    keys = ['upd_freq','eps_b', 'eps_t', 'sigma','lam']
    params = []
    for vals in itertools.product(update_frequencies, behaviour_eps, target_eps, sigmas, trace_factors):
        d = dict(zip(keys,vals))
        d['gamma'] = 0.9
        d['epoch'] = 50
        d['max_iters'] = 5000
        d['test_iters'] = 1
        params.append(d)
    return params

def plot_final_rewards(directory=None):
    if directory is None:
        directory=get_directory()
    # Check that the experiment has been run and that results are present
    if not os.path.isdir(directory):
        print("No results to parse in %s" % directory)
        return None

    data = utils.parse_results_pkl(directory, LEARNED_REWARD)
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
    each_plot = ['sigma', 'lam', 'upd_freq']
    file_name_template = 'graph-s{sigma}-l{lam}-u{upd_freq}.png'
    label_template = 'epsilon={eps_t}'

    p_dict = dict([(k,next(iter(v))) for k,v in all_params.items()])
    # Loop over plots
    for p1 in itertools.product(*[all_params[k] for k in each_plot]):
        for k,v in zip(each_plot,p1):
            p_dict[k] = v
        fig, ax = plt.subplots(1,1)
        ax.set_ylim([MIN_REWARD,MAX_REWARD])
        ax.set_xlabel('Behaviour epsilon')
        ax.set_ylabel('Cumulative reward')
        # Loop over curves in a plot
        for p2 in itertools.product(*[sorted(all_params[k]) for k in each_curve]):
            for k,v in zip(each_curve,p2):
                p_dict[k] = v
            x = []
            y = []
            for px in sorted(all_params[x_axis]):
                p_dict[x_axis] = px
                param_vals = tuple([p_dict[k] for k in keys])
                x.append(float(px))
                y.append(data.loc[param_vals])
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
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Cumulative reward')
    for score_function in [get_mean_rewards, get_final_rewards]:
        for params in get_params_best(directory, score_function, 1):
            print("Plotting params: ", params)

            series = utils.get_series_with_params_pkl(directory, params)
            mean = np.mean(series, axis=0)
            std = np.std(series, axis=0)
            epoch = params['epoch']
            x = [i*epoch for i in range(len(mean))]
            data.append((x, mean, std, 'LSTD sigma=%s'%params['sigma']))
            ax.plot(x,mean,label='LSTD sigma=%s'%params['sigma'])
    ax.legend(loc='best')
    file_name = os.path.join(directory, 'graph-best5.png')
    print("Saving file %s" % file_name)
    plt.savefig(file_name)
    plt.close(fig)

    return data
