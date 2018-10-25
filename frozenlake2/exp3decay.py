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

from agent.lstd_agent import LSTDAgent

import frozenlake2
import frozenlake2.features
import frozenlake2.utils

from frozenlake2 import ENV_NAME
from frozenlake2 import MAX_REWARD
from frozenlake2 import MIN_REWARD
from frozenlake2 import LEARNED_REWARD

import graph
import utils

def run_trial(gamma, upd_freq, eps_b, eps_t, sigma, lam, decay,
        directory=None, max_iters=5000, epoch=50, test_iters=1):
    """
    Run the learning algorithm on FrozenLake and return the number of
    iterations needed to learn the task.
    """
    args = locals()
    env_name = ENV_NAME
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = LSTDAgent(
            action_space=action_space,
            num_features=frozenlake2.features.ONE_HOT_NUM_FEATURES,
            discount_factor=gamma,
            features=frozenlake2.features.one_hot,
            use_importance_sampling=False,
            use_traces=True,
            trace_factor=lam,
            sigma=sigma,
            cuda=False,
            trace_type='replacing',
            decay=decay
    )
    agent.set_behaviour_policy("%.3f-epsilon"%eps_b)
    agent.set_target_policy("%.3f-epsilon"%eps_t)

    rewards = []
    steps_to_learn = None
    try:
        for iters in range(0,max_iters+1):
            if iters % upd_freq == 0:
                agent.update_weights()
            if epoch is not None and iters % epoch == 0:
                r = agent.test(e, test_iters, render=False, processors=1)
                rewards.append(r)
            agent.run_episode(e)
    except ValueError as e:
        tqdm.write(str(e))
        tqdm.write("Diverged")
    # If it diverged at some point...
    while len(rewards) < (max_iters/epoch)+1: 
        rewards.append([0]*test_iters)

    data = (args, rewards, steps_to_learn)
    file_name, file_num = utils.find_next_free_file("results", "pkl", directory)
    with open(file_name, "wb") as f:
        dill.dump(data, f)

def get_directory():
    return os.path.join(utils.get_results_directory(),__name__,"decay3")

def get_params_gridsearch():
    #behaviour_eps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    behaviour_eps = [0, 0.2, 0.4, 0.6, 0.8, 1]
    target_eps = [0, 0.1, 0.2, 0.3, 0.4]
    trace_factors = [0, 0.25, 0.5, 0.75, 1]
    sigmas = [0, 0.25, 0.5, 0.75, 1]
    decay = [1, 0.999, 0.995, 0.99, 0.95, 0.9]

    keys = ['eps_b', 'eps_t', 'sigma', 'lam','decay']
    params = []
    for vals in itertools.product(behaviour_eps, target_eps, sigmas, trace_factors, decay):
        d = dict(zip(keys,vals))
        d['gamma'] = 1
        d['epoch'] = 1
        d['max_iters'] = 500
        d['test_iters'] = 50
        d['upd_freq'] = 1
        params.append(d)
    return params

def get_plot_params_final_rewards():
    x_axis = 'eps_b'
    best_of = []
    average = []
    each_curve = ['eps_t']
    each_plot = ['sigma', 'lam', 'upd_freq']
    file_name_template = 'graph-s{sigma}-l{lam}-u{upd_freq}.png'
    label_template = '$\epsilon={eps_t}'
    xlabel = 'Behaviour Epsilon'
    ylabel = 'Cumulative reward'
    return locals()

def get_plot_params_best():
    file_name = 'graph-best.png'
    label_template = 'LSTD $\sigma={sigma}$, decay={decay}'
    #param_filters = []
    #param_filters = [{'decay': 1.0}, {'decay': 0.8}, {'decay': 0.6}]
    param_filters = [{'decay': 1.0}, {'decay': 0.999}, {'decay': 0.995}, {'decay': 0.99}, {'decay': 0.95}, {'decay': 0.9}]
    param_filters = [{'decay': 1.0}, {'decay': 0.999}, {'decay': 0.995}, {'decay': 0.99}, {'decay': 0.9}]
    return locals()

def get_plot_params_gridsearch():
    file_name = 'graph-gridsearch.png'
    axis_params = ['sigma', 'lam', 'eps_b', 'eps_t']
    axis_labels = ['$\sigma$', '$\lambda$', '$\epsilon_b$', '$\epsilon_t$']
    return locals()
