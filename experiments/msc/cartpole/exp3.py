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

from . import ENV_NAME
from . import MAX_REWARD
from . import MIN_REWARD
from . import LEARNED_REWARD

from . import features
from . import utils

import graph
import utils

def run_trial(gamma, upd_freq, eps_b, eps_t, sigma, lam, directory=None,
        max_iters=5000, epoch=50, test_iters=1):
    """
    Run the learning algorithm on CartPole and return the number of
    iterations needed to learn the task.
    """
    args = locals()
    env_name = ENV_NAME
    e = gym.make(env_name)
    e = features.Identity2(e)

    action_space = np.array([0,1])
    agent = LSTDAgent(
            action_space=action_space,
            num_features=e.observation_space.shape[0],
            discount_factor=gamma,
            use_importance_sampling=False,
            use_traces=True,
            trace_factor=lam,
            sigma=sigma
    )
    agent.set_behaviour_policy("%.3f-epsilon"%eps_b)
    agent.set_target_policy("%.3f-epsilon"%eps_t)

    rewards = []
    steps_to_learn = None
    try:
        for iters in range(0,max_iters+1):
            if iters % upd_freq == 0:
                agent.update_weights()
            if epoch is not None:
                if iters % epoch == 0:
                    r = agent.test(e, test_iters, render=False, processors=1)
                    rewards.append(r)
                    if np.mean(r) >= LEARNED_REWARD:
                        if steps_to_learn is None:
                            steps_to_learn = iters
            else:
                if r >= LEARNED_REWARD:
                    if steps_to_learn is None:
                        steps_to_learn = iters
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
    return os.path.join(utils.get_results_directory(),__name__,"part1")

def get_params_gridsearch():
    update_frequencies = [1]
    behaviour_eps = [0, 0.5]
    target_eps = [0]
    trace_factors = [0]
    sigmas = [0]
    #update_frequencies = [1,50,200]
    #behaviour_eps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    #target_eps = [0, 0.1, 0.2, 0.3, 0.4]
    #trace_factors = [0, 0.25, 0.5, 0.75, 1]
    #sigmas = [0, 0.25, 0.5, 0.75, 1]

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

def get_plot_params_final_rewards():
    x_axis = 'eps_b'
    best_of = []
    average = []
    each_curve = ['eps_t']
    each_plot = ['sigma', 'lam', 'upd_freq']
    file_name_template = 'graph-s{sigma}-l{lam}-u{upd_freq}.png'
    label_template = 'epsilon={eps_t}'
    xlabel = 'Behaviour Epsilon'
    ylabel = 'Cumulative reward'
    return locals()

def get_plot_params_best():
    file_name = 'graph-best.png'
    label_template = 'LSTD'
    param_filters = []
    return locals()

def get_plot_params_gridsearch():
    file_name = 'graph-gridsearch.png'
    axis_params = ['sigma', 'lam', 'eps_b', 'eps_t']
    axis_labels = ['$\sigma$', '$\lambda$', '$\epsilon_b$', '$\epsilon_t$']
    return locals()

def get_param_filters():
    return [()]
