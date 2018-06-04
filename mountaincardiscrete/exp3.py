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
import traceback
import datetime
import random

from agent.lstd_agent import LSTDAgent

import mountaincar 

from . import ENV_NAME
from . import MAX_REWARD
from . import MIN_REWARD
from . import LEARNED_REWARD

import utils

def run_trial(discount_factor, initial_value, num_divs,
        behaviour_eps, target_eps, trace_factor, 
        sigma, update_freq, epoch, max_iters, test_iters, directory):
    args=locals()
    try:
        env_name = ENV_NAME
        e = gym.make(env_name)
        start_time = datetime.datetime.now()

        action_space = np.array([0,1,2])
        obs_space = np.array([[-1.2, .6], [-0.07, 0.07]])
        divs = [num_divs,num_divs]
        def norm(x):
            return np.array([(s-r[0])/(r[1]-r[0]) for s,r in zip(x, obs_space)])
        def discretize(obs):
            obs = norm(obs)
            obs = [int(x*d) for x,d in zip(obs,divs)]
            obs = [x if x<d else x-1 for x,d in zip(obs,divs)]
            cp = np.cumprod([1]+divs)
            obs = np.sum([x*y for x,y in zip(obs,cp)])
            output = np.zeros([cp[-1]])
            output[obs] = 1
            return output
        agent = LSTDAgent(
                action_space=action_space,
                discount_factor=discount_factor,
                features=rbf,
                num_features=num_pos*num_vel,
                use_traces=True,
                trace_factor=trace_factor,
                sigma=sigma
        )
        agent.set_behaviour_policy("%f-epsilon" % behaviour_eps)
        agent.set_target_policy("%f-epsilon" % target_eps)

        rewards = []
        steps_to_learn = None
        for iters in range(0,max_iters+1):
            if iters % update_freq == 0:
                agent.update_weights()
            if epoch is not None and iters % epoch == 0:
                r = agent.test(e, test_iters, render=False, processors=1)
                rewards.append(r)
            agent.run_episode(e)

        while len(rewards) < (max_iters/epoch)+1: # Means it diverged at some point
            rewards.append([-200]*test_iters)

        data = (args, rewards, steps_to_learn)
        file_name, file_num = utils.find_next_free_file("results", "pkl",
                directory)
        with open(file_name, "wb") as f:
            dill.dump(data, f)

        return rewards,steps_to_learn
    except Exception as e:
        traceback.print_exc()
        print("Iterations:`",iters)

def get_directory():
    return os.path.join(utils.get_results_directory(),__name__,"part1")

def get_params_gridsearch():
    directory = get_directory()
    behaviour_eps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    target_eps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    trace_factors = [0, 0.25, 0.5, 0.75, 1]
    sigmas = [0, 0.25, 0.5, 0.75, 1]

    keys = ['behaviour_eps', 'target_eps', 'sigma','trace_factor']
    params = []
    for vals in itertools.product(behaviour_eps, target_eps, sigmas, trace_factors):
        d = dict(zip(keys,vals))
        d['discount_factor'] = 1
        d['initial_value'] = 0
        d['num_pos'] = 8
        d['num_vel'] = 8
        d['update_freq'] = 1
        d['epoch'] = 50
        d['max_iters'] = 3000
        d['test_iters'] = 1
        d["directory"] = os.path.join(directory, "l%f"%d['trace_factor'])
        params.append(d)
    return params

def get_plot_params_final_rewards():
    x_axis = 'behaviour_eps'
    best_of = []
    average = []
    each_curve = ['target_eps']
    each_plot = ['sigma', 'trace_factor', 'update_freq']
    file_name_template = 'graph-s{sigma}-l{trace_factor}-u{update_freq}.png'
    label_template = 'epsilon={target_eps}'
    xlabel = 'Behaviour Epsilon'
    ylabel = 'Cumulative reward'
    return locals()

def get_plot_params_best():
    file_name = 'graph-best.png'
    label_template = 'LSTD'
    return locals()

def get_plot_params_gridsearch():
    file_name = 'graph-gridsearch.png'
    axis_params = ['sigma', 'trace_factor', 'behaviour_eps', 'target_eps']
    axis_labels = ['$\sigma$', '$\lambda$', '$\epsilon_b$', '$\epsilon_t$']
    return locals()
