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

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent

import mountaincar 
import mountaincar.features
import mountaincar.utils

from mountaincar import ENV_NAME
from mountaincar import MAX_REWARD
from mountaincar import MIN_REWARD
from mountaincar import LEARNED_REWARD

import utils

def run_trial(discount_factor, initial_value, num_pos,
        num_vel, behaviour_eps, target_eps, trace_factor, 
        sigma, update_freq, epoch, max_iters, test_iters, directory):
    args=locals()
    try:
        env_name = ENV_NAME
        e = gym.make(env_name)
        start_time = datetime.datetime.now()

        action_space = np.array([0,1,2])
        obs_space = np.array([[-1.2, .6], [-0.07, 0.07]])
        centres = np.array(list(itertools.product(
                np.linspace(0,1,num_pos),
                np.linspace(0,1,num_vel))))
        def norm(x):
            return np.array([(s-r[0])/(r[1]-r[0]) for s,r in zip(x, obs_space)])
        def rbf(x):
            x = norm(x)
            dist = np.power(centres-x, 2).sum(axis=1,keepdims=True)
            return np.exp(-100*dist)
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
            if iters % upd_freq == 0:
                agent.update_weights()
            if epoch is not None and iters % epoch == 0:
                r = agent.test(e, test_iters, render=False, processors=1)
                rewards.append(r)
            agent.run_episode(e)

        while iters < max_iters: # Means it diverged at some point
            iters += 1
            rewards.append(None)

        data = (args, rewards, steps_to_learn)
        file_name, file_num = utils.find_next_free_file("results", "pkl",
                directory)
        with open(file_name, "wb") as f:
            dill.dump(data, f)

        return rewards,steps_to_learn
    except Exception as e:
        traceback.print_exc()
        print("Iterations:`",iters)

def get_params_gridsearch():
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

def plot_final_rewards(directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),exp.__name__,"part1")
    # Check that the experiment has been run and that results are present
    if not os.path.isdir(directory):
        print("No results to parse in %s" % directory)
        return None

    data = utils.parse_results_pkl(directory, LEARNED_REWARD)
    data = data.apply(lambda row: row.MRS/row.Count, axis=1)
    keys = data.index.names
    all_params = dict([(k, set(data.index.get_level_values(k))) for k in keys])

    #def run_trial(discount_factor, initial_value, num_pos,
    #        num_vel, behaviour_eps, target_eps, trace_factor, 
    #        sigma, update_freq, epoch, max_iters, test_iters, directory):
    # Graph stuff
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x_axis = 'behaviour_eps'
    best_of = []
    average = []
    each_curve = ['target_eps']
    each_plot = ['sigma', 'trace_factor', 'update_freq']
    file_name_template = 'graph-s{sigma}-l{trace_factor}-u{update_freq}.png'
    label_template = 'epsilon={target_eps}'

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
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")

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
        data.append((x, mean, std, 'LSTD'))
        ax.plot(x,mean,label='LSTD')
    ax.legend(loc='best')
    file_name = os.path.join(directory, 'graph-best.png')
    print("Saving file %s" % file_name)
    plt.savefig(file_name)
    plt.close(fig)

    return data

