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

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent

import frozenlake
import frozenlake.features
import frozenlake.utils

import utils

#discount_factors = ['1', '0.9', '0.5']
discount_factors = ['1']
update_frequencies = ['50', '200', '500']
behaviour_epsilons = ['1', '0.5', '0']
target_epsilons = ['0', '0.01', '0.05']
sigmas = ['0', '0.5', '1']
#trace_factors = ['0.01', '0.25', '0.5', '0.75', '0.99']
trace_factors = ['0.25', '0.75']

def _run_trial(gamma, upd_freq, eps_b, eps_t, sigma, lam,
        directory=None, max_iters=5000, test_iters=1):
    """
    Run the learning algorithm on FrozenLake and return the number of
    iterations needed to learn the task.
    """
    env_name = 'FrozenLake-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = LSTDAgent(
            action_space=action_space,
            num_features=frozenlake.features.ONE_HOT_NUM_FEATURES,
            discount_factor=gamma,
            features=frozenlake.features.one_hot,
            use_importance_sampling=False,
            use_traces=True,
            trace_factor=lam,
            sigma=sigma,
            cuda=False
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

def get_params_custom():
    params = []
    return params

def get_params_gridsearch():
    update_frequencies = [1,50,200]
    behaviour_eps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    target_eps = [0, 0.1, 0.2, 0.3, 0.4]
    trace_factors = [0, 0.25, 0.5, 0.75, 1]
    sigmas = [0, 0.25, 0.5, 0.75, 1]

    keys = ['upd_freq','eps_b', 'eps_t', 'sigma', 'lam']
    params = []
    for vals in itertools.product(update_frequencies, behaviour_eps, target_eps, sigmas, trace_factors):
        d = dict(zip(keys,vals))
        d['gamma'] = 1
        d['epoch'] = 50
        d['max_iters'] = 5000
        d['test_iters'] = 1
        params.append(d)
    return params

def get_params_nondiverged(directory):
    data = utils.parse_results_pkl(directory, LEARNED_REWARD)
    d = data.loc[data['MaxS'] > 1]
    params = [dict(zip(d.index.names,p)) for p in tqdm(d.index)]
    for d in params:
        d["directory"] = os.path.join(directory, "l%f"%d['lam'])
    return params

def get_mean_rewards(directory):
    data = utils.parse_results_pkl(directory, LEARNED_REWARD)
    mr_data = data.apply(lambda row: row.MRS/row.Count, axis=1)
    return mr_data

def get_final_rewards(directory):
    data = utils.parse_results_pkl(directory, LEARNED_REWARD)
    fr_data = data.apply(lambda row: row.MaxS/row.Count, axis=1)
    return fr_data

def get_ucb1_mean_reward(directory):
    data = utils.parse_results_pkl(directory, LEARNED_REWARD)
    count_total = data['Count'].sum()
    def ucb1(row):
        a = row.MRS/row.Count
        b = np.sqrt(2*np.log(count_total)/row.Count)
        return a+b
    score = data.apply(ucb1, axis=1)
    return score

def get_ucb1_final_reward(directory):
    data = utils.parse_results_pkl(directory, LEARNED_REWARD)
    count_total = data['Count'].sum()
    def ucb1(row):
        a = row.MaxS/row.Count
        b = np.sqrt(2*np.log(count_total)/row.Count)
        return a+b
    score = data.apply(ucb1, axis=1)
    return score

def get_params_best(directory, score_function, n=1):
    score = score_function(directory)
    if n == -1:
        n = score.size
    if n == 1:
        params = [score.idxmax()]
    else:
        score = score.sort_values(ascending=False)
        params = itertools.islice(score.index, n)
    return [dict(zip(score.index.names,p)) for p in params]

def run(n=1, proc=10, directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")
    print("Gridsearch")
    print("Environment: ", ENV_NAME)
    print("Directory: %s" % directory)
    print("Determines the best combination of parameters by the number of iterations needed to learn.")

    params = get_params_gridsearch()
    for p in params:
        p['directory'] = directory
    params = itertools.repeat(params, n)
    params = itertools.chain(*list(params))
    params = list(params)
    random.shuffle(params)
    utils.cc(_run_trial, params, proc=proc, keyworded=True)

def run2(n=1, m=10, proc=10, directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")

    params1 = get_params_best(directory, get_ucb1_mean_reward, m)
    params2 = get_params_best(directory, get_ucb1_final_reward, m)
    params = params1+params2

    print("Further refining gridsearch, exploring with UCB1")
    print("Environment: ", ENV_NAME)
    #print("Parameters: %s" % params)
    print("Directory: %s" % directory)

    for p in params:
        p['directory'] = directory
    params = itertools.repeat(params, n)
    params = itertools.chain(*list(params))
    utils.cc(_run_trial, params, proc=proc, keyworded=True)

def run3(n=100, proc=10, params=None, directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")

    params1 = get_params_best(directory, get_mean_rewards, 1)
    params2 = get_params_best(directory, get_final_rewards, 1)
    params = params1+params2

    print("Running more trials with the best parameters found so far.")
    print("Environment: ", ENV_NAME)
    print("Parameters: %s" % params)
    print("Directory: %s" % directory)

    for p in params:
        p['directory'] = directory
    params = itertools.repeat(params, n)
    params = itertools.chain(*list(params))
    utils.cc(_run_trial, params, proc=proc, keyworded=True)

def run_all(proc=10):
    run1(proc=proc)
    run2(proc=proc)
    parse_results2()
    run3(proc=proc)
    parse_results3()

if __name__ == "__main__":
    run_all()
