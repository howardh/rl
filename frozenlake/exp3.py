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

discount_factors = ['1', '0.9', '0.5']
update_frequencies = ['50', '200']
behaviour_epsilons = ['1', '0.5', '0.1', '0']
target_epsilons = ['0', '0.01', '0.05']
sigmas = ['0', '0.5', '1']
trace_factors = ['0.01', '0.5', '0.99']

def _run_trial(gamma, upd_freq, eps_b, eps_t, sigma, lam,
        directory=os.path.join(utils.get_results_directory(),"temp",__name__,"part1"),
        stop_when_learned=True, max_iters=10000):
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

    file_name,_ = utils.find_next_free_file(
            "g%.3f-u%d-eb%.3f-et%.3f-s%.3f-l%.3f" % (gamma, upd_freq, eps_b, eps_t, sigma, lam),
            "csv", directory)
    with open(file_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for iters in range(1,max_iters):
            agent.run_episode(e)
            if iters % upd_freq == 0:
                agent.update_weights()
            if iters % 500 == 0:
                rewards = agent.test(e, 100)
                csvwriter.writerow([iters, rewards])
                csvfile.flush()
                if stop_when_learned and np.mean(rewards) >= 0.78:
                    break
    return iters

def _worker(g,u,eb,et,s,l, directory=None):
    try:
        g = float(g)
        u = int(u)
        eb = float(eb)
        et = float(et)
        s = float(s)
        l = float(l)
        return _run_trial(g,u,eb,et,s,l,directory, False,5000)
    except KeyboardInterrupt:
        return None

def _worker2(params, directory=None):
    try:
        g,u,eb,et,s,l = params
        g = float(g)
        u = int(u)
        eb = float(eb)
        et = float(et)
        s = float(s)
        l = float(l)
        return _run_trial(g,u,eb,et,s,l,directory,False,5000)
    except KeyboardInterrupt:
        return None

def run1(n=10, proc=10, directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")
    print("Gridsearch")
    print("Environment: FrozenLake4x4")
    print("Directory: %s" % directory)
    print("Parameter space:")
    print("""
            \tDiscount factor: %s
            \tUpdate Frequencies: %s
            \tBehaviour Epsilons: %s
            \tTarget Epsilons: %s
            \tSigmas: %s
            \tTrace Factors: %s
    """ % (discount_factors, update_frequencies, behaviour_epsilons, target_epsilons, sigmas, trace_factors))
    print("Determines the best combination of parameters by the number of iterations needed to learn.")

    indices = pandas.MultiIndex.from_product(
            [discount_factors, update_frequencies, behaviour_epsilons,
                target_epsilons, sigmas, trace_factors],
            names=["Discount Factor", "Update Frequency", "Behaviour Epsilon",
                "Target Epsilon", "Sigma", "Lambda"])
    data = pandas.DataFrame(index=indices, columns=range(n))

    params = itertools.repeat(list(indices), n)
    params = itertools.chain.from_iterable(params)
    params = zip(params, itertools.repeat(directory))
    utils.cc(_worker2, params, proc=proc, keyworded=False)

def get_best_params1(directory, sigma=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")
    return utils.get_best_params_by_sigma(directory, learned_threshold=0.78)

def run2(n=500, proc=10, params=None, directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part2")
    if params is None:
        params = get_best_params1(os.path.join(utils.get_results_directory(),__name__,"part1"))

    print("Environment: FrozenLake4x4")
    print("Parameters: %s" % params)
    print("Running with best params found")
    print("Directory: %s" % directory)

    params = [p for p in params.values()]
    for p in params:
        p['directory'] = directory
    params = itertools.repeat(params, n)
    params = itertools.chain(*list(params))
    utils.cc(_worker, params, proc=proc, keyworded=True)

def parse_results2(directory=None):
    """
    Parse the CSV files produced by run2, and generates a graph.
    """
    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__,"part2")

    data = utils.parse_graphing_results(directory)

    # Plot
    for sigma in data.keys():
        mean = data[sigma][1]
        std = data[sigma][2]
        x = data[sigma][0]
        label = "sigma-%s"%sigma
        plt.fill_between(x, mean-std/2, mean+std/2, alpha=0.5)
        plt.plot(x, mean, label=label)
    plt.legend(loc='best')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(directory, "graph.png"))
    return data

def run_all(proc=10):
    run1(proc=proc)
    parse_results1()
    run2(proc=proc)
    parse_results2()

if __name__ == "__main__":
    run_all()
