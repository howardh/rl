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
from agent.lstd_agent import LSTDAgent

import frozenlake
import utils

discount_factors = ['1', '0.99', '0.9']
learning_rates = ['1', '0.1', '0.01', '0.001']
trace_factors = ['0', '0.8', '0.5', '0.2', '0']
sigmas = ['0', '0.5', '1']

def _run_trial(gamma, alpha, lam, sigma, directory=None, break_when_learned=False,
        n_episodes=5000):
    env_name = 'FrozenLake-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = TabularAgent(
            action_space=action_space,
            discount_factor=gamma,
            learning_rate=alpha,
            trace_factor=lam,
            sigma=sigma)

    file_name = utils.find_next_free_file("g%f-a%f-l%f-s%f" % (gamma, alpha, lam, sigma), "csv", directory)
    with open(file_name[0], 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow([0, 0])
        for iters in range(1,n_episodes):
            agent.run_episode(e)
            if iters % 500 == 0:
                rewards = agent.test(e, 100, max_steps=1000)
                csvwriter.writerow([iters, rewards])
                csvfile.flush()
                if break_when_learned and np.mean(rewards) >= 0.78:
                    break
    return iters

def _worker(i, directory=None):
    try:
        g,a,l,s = i
        g = float(g)
        a = float(a)
        l = float(l)
        s = float(s)
        return _run_trial(g,a,l,s,directory)
    except KeyboardInterrupt:
        return None

def _worker2(g,a,l,s,directory=None):
    try:
        return _run_trial(g,a,l,s,directory)
    except KeyboardInterrupt:
        return None

def run1(n=10, proc=10, directory=None):
    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__,"part1")

    print("Gridsearch")
    print("Environment: FrozenLake4x4")
    print("Parameter space:")
    print("\tDiscount factor: %s"%discount_factors)
    print("\tLearning rate: %s"%learning_rates)
    print("\tTrace Factor: %s"%[0])
    print("\tSigmas: %s"%[0])
    print("Determines the best combination of parameters by the number of iterations needed to learn.")
    print("Directory: %s" % directory)

    indices = pandas.MultiIndex.from_product(
            [discount_factors, learning_rates, [0], [0]],
            names=["Discount Factor", "Learning Rate", "Trace Factor", "Sigma"])
    data = pandas.DataFrame(index=indices, columns=range(n))

    params = itertools.repeat(list(indices), n)
    params = itertools.chain.from_iterable(params)
    params = zip(params, itertools.repeat(directory))
    utils.cc(_worker, params, proc=proc, keyworded=False)

def get_best_params(directory):
    data = utils.parse_results(directory, learned_threshold=0.78)
    mr, ttl = utils.sort_data(data)
    params = [eval(x) for x in ttl[0][0]]
    return utils.combine_params_with_names(data,params)

def run2(n=10, proc=10, params=None, directory=None):
    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__,"part2")
    if params is None:
        params = get_best_params(os.path.join(utils.get_results_directory(),__name__,"part1"))

    print(params)
        
    print("Environment: FrozenLake4x4")
    print("Parameters:")
    print("""
            \tDiscount factor: %s
            \tLearning rate: %s
    """ % (params['g'], params['a']))
    print("Runs stuff with the best parameters found during gridsearch")

    params = [params['g'], params['a'], 0, 0, directory]
    utils.cc(_run_trial, itertools.repeat(params,n), proc=proc, keyworded=False)

def parse_results2(directory=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__,"part2")

    data = utils.parse_graphing_results(directory)
    data = data[list(data.keys())[0]]

    mean = data[1]
    std = data[2]
    x = data[0]
    plt.fill_between(x, mean-std/2, mean+std/2, alpha=0.5)
    plt.plot(x, mean, label="Tabular")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    output = os.path.join(directory, "graph.png")
    plt.savefig(output)
    print("Graph saved at %s" % output)

    return (mean,std)

def run3(n=10, proc=10, directory=None):
    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__,"part3")

    print("Gridsearch")
    print("Environment: FrozenLake4x4")
    print("Parameter space:")
    print("\tDiscount factor: %s"%discount_factors)
    print("\tLearning rate: %s"%learning_rates)
    print("\tTrace Factor: %s"%trace_factors)
    print("\tSigmas: %s"%sigmas)
    print("Determines the best combination of parameters by the number of iterations needed to learn.")
    print("Directory: %s" % directory)

    indices = pandas.MultiIndex.from_product(
            [discount_factors, learning_rates, trace_factors, sigmas],
            names=["Discount Factor", "Learning Rate", "Trace Factor", "Sigma"])
    data = pandas.DataFrame(index=indices, columns=range(n))

    params = itertools.repeat(list(indices), n)
    params = itertools.chain.from_iterable(params)
    params = zip(params, itertools.repeat(directory))
    utils.cc(_worker, params, proc=proc, keyworded=False)

def get_best_params3(directory=None):
    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__,"part3")
    data = utils.parse_results(directory, learned_threshold=0.78)
    sigmas = set(data.index.get_level_values('s'))
    results = dict()
    for s in sigmas:
        df = data.xs(s,level='s')
        mr, ttl = utils.sort_data(df)
        params = [eval(x) for x in ttl[0][0]]
        params = utils.combine_params_with_names(df,params)
        params['s'] = eval(s)
        results[s] = params
    return results

def run4(n=100, proc=10, directory=None):
    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__,"part4")

    best_params = get_best_params3()

    print("Environment: FrozenLake4x4")
    print("Parameters: %s" % best_params)
    print("Running with best params found")
    print("Directory: %s" % directory)

    params = [p for p in best_params.values()]
    for p in params:
        p['directory'] = directory
    params = itertools.repeat(params, n)
    params = itertools.chain(*list(params))
    utils.cc(_worker2, params, proc=proc, keyworded=True)

def parse_results4(directory=None):
    """
    Parse the CSV files produced by run4, and generates a graph.
    """
    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__,"part4")

    data = utils.parse_graphing_results(directory)
    #return data

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
    output = os.path.join(directory, "graph.png")
    plt.savefig(output)
    print("Graph saved at %s" % output)

    return data

def run_all(proc=10):
    run1(n=10,proc=proc)
    run2(n=500,proc=proc)
    parse_results2()
    run3(n=10,proc=proc)
    run4(n=1000,proc=proc)
    parse_results4()

if __name__ == "__main__":
    run()
