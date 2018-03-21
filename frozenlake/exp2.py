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

import utils

#learning_rates = ['1', '0.1', '0.01']
learning_rates = np.logspace(np.log10(10),np.log10(.01),num=10,endpoint=True,base=10)
learning_rates = ["%.3f"%x for x in learning_rates]
discount_factors = ['1']
behaviour_epsilons = ['1', '0.5', '0']
target_epsilons = ['0']
sigmas = ['0', '0.5', '1']
trace_factors = ['0.01', '0.25', '0.5', '0.75', '0.99']

def _run_trial(alpha, gamma, eps_b, eps_t, sigma, lam,
        directory=os.path.join(utils.get_results_directory(),"temp",__name__,"part1"),
        stop_when_learned=True, max_iters=10000):
    """
    Run the learning algorithm on FrozenLake and return the number of
    iterations needed to learn the task.
    """
    env_name = 'FrozenLake-v0'
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

    file_name,_ = utils.find_next_free_file(
            "a%.3f-g%.3f-eb%.3f-et%.3f-s%.3f-l%.3f" % (alpha, gamma, eps_b, eps_t, sigma, lam),
            "csv", directory)
    with open(file_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for iters in range(0,max_iters+1):
            if iters % 50 == 0:
                rewards = agent.test(e, 100)
                csvwriter.writerow([iters, rewards])
                csvfile.flush()
                if stop_when_learned and np.mean(rewards) >= 0.78:
                    tqdm.write("Done learning.")
                    break
            agent.run_episode(e)
    return iters

def _worker(a,g,eb,et,s,l, directory=None,i=5000):
    try:
        a = float(a)
        g = float(g)
        eb = float(eb)
        et = float(et)
        s = float(s)
        l = float(l)
        return _run_trial(a,g,eb,et,s,l,directory, False,i)
    except KeyboardInterrupt:
        return None

def _worker2(params, directory=None):
    try:
        a,g,eb,et,s,l = params
        a = float(a)
        g = float(g)
        eb = float(eb)
        et = float(et)
        s = float(s)
        l = float(l)
        return _run_trial(a,g,eb,et,s,l,directory,False,5000)
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
            \tLearning Rates: %s
            \tDiscount factor: %s
            \tBehaviour Epsilons: %s
            \tTarget Epsilons: %s
            \tSigmas: %s
            \tTrace Factors: %s
    """ % (learning_rates, discount_factors, behaviour_epsilons, target_epsilons, sigmas, trace_factors))
    print("Determines the best combination of parameters by the number of iterations needed to learn.")

    indices = pandas.MultiIndex.from_product(
            [learning_rates, discount_factors, behaviour_epsilons,
                target_epsilons, sigmas, trace_factors],
            names=["Learning Rates", "Discount Factor", "Behaviour Epsilon",
                "Target Epsilon", "Sigma", "Lambda"])
    data = pandas.DataFrame(index=indices, columns=range(n))

    params = itertools.repeat(list(indices), n)
    params = itertools.chain.from_iterable(params)
    params = zip(params, itertools.repeat(directory))
    utils.cc(_worker2, params, proc=proc, keyworded=False)

def get_best_params1(directory=None, sigma=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")
    return utils.get_best_params_by_sigma(directory, learned_threshold=0.78)

def parse_results1():
    pass

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

def run3(n=100, proc=10, params=None, directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part3")

    learning_rates = ['1', '0.1', '0.01']
    discount_factors = ['1']
    behaviour_epsilons = ['0.50']
    target_epsilons = ['0']
    sigmas = ['0', '0.25', '0.5', '0.75', '1']
    trace_factors = ['0', '0.25', '0.5', '0.75', '1']

    keys = ["a","g","eb","et","s","l"]
    params = []
    for vals in itertools.product(learning_rates, discount_factors,
            behaviour_epsilons, target_epsilons, sigmas, trace_factors):
        d = dict(zip(keys,vals))
        d["directory"] = os.path.join(directory,
                "l%.3f-a%.3f"%(float(d['l']),float(d['a'])))
        d["i"] = 2000
        params.append(d)
    params = itertools.repeat(params, n)
    params = itertools.chain(*list(params))
    utils.cc(_worker, params, proc=proc, keyworded=True)

def parse_results3(directory=None):
    """
    Parse the CSV files produced by run2, and generates a graph.
    """
    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__,"part3")

    subdirs = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory,f))]
    for d in subdirs:
        data = utils.parse_graphing_results(os.path.join(directory,d))

        # Plot
        for sigma in data.keys():
            mean = data[sigma][1]
            std = data[sigma][2]
            x = data[sigma][0]
            label = "sigma-%s"%sigma
            plt.fill_between(x, mean-std/2, mean+std/2, alpha=0.5)
            plt.plot(x, mean, label=label)
        plt.legend(loc='best')
        plt.title("Lambda: %s" % d[1:])
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.savefig(os.path.join(directory, d, "graph.eps"), format='eps')
        plt.close()

        for sigma in data.keys():
            mean = data[sigma][1]
            x = data[sigma][0]
            label = "sigma-%s"%sigma
            plt.plot(x, mean, label=label)
        plt.legend(loc='best')
        plt.title("Lambda: %s" % d[1:])
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.savefig(os.path.join(directory, d, "graph2.eps"), format='eps')
        plt.close()

def run_all(proc=10):
    run1(proc=proc)
    run2(proc=proc)
    parse_results2()
    run3(proc=proc)
    parse_results3()

if __name__ == "__main__":
    run_all()
