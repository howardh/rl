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
from learner.learner import Optimizer

import frozenlake
import utils

discount_factors = ['1', '0.99', '0.9']
learning_rates = ['0.1', '0.01', '0.001']
indices = pandas.MultiIndex.from_product(
        [discount_factors, learning_rates],
        names=["Discount Factor", "Learning Rate"])

def _find_next_free_file(prefix, suffix, directory):
    import os
    if not os.path.isdir(directory):
        os.makedirs(directory)
    i = 0
    while True:
        path=os.path.join(directory,"%s-%d.%s" % (prefix, i, suffix))
        if not os.path.isfile(path):
            break
        i += 1
    return path

def _run_trial(gamma, alpha, directory=None, break_when_learned=False,
        n_episodes=5000):
    env_name = 'FrozenLake-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = TabularAgent(
            action_space=action_space,
            discount_factor=gamma,
            learning_rate=alpha,
            optimizer=Optimizer.NONE)

    file_name = _find_next_free_file("g%f-a%f" % (gamma, alpha), "csv", directory)
    with open(file_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for iters in range(1,n_episodes):
            agent.run_episode(e)
            if iters % 500 == 0:
                rewards = agent.test(e, 100, max_steps=1000)
                csvwriter.writerow([iters, rewards])
                csvfile.flush()
                if break_when_learned and np.mean(rewards) >= 0.78:
                    break
    #print("Policy:")
    #frozenlake.utils.print_policy(agent)
    return iters

def _run_trial1(gamma, alpha, directory=None):
    return _run_trial(gamma, alpha, directory, True, 5000)

def _run_trial2(gamma, alpha, directory=None):
    return _run_trial(gamma, alpha, directory, False, 5000)

def _worker(i, directory=None):
    try:
        g,a = indices[i]
        g = float(g)
        a = float(a)
        return _run_trial1(g,a,directory)
    except KeyboardInterrupt:
        return None

def _worker2(i, directory=None):
    try:
        g,a = i
        g = float(g)
        a = float(a)
        return _run_trial1(g,a,directory)
    except KeyboardInterrupt:
        return None

def run(n=10, proc=10, directory=None):
    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__,"part1")
    print("Gridsearch")
    print("Environment: FrozenLake4x4")
    print("Parameter space:")
    print("""
            \tDiscount factor: %s
            \tLearning rate: %s
    """ % (discount_factors, learning_rates))
    print("Determines the best combination of parameters by the number of iterations needed to learn.")
    data = pandas.DataFrame(index=indices, columns=range(n))

    params = itertools.repeat(list(indices), n)
    params = itertools.chain.from_iterable(params)
    params = zip(params, itertools.repeat(directory))
    utils.cc(_worker2, params, proc=proc, keyworded=False)

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

    params = [params['g'], params['a'], directory]
    utils.cc(_run_trial2, itertools.repeat(params,n), proc=proc, keyworded=False)

def parse_results(directory=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__,"part2")

    files = [f for f in os.listdir(directory) if
            os.path.isfile(os.path.join(directory,f))]
    data = []
    for file_name in tqdm(files, desc="Parsing File Contents"):
        try:
            full_path = os.path.join(directory,file_name)
            with open(full_path, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                results = [np.sum(eval(r[1])) for r in reader]
                if len(results) == 0:
                    os.remove(full_path)
                    continue
                data.append(results)
        except SyntaxError as e:
            print("Broken file: %s" % file_name)
        except Exception as e:
            print("Broken file: %s" % file_name)

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    plt.fill_between(range(0,len(mean)*500,500), mean-std/2, mean+std/2, alpha=0.5)
    plt.plot(range(0,len(mean)*500,500), mean, label="Tabular")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    output = os.path.join(directory, "graph.png")
    plt.savefig(output)
    print("Graph saved at %s" % output)

    return (mean,std)

if __name__ == "__main__":
    run()
