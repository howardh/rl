import torch
import gym
import numpy as np
import pprint
from tqdm import tqdm

import frozenlake2
from frozenlake2 import exp2
from frozenlake2 import exp3
from frozenlake2 import experiments
from frozenlake2 import graph

import utils

def run_all(proc=20):
    ## SGD
    #experiments.run1(exp2, n=1, proc=proc)
    #for _ in tqdm(range(10)):
    #    experiments.run2(exp2, n=10, m=100, proc=proc)
    #experiments.run3(exp2, n=15, proc=proc)
    #exp2.plot_best()
    #exp2.plot_final_rewards()

    ## LSTD
    experiments.run1(exp3, n=1, proc=proc)
    for _ in range(100):
        experiments.run2(exp3, n=10, m=100, proc=proc)
    #experiments.run3(exp3, n=15, proc=proc)
    exp3.plot_best()
    exp3.plot_final_rewards()

    #graph.graph_all()

if __name__ == "__main__":
    utils.set_results_directory("/home/ml/hhuang63/results/test")

    #all_params = [
    #        {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.0, 'lam': 0.0},
    #        {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.0, 'lam': 0.25},
    #        {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.0, 'lam': 0.5},
    #        {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.0, 'lam': 0.75},
    #        {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.0, 'lam': 1.0}
    #]
    #experiments.run3(exp2, n=100, proc=30, params=all_params)
    #exp2.plot_custom()
    run_all(10)
