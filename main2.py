import torch
import gym
import numpy as np
import pprint
from tqdm import tqdm

import frozenlake2
from frozenlake2 import exp2
from frozenlake2 import exp3
from frozenlake2 import graph

import experiments
import utils

def run_all(proc=20):
    ## SGD
    #experiments.run1(exp2, n=1, proc=proc)
    #for _ in tqdm(range(100)):
    #    experiments.run2(exp2, n=10, m=10, proc=proc)
    #experiments.run3(exp2, n=15, proc=proc)
    #experiments.plot_best(exp2)
    #experiments.plot_final_rewards(exp2)

    ### LSTD
    #experiments.run1(exp3, n=2, proc=proc)
    for _ in tqdm(range(100)):
        experiments.run2(exp3, n=10, m=100, proc=proc)
    #experiments.run3(exp3, n=15, proc=proc)
    experiments.plot_best(exp3)
    experiments.plot_final_rewards(exp3)

    graph.graph_all()

if __name__ == "__main__":
    utils.set_results_directory("/home/ml/hhuang63/results/final")
    params = [{'lam': 0.25, 'gamma': 1, 'epoch': 10, 'max_iters': 2000, 'alpha': 0.4641588833612779, 'eps_t': 0.1, 'eps_b': 0.0, 'test_iters': 50, 'sigma': 0.75}]
    #experiments.run3(exp2, n=30, proc=30, params=params)
    run_all(30)
