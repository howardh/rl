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
import random

import frozenlake
from frozenlake import ENV_NAME
from frozenlake import MAX_REWARD
from frozenlake import MIN_REWARD
from frozenlake import LEARNED_REWARD
from frozenlake import features
from frozenlake import utils
from frozenlake import exp2

import graph
import utils

run_trial = exp2.run_trial

get_directory = exp2.get_directory

def get_param_filters():
    return [{'sigma': 0.0}, {'sigma': 1.0}]

def get_params_best(directory, score_function, n=1):
    out0 = frozenlake.experiments.get_params_best(
            directory, score_function, n, {'sigma': 0.0})
    out1 = frozenlake.experiments.get_params_best(
            directory, score_function, n, {'sigma': 1.0})
    return out0+out1

def get_params_gridsearch():
    behaviour_eps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    target_eps = [0, 0.1, 0.2, 0.3, 0.4]
    trace_factors = [0, 0.25, 0.5, 0.75, 1]
    sigmas = [0, 1]
    #learning_rate = np.logspace(np.log10(10),np.log10(.001),num=13,endpoint=True,base=10).tolist()
    learning_rate = np.logspace(np.log10(1),np.log10(.001),num=10,endpoint=True,base=10).tolist()

    keys = ['eps_b', 'eps_t', 'sigma','lam', 'alpha']
    params = []
    for vals in itertools.product(behaviour_eps, target_eps, sigmas,
            trace_factors, learning_rate):
        d = dict(zip(keys,vals))
        d['gamma'] = 1
        d['epoch'] = 10
        d['max_iters'] = 2000
        d['test_iters'] = 50
        params.append(d)
    return params

def get_plot_params_final_rewards():
    raise NotImplementedError("This should not be graphed.")

def get_plot_params_best():
    file_name = 'graph-best4.png'
    label_template = 'SGD sigma={sigma}'
    param_filters = [{'sigma': 0.0}, {'sigma': 1.0}]
    return locals()
