import numpy as np
import gym
import itertools
import pandas
import dill
import csv
import os
from tqdm import tqdm
import time
import operator
import pprint
import sys
import traceback
import random

import frozenlake
from frozenlake import ENV_NAME
from frozenlake import MAX_REWARD
from frozenlake import MIN_REWARD
from frozenlake import LEARNED_REWARD

from frozenlake import features
from frozenlake import utils
from frozenlake import exp3

import graph
import utils

run_trial = exp3.run_trial

get_directory = exp3.get_directory

def get_param_filters():
    return [{'sigma': 0.5},
            {'sigma': 0.75}]
    return [{'sigma': 0.0},
            {'sigma': 0.25},
            {'sigma': 0.5},
            {'sigma': 0.75},
            {'sigma': 1.0}]

def get_params_gridsearch():
    update_frequencies = [1,50,200]
    behaviour_eps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    target_eps = [0, 0.1, 0.2, 0.3, 0.4]
    trace_factors = [0, 0.25, 0.5, 0.75, 1]
    sigmas = [0, 1]

    keys = ['upd_freq','eps_b', 'eps_t', 'sigma', 'lam']
    params = []
    for vals in itertools.product(update_frequencies, behaviour_eps, target_eps, sigmas, trace_factors):
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
    file_name = 'graph-best5.png'
    label_template = 'LSTD sigma={sigma}'
    return locals()
