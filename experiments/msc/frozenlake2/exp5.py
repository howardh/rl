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

from . import ENV_NAME
from . import MAX_REWARD
from . import MIN_REWARD
from . import LEARNED_REWARD

from . import features
from . import utils
from . import exp3

import graph
import utils

run_trial = exp3.run_trial

get_directory = exp3.get_directory

def get_param_filters():
    #return [{'sigma': 0.5}, {'sigma': 0.75}]
    #return [{'sigma': 0.25, 'lam':0.0}]
    return [{'sigma': 0.0}, {'sigma': 0.25}, {'sigma': 0.5}, {'sigma': 0.75}, {'sigma': 1.0}]
    return [{'sigma': 0.0}, {'sigma': 1.0}]

def get_params_gridsearch():
    update_frequencies = [1,50,200]
    behaviour_eps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    target_eps = [0, 0.1, 0.2, 0.3, 0.4]
    trace_factors = [0, 0.25, 0.5, 0.75, 1]
    sigmas = [0, 1]

    keys = ['upd_freq','eps_b', 'eps_t', 'sigma','lam']
    params = []
    for vals in itertools.product(update_frequencies, behaviour_eps, target_eps, sigmas, trace_factors):
        d = dict(zip(keys,vals))
        d['gamma'] = 0.9
        d['epoch'] = 50
        d['max_iters'] = 5000
        d['test_iters'] = 1
        params.append(d)
    return params

def get_plot_params_final_rewards():
    raise NotImplementedError("This should not be graphed.")

def get_plot_params_best():
    file_name = 'graph-best5.png'
    label_template = 'LSTD Q($\sigma={sigma}$)'
    return locals()
