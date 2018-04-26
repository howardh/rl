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
import fabulous
from fabulous.color import bold

from . import ENV_NAME
from . import MAX_REWARD
from . import MIN_REWARD
from . import LEARNED_REWARD

import utils


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


def run1(exp, n=1, proc=10, directory=None):
    if directory is None:
        directory=exp.get_directory()
    print("-"*80)
    print("Gridsearch: Run n trials on each combination of parameters.")
    to_print = [
            ("Environment",exp.ENV_NAME),
            ("Directory", directory)]
    for heading,val in to_print:
        print("%s: %s" % (bold(heading),val))

    params = exp.get_params_gridsearch()
    for p in params:
        p['directory'] = directory
    params = itertools.repeat(params, n)
    params = itertools.chain(*list(params))
    params = list(params)
    random.shuffle(params)
    utils.cc(exp.run_trial, params, proc=proc, keyworded=True)

def run2(exp, n=1, m=10, proc=10, directory=None):
    if directory is None:
        directory=exp.get_directory()

    params1 = get_params_best(directory, get_ucb1_mean_reward, m)
    params2 = get_params_best(directory, get_ucb1_final_reward, m)
    params = params1+params2

    print("-"*80)
    print("Further refining gridsearch, exploring with UCB1")
    to_print = [
            ("Environment",exp.ENV_NAME),
            #("Parameters", params),
            ("Directory", directory)]
    for heading,val in to_print:
        print("%s: %s" % (bold(heading),val))

    for p in params:
        p['directory'] = directory
    params = itertools.repeat(params, n)
    params = itertools.chain(*list(params))
    utils.cc(exp.run_trial, params, proc=proc, keyworded=True)

def run3(exp, n=100, proc=10, params=None, directory=None):
    if directory is None:
        directory=exp.get_directory()
    if params is None:
        if by_mean_reward:
            params1 = get_params_best(directory, get_mean_rewards, 1)
        else:
            params1 = []
        if by_final_reward:
            params2 = get_params_best(directory, get_final_rewards, 1)
        else:
            params2 = []
        params = params1+params2

    print("-"*80)
    print("Running more trials with the best parameters found so far.")
    to_print = [
            ("Environment",exp.ENV_NAME),
            ("Parameters", params),
            ("Directory", directory)]
    for heading,val in to_print:
        print("%s: %s" % (bold(heading),val))

    for p in params:
        p['directory'] = directory
    params = itertools.repeat(params, n)
    params = itertools.chain(*list(params))
    utils.cc(exp.run_trial, params, proc=proc, keyworded=True)
