import gym
import numpy as np
import os

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent
from learner.learner import Optimizer

import atari
from atari import features
from atari import utils
from atari import exp1

import utils

if __name__ == "__main__":
    atari.utils.compute_background()
    iters = atari.exp1._run_trial(0.9,200,0.2,0.01,os.path.join(utils.get_results_directory(),"atari.exp1","part1"), max_iters=1)
    print(iters)
    #import dill
    #data = dill.load(open("params.pkl", "rb"))
    #a_mat = data['a']
    #b_mat = data['b']
    #x = utils.solve_approx(a_mat, b_mat)
