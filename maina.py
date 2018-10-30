import gym
import numpy as np
import os

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent

import atari
from atari import features
from atari import utils
from atari import exp1
from atari import exp3

import utils

if __name__ == "__main__":
    #atari.utils.compute_background()
    iters = atari.exp1.run_trial(1e-6,1,0.2,0.01,1,0,os.path.join(utils.get_results_directory(),"atari.exp1","part1"), 5000, 50, 10)
    #iters = atari.exp3.run_trial(1,0.2,0.01,0,0,1,50,5000,10,os.path.join(utils.get_results_directory(),"atari.exp3","part1"))
    print(iters)
    #import dill
    #data = dill.load(open("params.pkl", "rb"))
    #a_mat = data['a']
    #b_mat = data['b']
    #x = utils.solve_approx(a_mat, b_mat)
