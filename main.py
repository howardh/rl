import os
import torch
import gym
import numpy as np

from experiments.hrl.main import run_trial_steps, run_gridsearch

#args,rewards = run_trial_steps(gamma=1,alpha=0.01,eps_b=0.1,eps_t=0,tau=0.01,test_iters=5,max_steps=1000000,min_replay_buffer_siz=10000,epoch=100,verbose=True)
results = run_gridsearch()
