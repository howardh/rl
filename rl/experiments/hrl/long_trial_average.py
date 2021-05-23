import numpy as np
import gym
import torch
from tqdm import tqdm
import dill
import os
import itertools

from rl.agent.dqn_agent import DQNAgent
from rl.agent.policy import get_greedy_epsilon_policy

from rl.environment.wrappers import DiscreteObservationToBox

from . import gridsearch
from .model import QFunction

from rl import utils

def plot(results_directory, plot_directory):
    results = utils.get_all_results(results_directory)

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set_size_inches(10,4)
    mean_rewards = []
    mean_sa_vals = []
    for k,v in results:
        params = dict(k)
        mean_rewards.append([np.mean(epoch) for epoch in v['rewards']])
        mean_sa_vals.append([np.mean(epoch) for epoch in v['state_action_values']])

    mean_rewards = np.mean(mean_rewards, axis=0)
    mean_sa_vals = np.mean(mean_sa_vals, axis=0)
    x = list(range(0,len(mean_rewards)*params['epoch'],params['epoch']))
    ax1.set_title('Testing Reward')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average Reward')
    ax1.plot(x,mean_rewards)
    ax2.set_title('Predicted Action Values')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Expected Return')
    ax2.plot(x,mean_sa_vals)

    file_name = os.path.join(plot_directory,'plot.png')
    if not os.path.isdir(plot_directory):
        os.makedirs(plot_directory)
    fig.savefig(file_name)
    plt.close(fig)
    print('Saved file', file_name)

def run(proc=3,n=10):
    utils.set_results_directory(
            os.path.join(utils.get_results_root_directory(),'hrl'))
    directory = os.path.join(utils.get_results_directory(),__name__)
    plot_directory = os.path.join(utils.get_results_directory(),'plots',__name__)

    funcs = [lambda: gridsearch.run_trial(gamma=1,alpha=0.001,eps_b=0.1,eps_t=0,tau=0.01,net_structure=(10,10),batch_size=256,epoch=1000,test_iters=10,verbose=False,directory=directory,max_steps=1000000) for _ in range(n)]
    utils.cc(funcs,proc=proc)
    plot(results_directory=directory,plot_directory=plot_directory)
