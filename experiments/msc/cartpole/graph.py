import os

#import cartpole
from . import exp1
from . import exp3
from . import exp4
from . import exp5
from . import MAX_REWARD
from . import MIN_REWARD

import experiments.msc as experiments
import graph
from graph import graph_data
import utils

def get_directory():
    return os.path.join(utils.get_results_directory(),__name__)

def graph_sgd(directory=None):
    data1 = experiments.plot_best(exp1)
    data4 = experiments.plot_best(exp4)
    data = data1+data4
    graph_data(data, "graph-sgd.png", get_directory(), ylims=[MIN_REWARD,MAX_REWARD])

    data = data4
    data = [(x,m,None,l) for x,m,s,l in data]
    graph_data(data, "graph-sgd-nostd.png", get_directory(), 
            ylims=[MIN_REWARD,MAX_REWARD],
            xlabel='Episodes',ylabel='Cumulative Rewards')

def graph_lstd(directory=None):
    data3 = experiments.plot_best(exp3)
    data5 = experiments.plot_best(exp5)
    data = data3+data5
    n = 21
    data = [(x[:n],m[:n],s[:n],l) for x,m,s,l in data]
    graph_data(data, "graph-lstd.png", get_directory(), ylims=[MIN_REWARD,MAX_REWARD])

    data = data5
    data = [(x,m,None,l) for x,m,s,l in data]
    graph_data(data, "graph-lstd-nostd.png", get_directory(), 
            ylims=[MIN_REWARD,MAX_REWARD],
            xlabel='Episodes',ylabel='Cumulative Rewards')

def graph_sarsa_tb(directory=None):
    data4 = experiments.plot_best(exp4)
    data5 = experiments.plot_best(exp5)
    data = data4+data5
    graph_data(data, "graph-sarsa-tb.png", get_directory(), ylims=[MIN_REWARD,MAX_REWARD])

def graph_all(directory=None):
    data1 = experiments.plot_best(exp1)
    data3 = experiments.plot_best(exp3)
    #data4 = experiments.plot_best(exp4)
    #data5 = experiments.plot_best(exp5)
    #data = data1+data3+data4+data5
    data = data1+data3
    graph_data(data, "graph-all.png", get_directory(),
            ylims=[MIN_REWARD,MAX_REWARD],
            xlabel='Episodes',ylabel='Cumulative Reward')

    data1 = experiments.plot_best(exp1, 
            [experiments.get_mean_rewards_first100])
    data3 = experiments.plot_best(exp3, 
            [experiments.get_mean_rewards_first100])
    data1 = [(x,m,s,"Q($\sigma$)") for x,m,s,l in data1]
    data3 = [(x,m,s,"LSTDQ($\sigma$)") for x,m,s,l in data3]
    data = data1+data3
    graph_data(data, "graph-all-first100.png", get_directory(),
            ylims=[MIN_REWARD,MAX_REWARD], xlabel='Episodes',
            ylabel='Cumulative Reward')
