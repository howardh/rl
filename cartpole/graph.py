import os

import cartpole
from cartpole import exp1
from cartpole import exp3
from cartpole import exp4
from cartpole import exp5
from cartpole import MAX_REWARD
from cartpole import MIN_REWARD

import experiments
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

def graph_lstd(directory=None):
    data3 = experiments.plot_best(exp3)
    data5 = experiments.plot_best(exp5)
    data = data3+data5
    n = 21
    data = [(x[:n],m[:n],s[:n],l) for x,m,s,l in data]
    graph_data(data, "graph-lstd.png", get_directory(), ylims=[MIN_REWARD,MAX_REWARD])

def graph_sarsa_tb(directory=None):
    data4 = experiments.plot_best(exp4)
    data5 = experiments.plot_best(exp5)
    data = data4+data5
    graph_data(data, "graph-sarsa-tb.png", get_directory(), ylims=[MIN_REWARD,MAX_REWARD])

def graph_all(directory=None):
    data1 = experiments.plot_best(exp1)
    data3 = experiments.plot_best(exp3)
    data4 = experiments.plot_best(exp4)
    data5 = experiments.plot_best(exp5)
    data = data1+data3+data4+data5
    graph_data(data, "graph-all.png", get_directory(),
            ylims=[MIN_REWARD,MAX_REWARD],
            xlabel='Episodes',ylabel='Cumulative Reward')
