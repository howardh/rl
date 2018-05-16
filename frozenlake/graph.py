import os

import frozenlake
from frozenlake import exp2
from frozenlake import exp3
from frozenlake import exp4
from frozenlake import exp5
from frozenlake import MAX_REWARD
from frozenlake import MIN_REWARD

import experiments
import graph
from graph import graph_data
import utils

def get_directory():
    return os.path.join(utils.get_results_directory(),__name__)

def graph_sgd(directory=None):
    data = experiments.plot_best(exp2)
    graph_data(data, "graph-sgd.png", get_directory(), ylims=[MIN_REWARD,MAX_REWARD])

def graph_lstd(directory=None):
    data3 = experiments.plot_best(exp3)
    data5 = experiments.plot_best(exp5)
    data = data3+data5
    #n = 51
    #data = [(x[:n],m[:n],s[:n],l) for x,m,s,l in data]
    graph_data(data, "graph-lstd.png", get_directory(), ylims=[MIN_REWARD,MAX_REWARD])

def graph_sarsa_tb(directory=None):
    data4 = experiments.plot_best(exp4)
    data5 = experiments.plot_best(exp5)
    data = data4+data5
    graph_data(data, "graph-sarsa-tb.png", get_directory(), ylims=[MIN_REWARD,MAX_REWARD])

def graph_all(directory=None):
    data2 = experiments.plot_best(exp2)
    data3 = experiments.plot_best(exp3)
    data = data2+data3
    graph_data(data, "graph-all.png", get_directory(), ylims=[MIN_REWARD,MAX_REWARD])
