import os

import frozenlake2
from frozenlake2 import exp2
from frozenlake2 import exp3
from frozenlake2 import exp4
from frozenlake2 import exp5
from frozenlake2 import MAX_REWARD
from frozenlake2 import MIN_REWARD

import experiments
import graph
from graph import graph_data
import utils

def get_directory():
    return os.path.join(utils.get_results_directory(),__name__)

def graph_sgd(directory=None):
    data2 = experiments.plot_best(exp2)
    data4 = experiments.plot_best(exp4)
    data = data4+data2
    graph_data(data, "graph-sgd.png", get_directory(), ylims=[MIN_REWARD,MAX_REWARD])

def graph_lstd(directory=None):
    data3 = experiments.plot_best(exp3)
    data5 = experiments.plot_best(exp5)
    data = data5+data3
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
    #data4 = experiments.plot_best(exp4)
    #data5 = experiments.plot_best(exp5)
    #data = data4+data5+data2+data3
    #data = [(x,m,None,l) for x,m,s,l in data]
    graph_data(data, "graph-all.png", get_directory(), ylims=[MIN_REWARD,MAX_REWARD])
