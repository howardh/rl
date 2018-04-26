import os

import frozenlake
from frozenlake import MAX_REWARD
from frozenlake import MIN_REWARD

import graph
from graph import graph_data
import utils

def get_directory():
    return os.path.join(utils.get_results_directory(),__name__)

def graph_sgd(directory=None):
    data = frozenlake.exp2.plot_best()
    graph_data(data, "graph-sgd.png", get_directory(), ylims=[MIN_REWARD,MAX_REWARD])

def graph_lstd(directory=None):
    data = frozenlake.exp3.plot_best()
    n = 51
    data = [(x[:n],m[:n],s[:n],l) for x,m,s,l in data]
    graph_data(data, "graph-lstd.png", get_directory(), ylims=[MIN_REWARD,MAX_REWARD])

def graph_all(directory=None):
    data2 = frozenlake.exp2.plot_best()
    data3 = frozenlake.exp3.plot_best()
    data = data2+data3
    graph_data(data, "graph-all.png", get_directory(), ylims=[MIN_REWARD,MAX_REWARD])
