import os

import cartpole
from cartpole import MAX_REWARD
from cartpole import MIN_REWARD

import graph
from graph import graph_data
import utils

def get_directory():
    return os.path.join(utils.get_results_directory(),__name__)

def graph_sgd(directory=None):
    data1 = cartpole.exp1.plot_best()
    data4 = cartpole.exp4.plot_best()
    data = data1+data4
    graph_data(data, "graph-sgd.png", get_directory(), ylims=[MIN_REWARD,MAX_REWARD])

def graph_lstd(directory=None):
    data3 = cartpole.exp3.plot_best()
    data5 = cartpole.exp5.plot_best()
    data = data3+data5
    n = 21
    data = [(x[:n],m[:n],s[:n],l) for x,m,s,l in data]
    graph_data(data, "graph-lstd.png", get_directory(), ylims=[MIN_REWARD,MAX_REWARD])

def graph_sarsa_tb(directory=None):
    data4 = cartpole.exp4.plot_best()
    data5 = cartpole.exp5.plot_best()
    data = data4+data5
    graph_data(data, "graph-sarsa-tb.png", get_directory(), ylims=[MIN_REWARD,MAX_REWARD])

def graph_all(directory=None):
    data1 = cartpole.exp1.plot_best()
    data3 = cartpole.exp3.plot_best()
    data4 = cartpole.exp4.plot_best()
    data5 = cartpole.exp5.plot_best()
    data = data1+data3+data4+data5
    graph_data(data, "graph-all.png", get_directory(), ylims=[MIN_REWARD,MAX_REWARD])
