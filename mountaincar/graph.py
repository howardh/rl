import os

import mountaincar
from mountaincar import MAX_REWARD
from mountaincar import MIN_REWARD

import graph
from graph import graph_data
import utils

def get_directory():
    return os.path.join(utils.get_results_directory(),__name__)

def graph_all(directory=None):
    data2 = mountaincar.exp2.plot_best()
    data3 = mountaincar.exp3.plot_best()
    data = data2+data3
    graph_data(data, "graph-all.png", get_directory(), ylims=[MIN_REWARD,MAX_REWARD])
