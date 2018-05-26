import os

import mountaincar
from mountaincar import exp2
from mountaincar import exp3
from mountaincar import MAX_REWARD
from mountaincar import MIN_REWARD

import experiments
import graph
from graph import graph_data
import utils

def get_directory():
    return os.path.join(utils.get_results_directory(),__name__)

def graph_all(directory=None):
    data2 = experiments.plot_best(exp2)
    data3 = experiments.plot_best(exp3)
    data = data2+data3
    graph_data(data, "graph-all.png", get_directory(),
            ylims=[MIN_REWARD,MAX_REWARD],
            xlabel='Episodes',ylabel='Cumulative Reward')
