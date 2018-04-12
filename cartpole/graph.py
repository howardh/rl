import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cartpole.exp1
import cartpole.exp3
import utils

def graph_all(directory=None):
    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__)
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    print("Graphing data...")

    data1 = cartpole.exp1.plot_best()
    data3 = cartpole.exp3.plot_best()
    data = data1+data3

    fig = plt.figure()

    # Exp 1
    for x,mean,std,label in data:
        plt.fill_between(x, mean-std/2, mean+std/2, alpha=0.3)
        plt.plot(x, mean, label=label)

    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    output = os.path.join(directory, "graph.png")
    plt.savefig(output)
    print("Graph saved at %s" % output)
