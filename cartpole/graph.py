import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cartpole.exp1
import cartpole.exp3
import cartpole.exp4
import cartpole.exp5
import utils

def graph_sgd(directory=None):
    data1 = cartpole.exp1.plot_best()
    data4 = cartpole.exp4.plot_best()
    data = data1+data4
    graph_data(data, "graph-sgd.png")

def graph_lstd(directory=None):
    data3 = cartpole.exp3.plot_best()
    data5 = cartpole.exp5.plot_best()
    data = data3+data5
    n = 21
    data = [(x[:n],m[:n],s[:n],l) for x,m,s,l in data]
    graph_data(data, "graph-lstd.png")

def graph_all(directory=None):
    data1 = cartpole.exp1.plot_best()
    data3 = cartpole.exp3.plot_best()
    data4 = cartpole.exp4.plot_best()
    data5 = cartpole.exp5.plot_best()
    data = data1+data3+data4+data5
    graph_data(data, "graph-all.png")

def graph_data(data, file_name, directory=None):
    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__)
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    print("Graphing data...")

    fig = plt.figure()

    # Exp 1
    for x,mean,std,label in data:
        plt.fill_between(x, mean-std/2, mean+std/2, alpha=0.3)
        plt.plot(x, mean, label=label)

    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    output = os.path.join(directory, file_name)
    plt.savefig(output)
    print("Graph saved at %s" % output)
