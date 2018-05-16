import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils

def get_data(params, directory, label=''):
    print("Parsing plotting data for params: ", params)
    series = utils.get_series_with_params(directory, params)
    print("# series: ", len(series))
    mean = np.mean(series, axis=0)
    std = np.std(series, axis=0)
    epoch = params['epoch']
    x = [i*epoch for i in range(len(mean))]

    return (x, mean, std, label)

def get_data_individual(params, directory, label=''):
    print("Parsing plotting data for params: ", params)
    series = utils.get_series_with_params(directory, params)
    print("# series: ", len(series))
    epoch = params['epoch']
    x = [i*epoch for i in range(len(series[0]))]

    return (x, series, label)


def graph_data(data, file_name, directory,
        ylims=None, xlabel='', ylabel=''):
    if directory is None:
        raise ValueError("Invalid directory: ", directory)
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    print("Graphing data...")

    fig, ax = plt.subplots(1,1)
    if ylims is not None:
        ax.set_ylim(ylims)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    for x,mean,std,label in data:
        if std is not None:
            plt.fill_between(x, mean-std/2, mean+std/2, alpha=0.3)
        plt.plot(x, mean, label=label)

    ax.legend(loc='best')
    output = os.path.join(directory, file_name)
    plt.savefig(output)
    print("Graph saved at %s" % output)
    plt.close(fig)

def graph_data_histogram(data, file_name, directory,
        xlabel='', ylabel=''):
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.hist(data, bins=20)
    output = os.path.join(directory, file_name)
    plt.savefig(output)
    print("Graph saved at %s" % output)
    plt.close(fig)
