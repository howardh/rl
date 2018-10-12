import os
import numpy as np
import itertools

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
    ax.grid()

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
    if type(data[0]) is not list:
        fig, ax = plt.subplots(1,1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        plt.hist(data, bins=20)
        output = os.path.join(directory, file_name)
        plt.savefig(output)
        print("Graph saved at %s" % output)
        plt.close(fig)
    else:
        ax0len = len(data)
        ax1len = len(data[0])
        fig, ax = plt.subplots(ax0len,ax1len,True,True)
        for i in range(ax0len):
            for j in range(ax1len):
                if data[i][j] is None:
                    continue
                if type(data[i][j]) is str:
                    ax[i][j].text(0,10,data[i][j])
                    ax[i][j].tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
                    ax[i][j].grid(False)
                    ax[i][j].axis('off')
                    continue
                ax[i][j].hist(data[i][j], bins=20)
        # Common axis labels
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        plt.grid(False)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # Save file
        output = os.path.join(directory, file_name)
        plt.savefig(output)
        print("Graph saved at %s" % output)
        plt.close(fig)

def graph_data_curves(data, file_name, directory,
        xlabel='', ylabel=''):
    raise NotImplementedError("TODO: This is just a copy and paste of the other thing for now.")
    if type(data[0]) is not list:
        fig, ax = plt.subplots(1,1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        plt.hist(data, bins=20)
        output = os.path.join(directory, file_name)
        plt.savefig(output)
        print("Graph saved at %s" % output)
        plt.close(fig)
    else:
        ax0len = len(data)
        ax1len = len(data[0])
        fig, ax = plt.subplots(ax0len,ax1len,True,True)
        for i in range(ax0len):
            for j in range(ax1len):
                if data[i][j] is None:
                    continue
                if type(data[i][j]) is str:
                    ax[i][j].text(0,10,data[i][j])
                    ax[i][j].tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
                    ax[i][j].grid(False)
                    ax[i][j].axis('off')
                    continue
                ax[i][j].hist(data[i][j], bins=20)
        # Common axis labels
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        plt.grid(False)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # Save file
        output = os.path.join(directory, file_name)
        plt.savefig(output)
        print("Graph saved at %s" % output)
        plt.close(fig)

def graph_matrix(file_name, directory,
        axis_vals, values,
        axis_labels=['majx','majy','minx','miny'],
        min_val=None,max_val=None):
    min_val=None
    max_val=None
    majx_vals, majy_vals, minx_vals, miny_vals = axis_vals
    if min_val == None:
        min_val = np.min(values)
    if max_val == None:
        max_val = np.max(values)
    cmap = matplotlib.cm.get_cmap('jet')
    norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
    fig, ax = plt.subplots(len(majy_vals)+1,len(majx_vals)+1,True,True)
    # Plot data
    img = None
    for i,j in itertools.product(range(len(majx_vals)),range(len(majy_vals))):
        img = ax[j,i+1].imshow(values[i,j,:,:].transpose(),
                interpolation='none',
                cmap=cmap)
        img.set_norm(norm)
        if i == 0:
            ax[j,i+1].set_ylabel(axis_labels[3])
        if j+1 == len(majy_vals):
            ax[j,i+1].set_xlabel(axis_labels[2])
        ax[j,i+1].set_xticklabels(minx_vals)
        ax[j,i+1].set_yticklabels(miny_vals)
        ax[j,i+1].set_adjustable('box-forced')
    # Major x-axis tick labels
    for i in range(len(majx_vals)):
        xlims = ax[-1,i+1].get_xlim()
        ylims = ax[-1,i+1].get_ylim()
        ax[-1,i+1].text((xlims[0]+xlims[1])/2,(ylims[0]+ylims[1])/2,majx_vals[i],horizontalalignment='center',verticalalignment='center')
        ax[-1,i+1].tick_params(labelcolor='none',
                top='off', bottom='off', left='off', right='off')
        ax[-1,i+1].grid(False)
        ax[-1,i+1].axis('off')
    # Major y-axis tick labels
    for j in range(len(majy_vals)):
        xlims = ax[-1,i+1].get_xlim()
        ylims = ax[-1,i+1].get_ylim()
        ax[j,0].text((xlims[0]+xlims[1])/2,(ylims[0]+ylims[1])/2,majy_vals[j],horizontalalignment='center',verticalalignment='center')
        ax[j,0].tick_params(labelcolor='none',
                top='off', bottom='off', left='off', right='off')
        ax[j,0].grid(False)
        ax[j,0].axis('off')
    # Get rid of bottom-left graph
    ax[-1,0].tick_params(labelcolor='none',
            top='off', bottom='off', left='off', right='off')
    ax[-1,0].grid(False)
    ax[-1,0].axis('off')
    # Major axis labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none',
            top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    fig.colorbar(img,ax=ax)

    # Save file
    output = os.path.join(directory, file_name)
    plt.savefig(output)
    print("Graph saved at %s" % output)
    plt.close(fig)
