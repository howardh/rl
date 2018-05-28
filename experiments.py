import numpy as np
import gym
import itertools
import pandas
import dill
import csv
import os
from tqdm import tqdm
import time
import operator
import pprint
import sys
import traceback
import random
import fabulous
from fabulous.color import bold
import scipy
import scipy.stats

import utils
import graph

# Parameter computation

def get_params_nondiverged(exp, directory):
    data = utils.parse_results(directory, exp.LEARNED_REWARD)
    d = data.loc[data['MaxS'] > 1]
    params = [dict(zip(d.index.names,p)) for p in tqdm(d.index)]
    for d in params:
        d["directory"] = os.path.join(directory, "l%f"%d['lam'])
    return params

def get_mean_rewards(exp, directory):
    data = utils.parse_results(directory, exp.LEARNED_REWARD)
    mr_data = data.apply(lambda row: row.MRS/row.Count, axis=1)
    return mr_data

def get_final_rewards(exp, directory):
    data = utils.parse_results(directory, exp.LEARNED_REWARD)
    fr_data = data.apply(lambda row: row.MaxS/row.Count, axis=1)
    return fr_data

def get_ucb1_mean_reward(exp, directory):
    data = utils.parse_results(directory, exp.LEARNED_REWARD)
    count_total = data['Count'].sum()
    def ucb1(row):
        a = row.MRS/row.Count
        b = np.sqrt(2*np.log(count_total)/row.Count)
        return a+b
    score = data.apply(ucb1, axis=1)
    return score

def get_ucb1_final_reward(exp, directory):
    data = utils.parse_results(directory, exp.LEARNED_REWARD)
    count_total = data['Count'].sum()
    def ucb1(row):
        a = row.MaxS/row.Count
        b = np.sqrt(2*np.log(count_total)/row.Count)
        return a+b
    score = data.apply(ucb1, axis=1)
    return score

def get_params_best(exp, directory, score_function, n=1, params={}):
    if type(params) is list:
        output = []
        for p in params:
            output += get_params_best(exp, directory, score_function,n,p)
        return output
    score = score_function(exp, directory)
    if len(params) > 0:
        keys = tuple([k for k in params.keys()])
        vals = tuple([params[k] for k in keys])
        score = score.xs(vals,level=keys)
    if n == -1:
        n = score.size
    if n == 1:
        output_params = [score.idxmax()]
    else:
        score = score.sort_values(ascending=False)
        output_params = itertools.islice(score.index, n)
    output_params = [dict(zip(score.index.names,p)) for p in output_params]
    for p in output_params:
        p.update(params)
    return output_params

# Experiments

def run1(exp, n=1, proc=10, directory=None):
    if directory is None:
        directory=exp.get_directory()

    print("-"*80)
    print("Gridsearch: Run n trials on each combination of parameters.")
    to_print = [
            ("Environment",exp.ENV_NAME),
            ("Directory", directory)]
    for heading,val in to_print:
        print("%s: %s" % (bold(heading),val))

    params = exp.get_params_gridsearch()
    for p in params:
        p['directory'] = directory
    params = itertools.repeat(params, n)
    params = itertools.chain(*list(params))
    params = list(params)
    params = utils.split_params(params)
    random.shuffle(params)
    utils.cc(exp.run_trial, params, proc=proc, keyworded=True)

def run2(exp, n=1, m=10, proc=10, directory=None):
    if directory is None:
        directory=exp.get_directory()

    params1 = get_params_best(exp, directory, get_ucb1_mean_reward, m)
    params2 = get_params_best(exp, directory, get_ucb1_final_reward, m)
    params = params1+params2

    print("-"*80)
    print("Further refining gridsearch, exploring with UCB1")
    to_print = [
            ("Environment",exp.ENV_NAME),
            #("Parameters", params),
            ("Directory", directory)]
    for heading,val in to_print:
        print("%s: %s" % (bold(heading),val))

    for p in params:
        p['directory'] = directory
    params = itertools.repeat(params, n)
    params = itertools.chain(*list(params))
    params = list(params)
    params = utils.split_params(params)
    utils.cc(exp.run_trial, params, proc=proc, keyworded=True)

def run3(exp, n=100, proc=10, params=None, directory=None,
        by_mean_reward=True, by_final_reward=True):
    if directory is None:
        directory=exp.get_directory()
    if params is None:
        if by_mean_reward:
            params1 = get_params_best(exp, directory, get_mean_rewards, 1)
        else:
            params1 = []
        if by_final_reward:
            params2 = get_params_best(exp, directory, get_final_rewards, 1)
        else:
            params2 = []
        params = params1+params2

    print("-"*80)
    print("Running more trials with the best parameters found so far.")
    to_print = [
            ("Environment",exp.ENV_NAME),
            ("Parameters", params),
            ("Directory", directory)]
    for heading,val in to_print:
        print("%s: %s" % (bold(heading),val))

    for p in params:
        p['directory'] = directory
    params = itertools.repeat(params, n)
    params = itertools.chain(*list(params))
    params = list(params)
    params = utils.split_params(params)
    utils.cc(exp.run_trial, params, proc=proc, keyworded=True)

# Plotting

def plot_final_rewards(exp):
    directory=exp.get_directory()

    # Check that the experiment has been run and that results are present
    if not os.path.isdir(directory):
        print("No results to parse in %s" % directory)
        return None

    data = utils.parse_results(directory, exp.LEARNED_REWARD)
    data = data.apply(lambda row: row.MRS/row.Count, axis=1)
    keys = data.index.names
    all_params = dict([(k, set(data.index.get_level_values(k))) for k in keys])

    # Plot parameters
    plot_params = exp.get_plot_params_final_rewards()
    x_axis = plot_params['x_axis']
    best_of = plot_params['best_of']
    average = plot_params['average']
    each_curve = plot_params['each_curve']
    each_plot = plot_params['each_plot']
    file_name_template = plot_params['file_name_template']
    label_template = plot_params['label_template']
    xlabel = plot_params['xlabel']
    ylabel = plot_params['ylabel']

    # Graph stuff
    p_dict = dict([(k,next(iter(v))) for k,v in all_params.items()])
    # Loop over plots
    for pp in itertools.product(*[all_params[k] for k in each_plot]):
        output_data = []
        for k,v in zip(each_plot,pp):
            p_dict[k] = v
        # Loop over curves in a plot
        for pc in itertools.product(*[sorted(all_params[k]) for k in each_curve]):
            for k,v in zip(each_curve,pc):
                p_dict[k] = v
            x = []
            y = []
            for px in sorted(all_params[x_axis]):
                p_dict[x_axis] = px
                vals = []
                for pb in itertools.product(*[sorted(all_params[k]) for k in best_of]):
                    for k,v in zip(each_curve,pb):
                        p_dict[k] = v
                    param_vals = tuple([p_dict[k] for k in keys])
                    vals.append(data.loc[param_vals])
                x.append(float(px))
                y.append(np.max(vals))
            output_data.append((x,y,None,label_template.format(**p_dict)))
        graph.graph_data(data=output_data, 
                file_name=file_name_template.format(**p_dict),
                directory=directory,
                ylims=[exp.MIN_REWARD,exp.MAX_REWARD],
                xlabel=xlabel, ylabel=ylabel)
    return data

def plot_best(exp):
    directory=exp.get_directory()

    plot_params = exp.get_plot_params_best()
    file_name = plot_params['file_name']
    label_template = plot_params['label_template']
    if 'param_filters' in plot_params:
        param_filters = plot_params['param_filters']
    else:
        param_filters = [{}]
    if len(param_filters) == 0:
        param_filters = [{}]

    data = []
    #score_functions = [get_mean_rewards, get_final_rewards]
    score_functions = [get_mean_rewards]
    labels = ['mean reward', 'final reward']
    for sf,l in zip(score_functions,labels):
        for param_filter in param_filters:
            params = get_params_best(exp, directory, sf, 1, param_filter)[0]
            print("Plotting params: ", params)
            data.append(graph.get_data(
                params, 
                directory,
                label=label_template.format(**params)))
    graph.graph_data(data, file_name, directory, xlabel='Episodes',
            ylabel='Cumulative Reward')

    return data

def plot_best_trials(exp, n):
    directory=exp.get_directory()

    params = get_params_best(exp, directory, get_mean_rewards, 1, [{}])[0][0]
    x,ys,label = graph.get_data_individual(params,directory)
    means = [np.mean(y) for y in ys]
    data = [(x,y,None,'') for y in ys]
    sorted_data = sorted(list(zip(data,means)),key=lambda a: a[1])
    data = sorted_data[:n]+sorted_data[-n:]
    data = [d[0] for d in data]
    graph.graph_data(data, 'all-trials.png', directory, xlabel='Episodes',
            ylabel='Cumulative Reward')

def plot_t_test(exp,label_template=''):
    directory=exp.get_directory()
    param_filters = [{}]
    if hasattr(exp,'get_param_filters'):
        param_filters = exp.get_param_filters()
    params = get_params_best(exp, directory, get_mean_rewards, 1, param_filters)
    data = []
    for p in params:
        x,ys,_ = graph.get_data_individual(p,directory)
        data.append((x,ys,p))
    output_data = []
    all_ps = [[None]*len(data) for _ in range(len(data))]
    for i1,i2 in itertools.combinations(range(len(data)),2):
        x1,y1,_ = data[i1]
        x2,y2,_ = data[i2]
        t = scipy.stats.ttest_ind(y1,y2,axis=0,equal_var=False)[1]
        all_ps[i1][i2] = t
        all_ps[i2][i1] = t
    for i in range(len(data)):
        all_ps[i][i] = label_template.format(**data[i][2])
    graph.graph_data_histogram(all_ps, 't-test-hist.png',
            directory,xlabel='p-value',ylabel='frequency')
    return

def plot_gridsearch(exp):
    """Plot a matrix of plots showing the average performance for each set of
    parameters.
    """
    directory=exp.get_directory()

    plot_param = exp.get_plot_params_gridsearch()
    axis_params = plot_param['axis_params']
    axis_labels = plot_param['axis_labels']

    data = utils.get_all_series(directory)
    all_params = dict([(k, sorted(list(set(data.index.get_level_values(k))))) for k in data.index.names])

    axis_vals = [all_params[k] for k in axis_params]
    vals = np.zeros([len(a) for a in axis_vals])
    for ind_val in itertools.product(*[enumerate(av) for av in axis_vals]):
        index = [i for i,v in ind_val]
        val = [v for i,v in ind_val]
        majx,majy,minx,miny = val
        series = data.xs(val,level=axis_params)
        def foo(row):
            try:
                return np.mean(row)
            except TypeError as e:
                print(row)
                print([len(r) for r in row])
                raise e
        #means = series.apply(lambda row: np.mean(row))
        means = series.apply(foo)
        vals[index[0],index[1],index[2],index[3]] = means.max()

    graph.graph_matrix('gridsearch.png', directory,
            axis_vals, vals, axis_labels=axis_labels)

    return

def plot_custom():
    from frozenlake import exp2

    directory=exp2.get_directory()

    all_params = [
            {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.0, 'lam': 0.25},
            {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.1, 'lam': 0.25},
            {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.2, 'lam': 0.25},
            {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.3, 'lam': 0.25}
    ]
    all_params = [
            {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.0, 'lam': 0.0},
            {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.0, 'lam': 0.25},
            {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.0, 'lam': 0.5},
            {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.0, 'lam': 0.75},
            {'eps_b': 0.0, 'test_iters': 50, 'sigma': 1.0, 'gamma': 1, 'epoch': 10, 'alpha': 0.4641588833612779, 'max_iters': 2000, 'eps_t': 0.0, 'lam': 1.0}
    ]
    data = []
    for params in all_params:
        print("Plotting params: ", params)
        #data.append(graph.get_data(params, directory,
        #        label='SGD eps_t=%f'%params['eps_t']))
        data.append(graph.get_data(params, directory,
                label='SGD lam=%f'%params['lam']))
    graph.graph_data(data, 'graph-custom.png', directory)

    return data

# Plotting across experiments

def plot_custom_best_mean(exps, labels):
    data = []
    for exp,l in zip(exps,labels):
        directory = exp.get_directory()
        params = get_params_best(exp, directory, get_mean_rewards, 1)[0]
        print("Plotting params: ", params)
        data.append(graph.get_data(
            params, 
            directory,
            label=l))
    output_directory = os.path.join(utils.get_results_directory(),__name__)
    graph.graph_data(data, 'foo.png', output_directory, xlabel='Episodes',
            ylabel='Cumulative Reward')
