import numpy as np
import gym
import itertools
import pandas
import multiprocessing
import dill
import csv
import os
from tqdm import tqdm
import time
import operator
import pprint
import random

from agent.linear_agent import LinearAgent

import cartpole
from cartpole import ENV_NAME
from cartpole import MAX_REWARD
from cartpole import MIN_REWARD
from cartpole import LEARNED_REWARD
from cartpole import features
from cartpole import utils

import utils

def _run_trial(gamma, alpha, eps_b, eps_t, sigma, lam, directory=None,
        max_iters=5000, epoch=50, test_iters=1):
    """
    Run the learning algorithm on CartPole and return the number of
    iterations needed to learn the task.
    """
    args = locals()
    env_name = 'CartPole-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1])
    agent = LinearAgent(
            action_space=action_space,
            learning_rate=alpha,
            num_features=cartpole.features.IDENTITY_NUM_FEATURES,
            discount_factor=gamma,
            features=cartpole.features.identity2,
            trace_factor=lam,
            sigma=sigma
    )
    agent.set_behaviour_policy("%.3f-epsilon"%eps_b)
    agent.set_target_policy("%.3f-epsilon"%eps_t)

    rewards = []
    steps_to_learn = None
    try:
        for iters in range(0,max_iters+1):
            if epoch is not None:
                if iters % epoch == 0:
                    r = agent.test(e, test_iters, render=False, processors=1)
                    rewards.append(r)
                    if np.mean(r) >= 190:
                        if steps_to_learn is None:
                            steps_to_learn = iters
            else:
                if r >= 190:
                    if steps_to_learn is None:
                        steps_to_learn = iters
            agent.run_episode(e)
    except ValueError as e:
        tqdm.write(str(e))
        tqdm.write("Diverged")

    while len(rewards) < (max_iters/epoch)+1: # Means it diverged at some point
        rewards.append([0]*test_iters)

    data = (args, rewards, steps_to_learn)
    file_name, file_num = utils.find_next_free_file("results", "pkl", directory)
    with open(file_name, "wb") as f:
        dill.dump(data, f)

def get_params_custom():
    params = []
    return params

def get_params_gridsearch():
    behaviour_eps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    target_eps = [0, 0.1, 0.2, 0.3, 0.4]
    trace_factors = [0, 0.25, 0.5, 0.75, 1]
    sigmas = [0, 0.25, 0.5, 0.75, 1]
    learning_rate = np.logspace(np.log10(10),np.log10(.001),num=13,endpoint=True,base=10).tolist()

    keys = ['eps_b', 'eps_t', 'sigma','lam', 'alpha']
    params = []
    for vals in itertools.product(behaviour_eps, target_eps, sigmas,
            trace_factors, learning_rate):
        d = dict(zip(keys,vals))
        d['gamma'] = 0.9
        d['epoch'] = 50
        d['max_iters'] = 5000
        d['test_iters'] = 1
        d["directory"] = os.path.join(directory, "l%f"%d['lam'])
        params.append(d)
    return params

def get_params_nondiverged(directory):
    data = utils.parse_results_pkl(directory, LEARNED_REWARD)
    d = data.loc[data['MaxS'] > 1]
    params = [dict(zip(d.index.names,p)) for p in tqdm(d.index)]
    for d in params:
        d["directory"] = os.path.join(directory, "l%f"%d['lam'])
    return params

def get_mean_rewards(directory):
    data = utils.parse_results_pkl(directory, LEARNED_REWARD)
    mr_data = data.apply(lambda row: row.MRS/row.Count, axis=1)
    return mr_data

def get_final_rewards(directory):
    data = utils.parse_results_pkl(directory, LEARNED_REWARD)
    fr_data = data.apply(lambda row: row.MaxS/row.Count, axis=1)
    return fr_data

def get_ucb1_mean_reward(directory):
    data = utils.parse_results_pkl(directory, LEARNED_REWARD)
    count_total = data['Count'].sum()
    def ucb1(row):
        a = row.MRS/row.Count
        b = np.sqrt(2*np.log(count_total)/row.Count)
        return a+b
    score = data.apply(ucb1, axis=1)
    return score

def get_ucb1_final_reward(directory):
    data = utils.parse_results_pkl(directory, LEARNED_REWARD)
    count_total = data['Count'].sum()
    def ucb1(row):
        a = row.MaxS/row.Count
        b = np.sqrt(2*np.log(count_total)/row.Count)
        return a+b
    score = data.apply(ucb1, axis=1)
    return score

def get_params_best(directory, score_function, n=1):
    score = score_function(directory)
    if n == -1:
        n = score.size
    if n == 1:
        params = [score.idxmax()]
    else:
        score = score.sort_values(ascending=False)
        params = itertools.islice(score.index, n)
    return [dict(zip(score.index.names,p)) for p in params]

def run(n=1, proc=10, directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")
    print("Gridsearch")
    print("Environment: ", ENV_NAME)
    print("Directory: %s" % directory)
    print("Determines the best combination of parameters by the number of iterations needed to learn.")

    params = get_params_gridsearch()
    for p in params:
        p['directory'] = directory
    params = itertools.repeat(params, n)
    params = itertools.chain(*list(params))
    params = list(params)
    random.shuffle(params)
    utils.cc(_run_trial, params, proc=proc, keyworded=True)

def parse_results(directory=None):
    # Check that the experiment has been run and that results are present
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")
    if not os.path.isdir(directory):
        print("No results to parse in %s" % directory)
        return None

    data = utils.parse_results_pkl(directory, 190)
    keys = data.index.names
    all_params = dict([(k, set(data.index.get_level_values(k))) for k in keys])

    # Graph stuff
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p_dict = dict([(k,next(iter(v))) for k,v in all_params.items()])
    for s,l in itertools.product(all_params['sigma'],all_params['lam']):
        fig, ax = plt.subplots(1,1)
        ax.set_ylim([0,210])
        ax.set_xlabel('Behaviour epsilon')
        ax.set_ylabel('Cumulative reward')
        p_dict['sigma'] = s
        p_dict['lam'] = l
        for te in all_params['eps_t']:
            x = []
            y = []
            p_dict['eps_t'] = te
            for be in sorted(all_params['eps_b']):
                p_dict['eps_b'] = be
                #p_dict = cast_params(p_dict)
                m = 0
                for a in all_params['alpha']:
                    p_dict['alpha'] = a
                    param_vals = tuple([p_dict[k] for k in keys])
                    val = data.loc[param_vals, 'MaxS']/data.loc[param_vals, 'Count']
                    m = max(m,val)
                x.append(be)
                y.append(m)
                #ax.set_prop_cycle(monochrome)
            ax.plot(x,y,label='epsilon=%f'%te)
        ax.legend(loc='best')
        file_name = os.path.join(directory, 'graph-s%f-l%f.png' % (s,l))
        print("Saving file %s" % file_name)
        plt.savefig(file_name)
        plt.close(fig)

    return data

def get_non_diverged(directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")
    data = utils.parse_results_pkl(directory, 190)
    d = data.loc[data['MaxS'] > 50]
    params = [dict(zip(d.index.names,p)) for p in tqdm(d.index)]
    return params

def get_best_params1(directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")
    data = utils.parse_results_pkl(directory, 190)
    #params = [('lam', 1.0), ('sigma', 1.0), ('eps_b', 0.4)]
    #for k,v in params:
    #    data = data.xs(v, level=k)
    #ttl_data = data.apply(lambda row: row.TTLS/row.Count, axis=1)
    #best_ttl_params = ttl_data.idxmin()
    mr_data = data.apply(lambda row: row.MRS/row.Count, axis=1)
    best_mr_params = mr_data.idxmax()
    fr_data = data.apply(lambda row: row.MaxS/row.Count, axis=1)
    best_fr_params = fr_data.idxmax()
    print("Done computing best parameters")
    #print("Best ttl params: ", best_ttl_params)
    #print("Best ttl: ", ttl_data.loc[best_ttl_params])
    print("Best mr params: ", best_mr_params)
    print("Best mr: ", mr_data.loc[best_mr_params])
    print("Best fr params: ", best_fr_params)
    print("Best fr: ", fr_data.loc[best_fr_params])
    keys = data.index.names
    #best_ttl_params = dict(zip(keys,best_ttl_params))
    best_mr_params = dict(zip(keys,best_mr_params))
    best_fr_params = dict(zip(keys,best_fr_params))
    #return best_ttl_params, best_mr_params, best_fr_params
    #best_mr_params.update(params)
    #best_fr_params.update(params)
    return best_mr_params, best_fr_params

def run2(n=1000, proc=10, params=None, directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")
    if params is None:
        params = get_best_params1()

    print("Environment: CartPole")
    print("Parameters: ", params)
    print("Running with best params found")
    print("Directory: %s" % directory)

    for d in params:
        d["directory"] = os.path.join(directory, "l%f"%d['lam'])
    params = itertools.repeat(params, n)
    params = itertools.chain(*list(params))
    params = list(params)
    random.shuffle(params)
    utils.cc(_run_trial, params, proc=proc, keyworded=True)

def parse_results2(directory=None, params=None, labels=None):
    """
    Parse the CSV files produced by run2, and generates a graph.
    """
    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if directory is None:
        input_directory = os.path.join(utils.get_results_directory(),__name__,"part1")
        output_directory = os.path.join(utils.get_results_directory(),__name__,"part1")
    else:
        input_directory = directory
        output_directory = directory
    if params is None:
        params = get_best_params1()
    if labels is None:
        labels = ['']*len(params)

    print("Computing series with given params")
    data = [utils.get_series_with_params_pkl(input_directory, p) for p in params]

    # Plot
    fig, ax = plt.subplots(1,1)
    ax.set_ylim([0,210])
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")
    for l,d in zip(labels,data):
        print(np.array(d).shape)
        mean = np.mean(d,axis=0)
        std = np.std(d,axis=0)
        x = [50*i for i in range(len(mean))] # FIXME temporary solution because lazy
        plt.fill_between(x, mean-std/2, mean+std/2, alpha=0.5)
        ax.plot(x, mean, label=l)
        print(mean)
    ax.legend(loc='best')
    file_name = os.path.join(output_directory, "graph.png")
    plt.savefig(file_name)
    plt.close(fig)
    print("Saved file: ", file_name)
    return data

def plot_final_rewards(directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")
    # Check that the experiment has been run and that results are present
    if not os.path.isdir(directory):
        print("No results to parse in %s" % directory)
        return None

    data = utils.parse_results_pkl(directory, LEARNED_REWARD)
    data = data.apply(lambda row: row.MRS/row.Count, axis=1)
    keys = data.index.names
    all_params = dict([(k, set(data.index.get_level_values(k))) for k in keys])

    # Graph stuff
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x_axis = 'eps_b'
    best_of = []
    average = []
    each_curve = ['eps_t']
    each_plot = ['sigma', 'lam', 'alpha']
    file_name_template = 'graph-s{sigma}-l{lam}-a{alpha}.png'
    label_template = 'epsilon={eps_t}'

    print(all_params)

    p_dict = dict([(k,next(iter(v))) for k,v in all_params.items()])
    # Loop over plots
    for p1 in itertools.product(*[all_params[k] for k in each_plot]):
        for k,v in zip(each_plot,p1):
            p_dict[k] = v
        fig, ax = plt.subplots(1,1)
        ax.set_ylim([MIN_REWARD,MAX_REWARD])
        ax.set_xlabel('Behaviour epsilon')
        ax.set_ylabel('Cumulative reward')
        # Loop over curves in a plot
        for p2 in itertools.product(*[sorted(all_params[k]) for k in each_curve]):
            for k,v in zip(each_curve,p2):
                p_dict[k] = v
            x = []
            y = []
            for px in sorted(all_params[x_axis]):
                p_dict[x_axis] = px
                param_vals = tuple([p_dict[k] for k in keys])
                x.append(float(px))
                y.append(data.loc[param_vals])
            ax.plot(x,y,label=label_template.format(**p_dict))
        ax.legend(loc='best')
        file_name = os.path.join(directory, file_name_template.format(**p_dict))
        print("Saving file %s" % file_name)
        plt.savefig(file_name)
        plt.close(fig)

    return data

def plot_best(directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = []

    fig, ax = plt.subplots(1,1)
    ax.set_ylim([MIN_REWARD,MAX_REWARD])
    ax.set_xlabel('Behaviour epsilon')
    ax.set_ylabel('Cumulative reward')
    for score_function in [get_mean_rewards, get_final_rewards]:
        params = get_params_best(directory, score_function, 1)[0]
        print("Plotting params: ", params)

        series = utils.get_series_with_params_pkl(directory, params)
        mean = np.mean(series, axis=0)
        std = np.std(series, axis=0)
        epoch = params['epoch']
        x = [i*epoch for i in range(len(mean))]
        data.append((x, mean, std, 'SGD'))
        ax.plot(x,mean,label='SGD')
    ax.legend(loc='best')
    file_name = os.path.join(directory, 'graph-best.png')
    print("Saving file %s" % file_name)
    plt.savefig(file_name)
    plt.close(fig)

    return data

def run_all():
    run(directory=os.path.join(utils.get_results_directory(),__name__,"part1"))
    run2(directory=os.path.join(utils.get_results_directory(),__name__,"part2"))
    parse_results2(directory=os.path.join(utils.get_results_directory(),__name__,"part2"))

if __name__ == "__main__":
    run_all()
