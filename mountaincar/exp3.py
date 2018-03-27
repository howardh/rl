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
import sys
import traceback
import datetime
import random

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent

import mountaincar 
import mountaincar.features
import mountaincar.utils

import utils

discount_factors = ['1', '0.9', '0.8']
update_frequencies = ['50', '200', '500']
behaviour_epsilons = ['1', '0.5', '0.1', '0']
target_epsilons = ['0.1', '0.05', '0']
#sigmas = ['0', '0.25', '0.5', '0.75', '1']
sigmas = ['0', '0.5', '1']
trace_factors = ['0.01', '0.25', '0.5', '0.75', '0.99']

def lstd_rbft_control(discount_factor, initial_value, num_pos,
        num_vel, behaviour_eps, target_eps, trace_factor, sigma, update_freq, epoch, max_iters, test_iters,
        directory):
    args=locals()
    try:
        env_name = 'MountainCar-v0'
        e = gym.make(env_name)
        start_time = datetime.datetime.now()

        action_space = np.array([0,1,2])
        obs_space = np.array([[-1.2, .6], [-0.07, 0.07]])
        centres = np.array(list(itertools.product(
                np.linspace(0,1,num_pos),
                np.linspace(0,1,num_vel))))
        def norm(x):
            return np.array([(s-r[0])/(r[1]-r[0]) for s,r in zip(x, obs_space)])
        def rbf(x):
            x = norm(x)
            dist = np.power(centres-x, 2).sum(axis=1,keepdims=True)
            return np.exp(-100*dist)
        agent = LSTDAgent(
                action_space=action_space,
                discount_factor=discount_factor,
                #initial_value=initial_value,
                features=rbf,
                num_features=num_pos*num_vel,
                use_traces=True,
                trace_factor=trace_factor,
                sigma=sigma
        )
        agent.set_behaviour_policy("%f-epsilon" % behaviour_eps)
        agent.set_target_policy("%f-epsilon" % target_eps)

        rewards = []
        steps_to_learn = None
        for iters in range(0,max_iters+1):
            if iters % update_freq == 0:
                agent.update_weights()
            if epoch is not None:
                if iters % epoch == 0:
                    r = agent.test(e, test_iters, render=False, processors=1)
                    rewards.append(r)
                    if np.mean(r) >= -110:
                        if steps_to_learn is None:
                            steps_to_learn = iters
            else:
                if r >= -110:
                    if steps_to_learn is None:
                        steps_to_learn = iters
            agent.run_episode(e)

        while iters < max_iters: # Means it diverged at some point
            iters += 1
            rewards.append(None)

        data = (args, rewards, steps_to_learn)
        file_name, file_num = utils.find_next_free_file("results", "pkl",
                directory)
        with open(file_name, "wb") as f:
            dill.dump(data, f)

        return rewards,steps_to_learn
    except Exception as e:
        #print(e)
        traceback.print_exc()
        print("Iterations:`",iters)
        #print(utils.torch_svd_inv(agent.learner.a_mat).numpy())

def run1(n=10, proc=10, directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")
    print("Gridsearch")
    print("Environment: MountainCar")
    print("Directory: %s" % directory)
    print("Determines the best combination of parameters by the number of iterations needed to learn.")

    #def lstd_rbft_control(discount_factor, initial_value, num_pos,
    #        num_vel, behaviour_eps, target_eps, trace_factor, sigma, update_freq, epoch, max_iters, test_iters,

    behaviour_eps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    target_eps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    trace_factors = [0, 0.25, 0.5, 0.75, 1]
    sigmas = [0, 0.25, 0.5, 0.75, 1]

    keys = ['behaviour_eps', 'target_eps', 'sigma','trace_factor']
    params = []
    #for d in params:
    #    d["directory"] = os.path.join(directory, "l%f"%d['trace_factor'])
    for vals in itertools.product(behaviour_eps, target_eps, sigmas, trace_factors):
        d = dict(zip(keys,vals))
        d['discount_factor'] = 1
        d['initial_value'] = 0
        d['num_pos'] = 8
        d['num_vel'] = 8
        d['update_freq'] = 1
        d['epoch'] = 50
        d['max_iters'] = 3000
        d['test_iters'] = 1
        d["directory"] = os.path.join(directory, "l%f"%d['trace_factor'])
        params.append(d)
    params = itertools.repeat(params, n)
    params = itertools.chain(*list(params))
    params = list(params)
    random.shuffle(params)
    utils.cc(lstd_rbft_control, params, proc=proc, keyworded=True)

def parse_results1(directory=None):
    # Check that the experiment has been run and that results are present
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")
    if not os.path.isdir(directory):
        print("No results to parse in %s" % directory)
        return None

    # Parse set of parameters
    files = []
    for d,_,file_names in os.walk(directory):
        files += [os.path.join(d,f) for f in file_names if os.path.isfile(os.path.join(d,f))]
    all_params = utils.collect_file_params_pkl(files)
    del all_params['directory']
    kv_pairs = list(all_params.items())
    vals = [v for k,v in kv_pairs]
    keys = [k for k,v in kv_pairs]

    indices = pandas.MultiIndex.from_product(vals, names=keys)

    # A place to store our results
    data = pandas.DataFrame(0, index=indices, columns=["MaxS", "Count"])
    data.sort_index(inplace=True)

    # Load results from all pickle files
    types = {'sigma': float, 
            'trace_factor': float,
            'behaviour_eps': float,
            'target_eps': float}
    def cast_params(param_dict):
        for k in param_dict.keys():
            if k in types:
                param_dict[k] = types[k](param_dict[k])
        return param_dict
    for file_name in tqdm(files, desc="Parsing File Contents"):
        with open(os.path.join(directory,file_name), 'rb') as f:
            try:
                x = dill.load(f)
            except Exception as e:
                tqdm.write("Skipping %s" % file_name)
                continue
        params = x[0]
        params = cast_params(params)
        param_vals = tuple([params[k] for k in keys])

        # Stuff
        #print(data)
        #print(param_vals)
        series = x[1]
        series = np.mean(series, axis=1)
        data.loc[param_vals, 'MaxS'] += series[-1]
        data.loc[param_vals, 'Count'] += 1

    # Check for missing data
    missing_count = 0
    for i in data.index:
        if data.loc[i, 'Count'] <= 1:
            #print("No data for index ", i)
            print(dict(zip(keys, i)), ',')
            missing_count += 1
    print("%d data points missing" % missing_count)

    # Graph stuff
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p_dict = dict([(k,next(iter(v))) for k,v in all_params.items()])
    for s,l in itertools.product(all_params['sigma'], all_params['trace_factor']):
        fig, ax = plt.subplots(1,1)
        ax.set_ylim([-205,-100])
        ax.set_xlabel('Behaviour epsilon')
        ax.set_ylabel('Cumulative reward')
        p_dict['sigma'] = s
        p_dict['trace_factor'] = l
        for te in all_params['target_eps']:
            x = []
            y = []
            p_dict['target_eps'] = te
            for be in sorted(all_params['behaviour_eps']):
                p_dict['behaviour_eps'] = be
                p_dict = cast_params(p_dict)
                param_vals = tuple([p_dict[k] for k in keys])

                x.append(be)
                y.append(data.loc[param_vals, 'MaxS']/data.loc[param_vals, 'Count'])
                #ax.set_prop_cycle(monochrome)
            ax.plot(x,y)
        file_name = os.path.join(directory, 'graph-s%f-l%f.png' % (s,l))
        print("Saving file %s" % file_name)
        plt.savefig(file_name)
        plt.close(fig)

    return data,(x,y)

def get_best_params1(directory=None, sigma=None):
    raise NotImplementedError()
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")
    return utils.get_best_params_by_sigma(directory, learned_threshold=190)

def run2(n=1000, proc=10, params=None, directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part2")
    if params is None:
        params = get_best_params1()

    print("Environment: CartPole")
    print("Parameters: %s" % params)
    print("Running with best params found")
    print("Directory: %s" % directory)

    params = [p for p in params.values()]
    for p in params:
        p['directory'] = directory
    params = itertools.repeat(params, n)
    params = itertools.chain(*list(params))
    utils.cc(_worker, params, proc=proc, keyworded=True)

def parse_results2(directory=None):
    """
    Parse the CSV files produced by run2, and generates a graph.
    """
    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__,"part2")

    data = utils.parse_graphing_results(directory)

    # Plot
    for sigma in data.keys():
        mean = data[sigma][1]
        std = data[sigma][2]
        x = data[sigma][0]
        label = "sigma-%s"%sigma
        plt.fill_between(x, mean-std/2, mean+std/2, alpha=0.5)
        plt.plot(x, mean, label=label)
    plt.legend(loc='best')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(directory, "graph.png"))
    return data

def run3(n=100, proc=10, params=None, directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part3")
    print("Running MountainCar with a range of lambdas and sigmas.")
    print("Saving results in %s" % directory)
    #def lstd_rbft_control(discount_factor, initial_value, num_pos, num_vel,behaviour_eps, target_eps, trace_factor, sigma, update_freq, epoch, max_iters, test_iters, results_dir): 

    trace_factors = [0, 0.25, 0.5, 0.75, 1]
    sigmas = [0, 0.25, 0.5, 0.75, 1]

    keys = ['sigma','trace_factor']
    params = []
    for vals in itertools.product(sigmas, trace_factors):
        d = dict(zip(keys,vals))
        d['discount_factor'] = 1
        d['initial_value'] = 0
        d['num_pos'] = 8
        d['num_vel'] = 8
        d['behaviour_eps'] = 0.1
        d['target_eps'] = 0
        d['update_freq'] = 1
        d['epoch'] = 50
        d['max_iters'] = 3000
        d['test_iters'] = 1
        d["directory"] = os.path.join(directory, "l%f"%d['trace_factor'])
        params.append(d)
    params = itertools.repeat(params, n)
    params = itertools.chain(*list(params))
    params = list(params)
    random.shuffle(params)
    utils.cc(lstd_rbft_control, params, proc=proc, keyworded=True)

def parse_results3(directory=None):
    """
    Parse the CSV files produced by run2, and generates a graph.
    """
    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__,"part3")

    subdirs = [os.path.join(directory,f) for f in os.listdir(directory) if os.path.isdir(os.path.join(directory,f))]
    for d in subdirs:
        data = utils.parse_graphing_results_pkl(d)

        # Plot
        for sigma in data.keys():
            mean = data[sigma][1]
            std = data[sigma][2]
            x = data[sigma][0]
            label = "sigma-%s"%sigma
            plt.fill_between(x, mean-std/2, mean+std/2, alpha=0.5)
            plt.plot(x, mean, label=label)
        plt.legend(loc='best')
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.savefig(os.path.join(d, "graph.png"))
        plt.close()

        for sigma in data.keys():
            mean = data[sigma][1]
            x = data[sigma][0]
            label = "sigma-%s"%sigma
            plt.plot(x, mean, label=label)
        plt.legend(loc='best')
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.savefig(os.path.join(d, "graph2.png"))
        plt.close()

def run_all(proc=10):
    part1_dir = os.path.join(utils.get_results_directory(),__name__,"part1")
    part2_dir = os.path.join(utils.get_results_directory(),__name__,"part2")

    run(directory=part1_dir, proc=proc)
    parse_results(directory=part1_dir)
    run2(directory=part2_dir, proc=proc)
    parse_results2(directory=part2_dir)

if __name__ == "__main__":
    run_all()
