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

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent

import cartpole 
from cartpole import features
from cartpole import utils

import utils

discount_factors = ['1', '0.9', '0.8']
update_frequencies = ['50', '200', '500']
behaviour_epsilons = ['1', '0.5', '0.1', '0']
target_epsilons = ['0.1', '0.05', '0']
#sigmas = ['0', '0.25', '0.5', '0.75', '1']
sigmas = ['0', '0.5', '1']
trace_factors = ['0.01', '0.25', '0.5', '0.75', '0.99']

def _run_trial(gamma, upd_freq, eps_b, eps_t, sigma, lam, directory=None,
        stop_when_learned=False, max_iters=5000, max_trials=None):
    """
    Run the learning algorithm on CartPole and return the number of
    iterations needed to learn the task.
    """
    env_name = 'CartPole-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1])
    agent = LSTDAgent(
            action_space=action_space,
            num_features=cartpole.features.IDENTITY_NUM_FEATURES,
            discount_factor=gamma,
            features=cartpole.features.identity2,
            use_importance_sampling=False,
            use_traces=True,
            trace_factor=lam,
            sigma=sigma
    )
    agent.set_behaviour_policy("%.3f-epsilon"%eps_b)
    agent.set_target_policy("%.3f-epsilon"%eps_t)

    file_name,num = utils.find_next_free_file(
            "g%.3f-u%d-eb%.3f-et%.3f-s%.3f-l%.3f" % (gamma, upd_freq, eps_b, eps_t, sigma, lam),
            "csv", directory)
    if max_trials is not None and num >= max_trials:
        # Delete the created file, since we're not using it anymore
        os.remove(file_name) # TODO: This doesn't seem to be working
        return None
    with open(file_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for iters in range(0,max_iters+1):
            if iters % upd_freq == 0:
                agent.update_weights()
            if iters % 50 == 0:
                agent.set_target_policy("0-epsilon")
                rewards = agent.test(e, 100)
                agent.set_target_policy("%.3f-epsilon"%eps_t)
                csvwriter.writerow([iters, rewards])
                csvfile.flush()
                if stop_when_learned and np.mean(rewards) >= 190:
                    break
            agent.run_episode(e)
    return iters

def _worker(g,u,eb,et,s,l, directory=None):
    try:
        g = float(g)
        u = int(u)
        eb = float(eb)
        et = float(et)
        s = float(s)
        l = float(l)
        m = None
        return _run_trial(g,u,eb,et,s,l,directory,False,5000,m)
    except KeyboardInterrupt:
        return None
    except Exception as e:
        traceback.print_exc()
        raise e

def _worker2(params, directory=None):
    try:
        g,u,eb,et,s,l = params
        g = float(g)
        u = int(u)
        eb = float(eb)
        et = float(et)
        s = float(s)
        l = float(l)
        m = None
        return _run_trial(g,u,eb,et,s,l,directory,False,5000,m)
    except KeyboardInterrupt:
        return None
    #except Exception as e:
    #    traceback.print_exc()
    #    raise e

def run(n=10, proc=10, directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")
    print("Gridsearch")
    print("Environment: CartPole")
    print("Directory: %s" % directory)
    print("Determines the best combination of parameters by the number of iterations needed to learn.")

    discount_factors = ['0.9']
    update_frequencies = ['1', '50', '200']
    behaviour_epsilons = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
    target_epsilons = ['0', '0.1', '0.2', '0.3']
    sigmas = ['0', '0.25', '0.5', '0.75', '1']
    trace_factors = ['0', '0.25', '0.5', '0.75', '1']

    indices = pandas.MultiIndex.from_product(
            [discount_factors, update_frequencies, behaviour_epsilons,
                target_epsilons, sigmas, trace_factors],
            names=["Discount Factor", "Update Frequency", "Behaviour Epsilon",
                "Target Epsilon", "Sigma", "Lambda"])
    data = pandas.DataFrame(index=indices, columns=range(n))

    params = itertools.repeat(list(indices), n)
    params = itertools.chain.from_iterable(params)
    params = zip(params, itertools.repeat(directory))
    utils.cc(_worker2, params, proc=proc, keyworded=False)

def parse_results(directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")
    # Check that the experiment has been run and that results are present
    if not os.path.isdir(directory):
        print("No results to parse in %s" % directory)
        return None

    files = [f for f in os.listdir(directory) if
            os.path.isfile(os.path.join(directory,f))]
    all_params = utils.collect_file_params(files)
    data = utils.parse_results(directory)

    vals = [v for k,v in all_params.items()]
    keys = data.index.names
    print(all_params)

    # Graph stuff
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p_dict = dict([(k,next(iter(v))) for k,v in all_params.items()])
    for s,l in itertools.product(all_params['s'], all_params['l']):
        fig, ax = plt.subplots(1,1)
        ax.set_ylim([0,200])
        ax.set_xlabel('Behaviour epsilon')
        ax.set_ylabel('Cumulative reward')
        p_dict['s'] = s # sigma
        p_dict['l'] = l # trace factor
        for te in all_params['et']:
            x = []
            y = []
            p_dict['et'] = te
            for be in sorted(all_params['eb']):
                p_dict['eb'] = be
                param_vals = tuple([p_dict[k] for k in keys])

                x.append(float(be))
                y.append(data.loc[param_vals, 'MRS']/data.loc[param_vals, 'Count'])
                #ax.set_prop_cycle(monochrome)
            ax.plot(x,y,label='epsilon=%s'%te)
        ax.legend(loc='best')
        file_name = os.path.join(directory, 'graph-s%s-l%s.png' % (s,l))
        print("Saving file %s" % file_name)
        plt.savefig(file_name)
        plt.close(fig)

    return data

def get_best_params1(directory=None, sigma=None):
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

    print("Running Cartpole with a range of lambdas and sigmas.")
    print("Saving results in %s" % directory)

    discount_factors = ['0.9']
    update_frequencies = ['50']
    behaviour_epsilons = ['0.5']
    target_epsilons = ['0.5']
    sigmas = ['0', '0.25', '0.5', '0.75', '1']
    trace_factors = ['0', '0.25', '0.5', '0.75', '1']

    keys = ["g","u","eb","et","s","l"]
    params = []
    for vals in itertools.product(discount_factors, update_frequencies,
            behaviour_epsilons, target_epsilons, sigmas, trace_factors):
        d = dict(zip(keys,vals))
        d["directory"] = os.path.join(directory, "l%f"%float(d['l']))
        params.append(d)
    params = itertools.repeat(params, n)
    params = itertools.chain(*list(params))
    utils.cc(_worker, params, proc=proc, keyworded=True)

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
        data = utils.parse_graphing_results(d)

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
