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

def run(n=10, proc=10, directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")
    print("Gridsearch")
    print("Environment: CartPole")
    print("Directory: %s" % directory)
    print("Parameter space:")
    print("""
            \tDiscount factor: %s
            \tUpdate Frequencies: %s
            \tBehaviour Epsilons: %s
            \tTarget Epsilons: %s
            \tSigmas: %s
            \tTrace Factors: %s
    """ % (discount_factors, update_frequencies, behaviour_epsilons, target_epsilons, sigmas, trace_factors))
    print("Determines the best combination of parameters by the number of iterations needed to learn.")
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

def parse_results(directory):
    # Check that the experiment has been run and that results are present
    if not os.path.isdir(directory):
        print("No results to parse in %s" % directory)
        return None

    import re
    # Check if pickle files are there
    results_file_name = os.path.join(directory, "results.pkl") 
    sorted_results_file_name = os.path.join(directory, "sorted_results.pkl") 
    if os.path.isfile(results_file_name) and os.path.isfile(sorted_results_file_name):
        with open(results_file_name, 'rb') as f:
            data = dill.load(f)
        with open(sorted_results_file_name, 'rb') as f:
            sorted_data = dill.load(f)
        return data, sorted_data

    # Parse set of parameters
    files = [f for f in os.listdir(directory) if
            os.path.isfile(os.path.join(directory,f))]
    pattern = re.compile(r'^g((0|[1-9]\d*)(\.\d+)?)-u((0|[1-9]\d*)(\.\d+)?)-eb((0|[1-9]\d*)(\.\d+)?)-et((0|[1-9]\d*)(\.\d+)?)-s((0|[1-9]\d*)(\.\d+)?)-l((0|[1-9]\d*)(\.\d+)?)-(0|[1-9]\d*)\.csv$')
    g = set()
    u = set()
    eb = set()
    et = set()
    s = set()
    l = set()
    for file_name in tqdm(files, desc="Parsing File Names"):
        regex_result = pattern.match(file_name)
        if regex_result is None:
            continue
        g.add(regex_result.group(1))
        u.add(regex_result.group(4))
        eb.add(regex_result.group(7))
        et.add(regex_result.group(10))
        s.add(regex_result.group(13))
        l.add(regex_result.group(16))

    indices = pandas.MultiIndex.from_product([g, u, eb, et, s, l],
            names=["Discount Factor", "Update Frequency", "Behaviour Epsilon", "Target Epsilon", "Sigma", "Trace Factor"])
    # A place to store our results
    data = pandas.DataFrame(0, index=indices, columns=["Sum", "Count", "Mean"]) # Average sum of rewards
    data2 = pandas.DataFrame(0, index=indices, columns=["Sum", "Count", "Mean"]) # Average time to learn
    for i in data.index:
        data.loc[i]['Mean'] = sys.maxsize # FIXME: Hack. Can't use np.inf here.

    # Load results from all csv files
    for file_name in tqdm(files, desc="Parsing File Contents"):
        regex_result = pattern.match(file_name)
        if regex_result is None:
            print("Invalid file. Skipping.")
            continue
        gamma = regex_result.group(1)
        upd_freq = regex_result.group(4)
        eps_b = regex_result.group(7)
        eps_t = regex_result.group(10)
        sigma = regex_result.group(13)
        lam = regex_result.group(16)
        with open(os.path.join(directory,file_name), 'r') as csvfile:
            try:
                reader = csv.reader(csvfile, delimiter=',')
                results = [(int(r[0]), np.sum(eval(r[1]))) for r in reader]
                time_to_learn = None
                for a,b in results:
                    if b >= 190:
                        time_to_learn = a
                        break
                if time_to_learn is None:
                    time_to_learn = results[-1][0]

                # Sum of rewards
                row = data.loc[(gamma, upd_freq, eps_b, eps_t, sigma, lam)]
                row['Sum'] += results[-1][1]
                row['Count'] += 1
                row['Mean'] = row['Sum']/row['Count']

                # Sum of learning time
                row2 = data.loc[(gamma, upd_freq, eps_b, eps_t, sigma, lam)]
                row['Sum'] += time_to_learn
                row['Count'] += 1
                row['Mean'] = row['Sum']/row['Count']
            except Exception:
                pass

    # Save results
    sorted_data = sorted([(i,data.loc[i]["Mean"]) for i in indices], key=operator.itemgetter(1))
    sorted_data.reverse()
    with open(results_file_name, 'wb') as f:
        dill.dump(data, f)
    with open(sorted_results_file_name, 'wb') as f:
        dill.dump(sorted_data, f)

    # Split by sigma value
    by_sigma = dict()
    for sigma in s:
        df = data.iloc[data.index.get_level_values('Sigma') == sigma]
        sorted_df = sorted([(i,df.loc[i]["Mean"]) for i in df.index], key=operator.itemgetter(1))
        sorted_df.reverse()
        by_sigma[sigma] = (df, sorted_df)

    return data, sorted_data

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
        d['behaviour_eps'] = 0
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
