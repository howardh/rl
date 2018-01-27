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

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent
from learner.learner import Optimizer

import frozenlake
import frozenlake.features
import frozenlake.utils

import utils

discount_factors = ['1', '0.99', '0.9']
update_frequencies = ['50', '200']
behaviour_epsilons = ['1', '0.5', '0.1', '0']
target_epsilons = ['0', '0.01', '0.05']
sigmas = ['0', '0.5', '1']
trace_factors = ['0.01', '0.5', '0.99']
INDICES = pandas.MultiIndex.from_product(
        [discount_factors, update_frequencies, behaviour_epsilons,
            target_epsilons, sigmas, trace_factors],
        names=["Discount Factor", "Update Frequency", "Behaviour Epsilon",
            "Target Epsilon", "Sigma", "Lambda"])

def _run_trial(gamma, upd_freq, eps_b, eps_t, sigma, lam,
        directory=os.path.join(utils.get_results_directory(),"temp",__name__,"part1"),
        stop_when_learned=True, max_iters=10000):
    """
    Run the learning algorithm on FrozenLake and return the number of
    iterations needed to learn the task.
    """
    env_name = 'FrozenLake-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = LSTDAgent(
            action_space=action_space,
            num_features=frozenlake.features.ONE_HOT_NUM_FEATURES,
            discount_factor=gamma,
            features=frozenlake.features.one_hot,
            use_importance_sampling=False,
            use_traces=True,
            trace_factor=lam,
            sigma=sigma,
            cuda=False
    )
    agent.set_behaviour_policy("%.3f-epsilon"%eps_b)
    agent.set_target_policy("%.3f-epsilon"%eps_t)

    file_name,_ = utils.find_next_free_file(
            "g%.3f-u%d-eb%.3f-et%.3f-s%.3f-l%.3f" % (gamma, upd_freq, eps_b, eps_t, sigma, lam),
            "csv", directory)
    with open(file_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for iters in range(1,max_iters):
            agent.run_episode(e)
            if iters % upd_freq == 0:
                agent.update_weights()
                rewards = agent.test(e, 100)
                csvwriter.writerow([iters, rewards])
                csvfile.flush()
                if stop_when_learned and np.mean(rewards) >= 0.78:
                    break
    return iters

def _worker(i, directory=None):
    try:
        g,u,eb,et,s,l = INDICES[i]
        g = float(g)
        u = int(u)
        eb = float(eb)
        et = float(et)
        s = float(s)
        l = float(l)
        return _run_trial(g,u,eb,et,s,l,directory)
    except KeyboardInterrupt:
        return None

def _worker2(params, directory=None):
    try:
        g,u,eb,et,s,l = params
        g = float(g)
        u = int(u)
        eb = float(eb)
        et = float(et)
        s = float(s)
        l = float(l)
        return _run_trial(g,u,eb,et,s,l,directory,False,5000)
    except KeyboardInterrupt:
        return None

def run(n=10, proc=20, directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")
    print("Gridsearch")
    print("Environment: FrozenLake4x4")
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
    data = pandas.DataFrame(index=INDICES, columns=range(n))

    futures = []
    from concurrent.futures import ProcessPoolExecutor
    try:
        with ProcessPoolExecutor(max_workers=proc) as executor:
            for i in tqdm(INDICES, desc="Adding jobs"):
                future = [executor.submit(_worker2, i, directory) for _ in range(n)]
                data.loc[i] = future
                futures += future
            pbar = tqdm(total=len(futures), desc="Job completion")
            while len(futures) > 0:
                count = [f.done() for f in futures].count(True)
                pbar.update(count)
                futures = [f for f in futures if not f.done()]
                time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard Interrupt Detected")
    except Exception:
        print("Something broke")

def parse_results(directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")

    data = utils.parse_results(directory, 0.78)

    # Split by sigma value
    sigmas = list(data.index.get_level_values('s').unique())
    by_sigma = dict()
    for sigma in sigmas:
        df = data.iloc[data.index.get_level_values('s') == sigma]
        sorted_df = utils.sort_data(df)
        by_sigma[sigma] = (df, sorted_df[0], sorted_df[1])

    return data, by_sigma

def get_best_params(directory, sigma=None):
    d, sd = parse_results(directory)
    by_sigma = dict()
    best_params = dict()
    for s,d in sd.items():
        df, by_mr, by_ttl = d
        by_sigma[s] = (df, by_mr, by_ttl)
        best_params[s] = dict(zip(df.index.names,by_sigma[s][2][0][0]))
    return best_params

def run2(n=1000, proc=20, params=None, directory=None):

    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part2")
    if params is None:
        params = get_best_params(os.path.join(utils.get_results_directory(),__name__,"part1"))

    for k,v in params.items():
        if type(v) is dict:
            params[k] = (v['g'], v['u'], v['eb'], v['et'], v['s'], v['l'])
        elif type(v) is tuple:
            pass
        else:
            raise Exception("Invalid Parameters")

    print("Gridsearch")
    print("Environment: FrozenLake4x4")
    print("Parameter space:")
    for s in params.keys():
        print("\tSigma: %s" % s)
        print("""
                \tDiscount factor: %s
                \tUpdate Frequency: %s
                \tBehaviour Epsilon: %s
                \tTarget Epsilon: %s
                \tSigma: %s
                \tTrace Factor: %s
        """ % params[s])
    print("Run through FrozenLake many times with the parameters obtained through gridsearch")

    futures = []
    from concurrent.futures import ProcessPoolExecutor
    try:
        with ProcessPoolExecutor(max_workers=proc) as executor:
            for s in params.keys():
                future = [executor.submit(_worker2, params[s], directory) for
                        _ in tqdm(range(n), desc="Adding jobs Sigma=%s"%s)]
                futures += future
            pbar = tqdm(total=len(futures), desc="Job completion")
            while len(futures) > 0:
                count = [f.done() for f in futures].count(True)
                pbar.update(count)
                futures = [f for f in futures if not f.done()]
                time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard Interrupt Detected")
    except Exception:
        print("Something broke")

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

    # Check if pickle files are there
    results_file_name = os.path.join(directory, "results.pkl") 
    sorted_results_file_name = os.path.join(directory, "sorted_results.pkl") 
    if os.path.isfile(results_file_name) and os.path.isfile(sorted_results_file_name):
        with open(results_file_name, 'rb') as f:
            data = dill.load(f)
        with open(sorted_results_file_name, 'rb') as f:
            sorted_data = dill.load(f)
        print(data)
        pprint.pprint(sorted_data)
        return

    # Parse set of parameters
    files = [f for f in os.listdir(directory) if
            os.path.isfile(os.path.join(directory,f))]
    pattern = re.compile(r'^g((0|[1-9]\d*)(\.\d+)?)-u((0|[1-9]\d*)(\.\d+)?)-eb((0|[1-9]\d*)(\.\d+)?)-et((0|[1-9]\d*)(\.\d+)?)-s((0|[1-9]\d*)(\.\d+)?)-l((0|[1-9]\d*)(\.\d+)?)-(0|[1-9]\d*)\.csv$')

    # A place to store our results
    data = dict()
    params = dict()

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
        if sigma not in params.keys():
            params[sigma] = (gamma, upd_freq, eps_b, eps_t, sigma, lam)
        if sigma not in data.keys():
            data[sigma] = []
        with open(os.path.join(directory,file_name), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            results = [np.sum(eval(r[1])) for r in reader]
            data[sigma] += [results]

    # Remove all incomplete series
    for s in data.keys():
        max_len = max([len(x) for x in data[s]])
        data[s] = [x for x in data[s] if len(x) == max_len]

    # Compute stuff
    m = dict()
    v = dict()
    s = dict()
    u = dict()
    for sigma in data.keys():
        m[sigma] = np.mean(data[sigma], axis=0)
        v[sigma] = np.var(data[sigma], axis=0)
        s[sigma] = np.std(data[sigma], axis=0)
        u[sigma] = params[sigma][1]

    # Plot
    for sigma in data.keys():
        mean = m[sigma]
        std = s[sigma]
        x = [i*int(u[sigma]) for i in range(len(mean))]
        label = "sigma-%s"%sigma
        plt.fill_between(x, mean-std/2, mean+std/2, alpha=0.5)
        plt.plot(x, mean, label=label)
    plt.legend(loc='best')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(directory, "graph.png"))
    return data, m, s, u

def run_all(proc=10):
    run(proc=proc)
    parse_results()
    run2(proc=proc)
    parse_results2()

if __name__ == "__main__":
    run_all()
