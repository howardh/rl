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

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent
from learner.learner import Optimizer

import frozenlake
from frozenlake import features
from frozenlake import utils

import utils

discount_factors = ['1', '0.99', '0.9']
update_frequencies = ['50', '200', '500']
behaviour_epsilons = ['1', '0.5', '0.1', '0']
target_epsilons = ['0', '0.01', '0.05']
INDICES = pandas.MultiIndex.from_product(
        [discount_factors, update_frequencies, behaviour_epsilons, target_epsilons],
        names=["Discount Factor", "Update Frequencies", "Behaviour Epsilon", "Target Epsilon"])

def _run_trial(gamma, upd_freq, eps_b, eps_t, directory=None,
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
            sigma=1
    )

    file_name,_ = utils.find_next_free_file(
            "g%.3f-u%d-eb%.3f-et%.3f" % (gamma, upd_freq, eps_b, eps_t),
            "csv", directory)
    with open(file_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for iters in range(0,max_iters+1):
            if iters % upd_freq == 0:
                agent.update_weights()
                rewards = agent.test(e, 100)
                csvwriter.writerow([iters, rewards])
                csvfile.flush()
                if stop_when_learned and np.mean(rewards) >= 0.78:
                    break
            agent.run_episode(e)
    return iters

def _worker(i, directory=None):
    try:
        g,u,eb,et = INDICES[i]
        g = float(g)
        u = int(u)
        eb = float(eb)
        et = float(et)
        return _run_trial(g,u,eb,et,directory)
    except KeyboardInterrupt:
        return None

def _worker2(params, directory=None):
    try:
        g,u,eb,et = params
        g = float(g)
        u = int(u)
        eb = float(eb)
        et = float(et)
        return _run_trial(g,u,eb,et,directory,False,5000)
    except KeyboardInterrupt:
        return None
    except Exception as e:
        #print(e)
        pass

def run(n=10, proc=10, directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")
    print("Gridsearch")
    print("Environment: FrozenLake4x4")
    print("Parameter space:")
    print("""
            \tDiscount factor: %s
            \tUpdate Frequencies: %s
            \tBehaviour Epsilons: %s
            \tTarget Epsilons: %s
    """ % (discount_factors, update_frequencies, behaviour_epsilons, target_epsilons))
    print("Determines the best combination of parameters by the number of iterations needed to learn.")
    print("Directory: %s" % directory)

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
    for i in tqdm(INDICES, desc="Retrieving results"):
        future = data.loc[i]
        data.loc[i] = [f.result() if f.done() else f for f in future]
    if len(futures) > 0:
        sorted_data = None # TODO
    else:
        sorted_data = sorted([(str(i),np.mean(data.loc[i])) for i in INDICES], key=operator.itemgetter(1))
        sorted_data.reverse()
    data.to_csv(os.path.join(directory, "results.csv"))
    dill.dump(data, open(os.path.join(directory, "results.pkl"),'wb'))
    dill.dump(sorted_data, open(os.path.join(directory, "sorted_results.pkl"),'wb'))
    return data, sorted_data

def parse_results(directory, learned_threshold=None):
    """
    Given a directory containing the results of experiments as CSV files,
    compute statistics on the data, and return it as a Pandas dataframe.

    CSV format:
        Two columns:
        * Time
            An integer value, which could represent episode count, step count,
            or clock time
        * Rewards
            A list of floating point values, where each value represents the
            total reward obtained from each test run. This list may be of any
            length, and must be wrapped in double quotes.
    e.g.
        0,"[0,0,0]"
        10,"[1,0,0,0,0]"
        20,"[0,1,0,1,0]"
        30,"[0,1,1,0,1,0,1]"

    Pandas Dataframe format:
        Columns:
        * MRS - Mean Reward Sum
            Given the graph of the mean testing reward over time, the MRS is
            the average of these testing rewards over time.
        * TTLS - Time to Learn Sum
            The first time step at which the testing reward matches/surpasses the given
            threshold. Units may be in episodes, steps, or clock time,
            depending on the units used in the CSV data.
        * Count
            Number of trials that were run with the given parameters.
        Indices:
            Obtained from the parameters in the file names.
    """
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
    params = utils.collect_file_params(files)
    param_names = list(params.keys())
    param_vals = [params[k] for k in param_names]

    # A place to store our results
    indices = pandas.MultiIndex.from_product(param_vals, names=param_names)
    data = pandas.DataFrame(0, index=indices, columns=["MRS", "TTLS", "Count"])
    data['MRS'] = data.MRS.astype(float)
    data.sortlevel(inplace=True)

    # Load results from all csv files
    for file_name in tqdm(files, desc="Parsing File Contents"):
        file_params = utils.parse_file_name(file_name)
        if file_params is None:
            print("Invalid file. Skipping.")
            continue
        output = utils.parse_file(os.path.join(directory,file_name),
                learned_threshold)
        if output is None:
            tqdm.write("Skipping empty file: %s" % file_name)
            continue
        mr,ttl = output
        key = tuple([file_params[k] for k in param_names])
        data.loc[key,'MRS'] += mr
        if ttl is None:
            data.loc[key,'TTLS'] = None
        else:
            data.loc[key,'TTLS'] += ttl
        data.loc[key,'Count'] += 1

    # Display results
    print("Sorting by MR")
    sorted_by_mr = [(i,data.loc[i,'MRS']/data.loc[i,'Count']) for i in data.index]
    sorted_by_mr = sorted(sorted_by_mr, key=operator.itemgetter(1))
    sorted_by_mr.reverse()
    print("Sorting by TTL")
    sorted_by_ttl = [(i,data.loc[i,'TTLS']/data.loc[i,'Count']) for i in data.index]
    sorted_by_ttl = sorted(sorted_by_ttl, key=operator.itemgetter(1))
    return data, sorted_by_mr, sorted_by_ttl

def get_best_params(directory):
    d, sd_mr, sd_ttl = parse_results(directory, 0.78)
    names = list(d.index.names)
    vals = [eval(x) for x in sd_ttl[0][0]]
    params = dict(zip(names, vals))
    return params['g'], params['u'], params['eb'], params['et']

def run2(n=1000, proc=10, params=None, directory=None):
    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__,"part2")
    if params is None:
        params = get_best_params(os.path.join(utils.get_results_directory(),__name__,"part1"))

    print("Environment: FrozenLake4x4")
    print("Parameter space:")
    print("""
            \tDiscount factor: %s
            \tUpdate Frequency: %s
            \tBehaviour Epsilon: %s
            \tTarget Epsilon: %s
    """ % params)
    print("Run through FrozenLake many times with the parameters obtained through gridsearch")

    futures = []
    from concurrent.futures import ProcessPoolExecutor
    try:
        with ProcessPoolExecutor(max_workers=proc) as executor:
            future = [executor.submit(_worker2, params, directory) for
                    _ in tqdm(range(n), desc="Adding jobs")]
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
    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__,"part2")

    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    params = utils.collect_file_params(files)
    param_names = list(params.keys())
    param_vals = [params[k] for k in param_names]

    # A place to store our results
    data = []

    # Load results from all csv files
    for file_name in tqdm(files, desc="Parsing File Contents"):
        if utils.parse_file_name(file_name) is None:
            print("Invalid file. Skipping.")
            continue
        with open(os.path.join(directory,file_name), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            results = [np.sum(eval(r[1])) for r in reader]
            data += [results]

    # Remove all incomplete series
    max_len = max([len(x) for x in data])
    data = [x for x in data if len(x) == max_len]

    # Compute stuff
    m = np.mean(data, axis=0)
    v = np.var(data, axis=0)
    s = np.std(data, axis=0)
    u = int(list(params['u'])[0])

    # Plot
    print(len(m))
    print(len([i*u for i in range(len(m))]))
    plt.errorbar([i*u for i in range(len(m))],m,xerr=0,yerr=s)
    plt.savefig(os.path.join(directory, "graph.png"))
    return data, m, s, u

def run_all(proc=10):
    run(proc=proc,directory=os.path.join(utils.get_results_directory(),__name__,"part1"))
    run2(proc=proc,directory=os.path.join(utils.get_results_directory(),__name__,"part2"))
    parse_results2(directory=os.path.join(utils.get_results_directory(),__name__,"part2"))

if __name__ == "__main__":
    run_all()
