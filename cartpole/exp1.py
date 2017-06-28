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

import cartpole
from cartpole import features
from cartpole import utils

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
            sigma=1
    )
    agent.set_behaviour_policy("%.3f-epsilon"%eps_b)
    agent.set_target_policy("%.3f-epsilon"%eps_t)

    file_name = utils.find_next_free_file(
            "g%.3f-u%d-eb%.3f-et%.3f" % (gamma, upd_freq, eps_b, eps_t),
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
                if stop_when_learned and np.mean(rewards) >= 190:
                    break
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

def run(n=10, proc=10,
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")):
    print("Gridsearch")
    print("Environment: CartPole")
    print("Parameter space:")
    print("""
            \tDiscount factor: %s
            \tUpdate Frequencies: %s
            \tBehaviour Epsilons: %s
            \tTarget Epsilons: %s
    """ % (discount_factors, update_frequencies, behaviour_epsilons, target_epsilons))
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

def parse_results(directory):
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
    pattern = re.compile(r'^g((0|[1-9]\d*)(\.\d+)?)-u((0|[1-9]\d*)(\.\d+)?)-eb((0|[1-9]\d*)(\.\d+)?)-et((0|[1-9]\d*)(\.\d+)?)-(0|[1-9]\d*)\.csv$')
    g = set()
    u = set()
    eb = set()
    et = set()
    for file_name in tqdm(files, desc="Parsing File Names"):
        regex_result = pattern.match(file_name)
        if regex_result is None:
            continue
        g.add(regex_result.group(1))
        u.add(regex_result.group(4))
        eb.add(regex_result.group(7))
        et.add(regex_result.group(10))

    indices = pandas.MultiIndex.from_product([g, u, eb, et],
            names=["Discount Factor", "Update Frequencies", "Behaviour Epsilon", "Target Epsilon"])
    # A place to store our results
    data = pandas.DataFrame(0, index=indices, columns=["Sum", "Count", "Mean"])

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
        with open(os.path.join(directory,file_name), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            results = [(int(r[0]), np.sum(eval(r[1]))) for r in reader]
            row = data.loc[(gamma, upd_freq, eps_b, eps_t)]
            row['Sum'] += results[-1][0]
            row['Count'] += 1
            row['Mean'] = row['Sum']/row['Count']

    # Display results
    sorted_data = sorted([(i,np.mean(data.loc[i])) for i in indices], key=operator.itemgetter(1))
    sorted_data.reverse()
    return data, sorted_data

def get_best_params(directory):
    d, sd = parse_results(directory)
    return eval(sd[-1][0])

def run2(n=1000, proc=10, params=None, directory=None):

    if params is None:
        params = get_best_params(os.path.join(utils.get_results_directory(),__name__,"part1"))
    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__,"part2")

    print(type(params))

    print("Gridsearch")
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
    pattern = re.compile(r'^g((0|[1-9]\d*)(\.\d+)?)-u((0|[1-9]\d*)(\.\d+)?)-eb((0|[1-9]\d*)(\.\d+)?)-et((0|[1-9]\d*)(\.\d+)?)-(0|[1-9]\d*)\.csv$')
    g = set()
    u = set()
    eb = set()
    et = set()
    for file_name in tqdm(files, desc="Parsing File Names"):
        regex_result = pattern.match(file_name)
        if regex_result is None:
            continue
        g.add(regex_result.group(1))
        u.add(regex_result.group(4))
        eb.add(regex_result.group(7))
        et.add(regex_result.group(10))

    # A place to store our results
    data = []

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
        with open(os.path.join(directory,file_name), 'r') as csvfile:
            try:
                reader = csv.reader(csvfile, delimiter=',')
                results = [np.sum(eval(r[1])) for r in reader]
                data += [results]
            except Exception:
                print("Error processing %s" % file_name)

    # Remove all incomplete series
    max_len = max([len(x) for x in data])
    data = [x for x in data if len(x) == max_len]

    # Compute stuff
    m = np.mean(data, axis=0)
    v = np.var(data, axis=0)
    s = np.std(data, axis=0)

    # Plot
    print(len(m))
    print(len([i*int(next(iter(u))) for i in range(len(m))]))
    plt.errorbar([i*int(next(iter(u))) for i in range(len(m))],m,xerr=0,yerr=s)
    plt.savefig(os.path.join(directory, "graph.png"))
    return data, m, v

def run_all():
    run(directory=os.path.join(utils.get_results_directory(),__name__,"part1"))
    run2(directory=os.path.join(utils.get_results_directory(),__name__,"part2"))
    parse_results2(directory=os.path.join(utils.get_results_directory(),__name__,"part2"))

if __name__ == "__main__":
    run_all()
