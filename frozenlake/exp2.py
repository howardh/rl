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

discount_factors = ['1', '0.99', '0.9']
update_frequencies = ['50', '200', '500']
behaviour_epsilons = ['1', '0.5', '0.1', '0']
target_epsilons = ['0', '0.01', '0.05']
indices = pandas.MultiIndex.from_product(
        [discount_factors, update_frequencies, behaviour_epsilons, target_epsilons],
        names=["Discount Factor", "Update Frequencies", "Behaviour Epsilon", "Target Epsilon"])

def _find_next_free_file(prefix, suffix, directory):
    import os
    if not os.path.isdir(directory):
        os.makedirs(directory)
    i = 0
    while True:
        path=os.path.join(directory,"%s-%d.%s" % (prefix, i, suffix))
        if not os.path.isfile(path):
            break
        i += 1
    return path

def _run_trial(gamma, upd_freq, eps_b, eps_t, directory=None):
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

    file_name = _find_next_free_file(
            "g%.3f-u%d-eb%.3f-et%.3f" % (gamma, upd_freq, eps_b, eps_t),
            "csv", directory)
    with open(file_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for iters in range(1,10000):
            agent.run_episode(e)
            if iters % upd_freq == 0:
                agent.update_weights()
                rewards = agent.test(e, 100)
                csvwriter.writerow([iters, rewards])
                csvfile.flush()
                if np.mean(rewards) >= 0.78:
                    break
    return iters

def _worker(i, directory=None):
    try:
        g,u,eb,et = indices[i]
        g = float(g)
        u = int(u)
        eb = float(eb)
        et = float(et)
        return _run_trial(g,u,eb,et,directory)
    except KeyboardInterrupt:
        return None

def run(n=10, proc=10, directory="results/%s/%s"%(__name__, time.strftime("%Y-%m-%d_%H-%M-%S"))):
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
    data = pandas.DataFrame(index=indices, columns=range(n))

    futures = []
    from concurrent.futures import ProcessPoolExecutor
    try:
        with ProcessPoolExecutor(max_workers=proc) as executor:
            for i in tqdm(range(len(indices)), desc="Adding jobs"):
                future = [executor.submit(_worker, i, directory) for _ in range(n)]
                data.loc[indices[i]] = future
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
    for i in tqdm(range(len(indices)), desc="Retrieving results"):
        future = data.loc[indices[i]]
        data.loc[indices[i]] = [f.result() if f.done() else f for f in future]
    if len(futures) > 0:
        sorted_data = None # TODO
    else:
        sorted_data = sorted([(str(i),np.mean(data.loc[i])) for i in indices], key=operator.itemgetter(1))
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

    indices = pandas.MultiIndex.from_product([g, u, eb, et],
            names=["Discount Factor", "Update Frequencies", "Behaviour Epsilon", "Target Epsilon"])
    # A place to store our results
    data = pandas.DataFrame(0, index=indices, columns=["Sum", "Count", "Mean"])

    # Load results from all csv files
    for file_name in tqdm(files, desc="Parsing File Contents"):
        #print("Parsing %s" % file_name)
        regex_result = pattern.match(file_name)
        if regex_result is None:
            print("Invalid file. Skipping.")
            continue
        gamma = regex_result.group(1)
        upd_freq = regex_result.group(4)
        eps_b = regex_result.group(7)
        eps_t = regex_result.group(10)
        #print("Gamma %s, uf %s, eps_b %s, eps_t %s" % (gamma, upd_freq, eps_b, eps_t))
        with open(os.path.join(directory,file_name), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            results = [(int(r[0]), np.sum(eval(r[1]))) for r in reader]
            row = data.loc[(gamma, upd_freq, eps_b, eps_t)]
            row['Sum'] += results[-1][0]
            row['Count'] += 1
            row['Mean'] = row['Sum']/row['Count']

    # Display results
    with pandas.option_context('display.max_rows', None, 'display.max_columns', 3):
        print(data)
    return data

if __name__ == "__main__":
    run()
