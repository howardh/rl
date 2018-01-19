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

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent
from learner.learner import Optimizer

import frozenlake
import utils

discount_factors = ['1', '0.99', '0.9']
learning_rates = ['0.1', '0.01', '0.001']
optimizers = ['Optimizer.RMS_PROP', 'Optimizer.NONE']
indices = pandas.MultiIndex.from_product(
        [discount_factors, learning_rates, optimizers],
        names=["Discount Factor", "Learning Rate", "Optimizer"])

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

def _run_trial(gamma, alpha, op, directory=None):
    env_name = 'FrozenLake-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = TabularAgent(
            action_space=action_space,
            discount_factor=gamma,
            learning_rate=alpha,
            optimizer=op)

    file_name = _find_next_free_file("g%f-a%f-o%s" % (gamma, alpha, op), "csv", directory)
    with open(file_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for iters in range(1,10000):
            agent.run_episode(e)
            if iters % 500 == 0:
                rewards = agent.test(e, 100, max_steps=1000)
                csvwriter.writerow([iters, rewards])
                csvfile.flush()
                if np.mean(rewards) >= 0.78:
                    break
    return iters

def _worker(i, directory=None):
    try:
        g,a,o = indices[i]
        g = float(g)
        a = float(a)
        o = eval(o)
        return _run_trial(g,a,o,directory)
    except KeyboardInterrupt:
        return None

def run(n=10, proc=10, directory=None):
    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__)
    print("Gridsearch")
    print("Environment: FrozenLake4x4")
    print("Parameter space:")
    print("""
            \tDiscount factor: %s
            \tLearning rate: %s
            \tOptimizer: %s
    """ % (discount_factors, learning_rates, optimizers))
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

if __name__ == "__main__":
    run()
