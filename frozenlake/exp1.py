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

def _run_trial(gamma, alpha, op, directory=None, break_when_learned=False,
        n_episodes=10000):
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
                if break_when_learned and np.mean(rewards) >= 0.78:
                    break
    #print("Policy:")
    #frozenlake.utils.print_policy(agent)
    return iters

def _run_trial1(gamma, alpha, op, directory=None):
    return _run_trial(gamma, alpha, op, directory, True, 10000)

def _run_trial2(gamma, alpha, op, directory=None):
    return _run_trial(gamma, alpha, op, directory, False, 5000)

def _worker(i, directory=None):
    try:
        g,a,o = indices[i]
        g = float(g)
        a = float(a)
        o = eval(o)
        return _run_trial1(g,a,o,directory)
    except KeyboardInterrupt:
        return None

def run(n=10, proc=10, directory=None):
    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__,"part1")
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

def get_best_params(directory):
    with open(os.path.join(directory,"sorted_results.pkl"), "rb") as f:
        sorted_data = dill.load(f)
    return sorted_data[-1][0]

def run2(n=10, proc=10, params=None, directory=None):
    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__,"part2")
    if params is None:
        params = get_best_params(os.path.join(utils.get_results_directory(),__name__,"part1"))
        
    print("Environment: FrozenLake4x4")
    print("Parameters:")
    print("""
            \tDiscount factor: %s
            \tLearning rate: %s
            \tOptimizer: %s
    """ % (discount_factors, learning_rates, optimizers))
    print("Determines the best combination of parameters by the number of iterations needed to learn.")

    params = [eval(p) for p in eval(params)] + [directory]
    utils.cc(_run_trial2, itertools.repeat(params,n), proc=proc, keyworded=False)

def parse_results(directory=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if directory is None:
        directory = os.path.join(utils.get_results_directory(),__name__,"part2")

    files = [f for f in os.listdir(directory) if
            os.path.isfile(os.path.join(directory,f))]
    data = []
    for file_name in tqdm(files, desc="Parsing File Contents"):
        try:
            full_path = os.path.join(directory,file_name)
            with open(full_path, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                results = [np.sum(eval(r[1])) for r in reader]
                if len(results) == 0:
                    os.remove(full_path)
                    continue
                data.append(results)
        except SyntaxError as e:
            print("Broken file: %s" % file_name)
        except Exception as e:
            print("Broken file: %s" % file_name)

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    plt.fill_between(range(0,len(mean)*500,500), mean-std/2, mean+std/2, alpha=0.5)
    plt.plot(range(0,len(mean)*500,500), mean, label="Tabular")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    output = os.path.join(directory, "graph.png")
    plt.savefig(output)
    print("Graph saved at %s" % output)

    return (mean,std)

if __name__ == "__main__":
    run()
