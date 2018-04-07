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

    while len(rewards) < (max_iters%epoch)+1: # Means it diverged at some point
        rewards.append([0]*test_iters)

    data = (args, rewards, steps_to_learn)
    file_name, file_num = utils.find_next_free_file("results", "pkl", directory)
    with open(file_name, "wb") as f:
        dill.dump(data, f)

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
        g,a,eb,et,s,l = params
        g = float(g)
        a = float(a)
        eb = float(eb)
        et = float(et)
        s = float(s)
        l = float(l)
        m = None
        return _run_trial(g,a,eb,et,s,l,directory,False,5000,m)
    except KeyboardInterrupt:
        return None

def run(n=10, proc=10, directory=None):
    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part1")
    print("Gridsearch")
    print("Environment: CartPole")
    print("Directory: %s" % directory)
    print("Determines the best combination of parameters by the number of iterations needed to learn.")

    behaviour_eps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    target_eps = [0, 0.1, 0.2, 0.3, 0.4]
    trace_factors = [0, 0.25, 0.5, 0.75, 1]
    sigmas = [0, 0.25, 0.5, 0.75, 1]
    learning_rate = np.logspace(np.log10(10),np.log10(.001),num=13,endpoint=True,base=10).tolist()

    
    #def _run_trial(gamma, alpha, eps_b, eps_t, sigma, lam, directory=None,
    #    max_iters=5000, epoch=50, test_iters=1):
    keys = ['eps_b', 'eps_t', 'sigma','lam', 'alpha']
    params = []
    #for d in params:
    #    d["directory"] = os.path.join(directory, "l%f"%d['lam'])
    for vals in itertools.product(behaviour_eps, target_eps, sigmas,
            trace_factors, learning_rate):
        d = dict(zip(keys,vals))
        d['gamma'] = 0.9
        d['epoch'] = 50
        d['max_iters'] = 5000
        d['test_iters'] = 1
        d["directory"] = os.path.join(directory, "l%f"%d['lam'])
        params.append(d)
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
    return data
    keys = data.index.names
    all_params = dict([(k, set(data.index.get_level_values(k))) for k in keys])

    ## Check for missing data
    #count = 0
    #for i in data.index:
    #    if data.loc[i, 'MaxS'] > 1:
    #        #print("No data for index ", i)
    #        #print(dict(zip(keys, i)), ',')
    #        count += 1
    #print("%d non-diverged data points" % count)

    # Graph stuff
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p_dict = dict([(k,next(iter(v))) for k,v in all_params.items()])
    for s,l in itertools.product(all_params['sigma'],all_params['lam']):
        fig, ax = plt.subplots(1,1)
        ax.set_ylim([0,200])
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
