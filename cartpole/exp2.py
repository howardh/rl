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
from learner.learner import Optimizer

import cartpole 
from cartpole import features
from cartpole import utils

import utils

discount_factors = ['1', '0.9', '0.8']
update_frequencies = ['50', '200']
behaviour_epsilons = ['1', '0.5', '0.1', '0']
target_epsilons = ['0', '0.01', '0.05']
sigmas = ['0', '0.25', '0.5', '0.75', '1']
trace_factors = ['0.01', '0.5', '0.99']
INDICES = pandas.MultiIndex.from_product(
        [discount_factors, update_frequencies, behaviour_epsilons,
            target_epsilons, sigmas, trace_factors],
        names=["Discount Factor", "Update Frequency", "Behaviour Epsilon",
            "Target Epsilon", "Sigma", "Lambda"])

def _run_trial(gamma, upd_freq, eps_b, eps_t, sigma, lam, directory=None,
        stop_when_learned=True, max_iters=10000, max_trials=None):
    """
    Run the learning algorithm on CartPole and return the number of
    iterations needed to learn the task.
    """
    try:
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
    except Exception as e:
        traceback.print_exc()

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
    except Exception as e:
        traceback.print_exc()
        raise e

def _worker2(params, directory=None):
    try:
        g,u,eb,et,s,l,m = params
        g = float(g)
        u = int(u)
        eb = float(eb)
        et = float(et)
        s = float(s)
        l = float(l)
        return _run_trial(g,u,eb,et,s,l,directory,False,5000,m)
    except KeyboardInterrupt:
        return None
    #except Exception as e:
    #    traceback.print_exc()
    #    raise e

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
                params = i + (n,)
                future = [executor.submit(_worker2, params, directory) for _ in range(n)]
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
    except Exception as e:
        print("Something broke")
        print(e)

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

def get_best_params(directory, sigma=None):
    d, sd = parse_results(directory)
    sigmas = d.index.get_level_values('Sigma').unique()
    by_sigma = dict()
    for s in sigmas:
        df = d.iloc[d.index.get_level_values('Sigma') == s]
        sorted_df = sorted([(i,df.loc[i]["Mean"]) for i in df.index], key=operator.itemgetter(1))
        by_sigma[s] = (df, sorted_df)
    best_params = dict()
    for s in sigmas:
        best_params[s] = by_sigma[s][1][-1][0]
    return best_params

def run2(n=1000, proc=10, params=None, directory=None):

    if directory is None:
        directory=os.path.join(utils.get_results_directory(),__name__,"part2")
    if params is None:
        params = get_best_params(os.path.join(utils.get_results_directory(),__name__,"part1"))

    print("Gridsearch")
    print("Environment: CartPole")
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
    print("Run through CartPole many times with the parameters obtained through gridsearch")

    futures = []
    from concurrent.futures import ProcessPoolExecutor
    try:
        with ProcessPoolExecutor(max_workers=proc) as executor:
            for s in params.keys():
                future = [executor.submit(_worker2, params[s]+(n,), directory) for
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
    #if os.path.isfile(results_file_name) and os.path.isfile(sorted_results_file_name):
    #    with open(results_file_name, 'rb') as f:
    #        data = dill.load(f)
    #    with open(sorted_results_file_name, 'rb') as f:
    #        sorted_data = dill.load(f)
    #    print(data)
    #    pprint.pprint(sorted_data)
    #    return

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
        try:
            full_path = os.path.join(directory,file_name)
            with open(full_path, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                results = [np.sum(eval(r[1])) for r in reader]
                if len(results) == 0:
                    os.remove(full_path)
                    continue
                data[sigma] += [results]
        except SyntaxError as e:
            print("Broken file: %s" % file_name)

    # Remove all incomplete series
    for s in data.keys():
        max_len = max([len(x) for x in data[s]])
        data[s] = [x for x in data[s] if len(x) == max_len]

    # Compute stuff
    m = dict()
    v = dict()
    s = dict()
    for sigma in data.keys():
        m[sigma] = np.mean(data[sigma], axis=0)
        v[sigma] = np.var(data[sigma], axis=0)
        s[sigma] = np.std(data[sigma], axis=0)

    # Plot
    for sigma in data.keys():
        u = params[sigma][1]
        plt.errorbar([i*int(u) for i in
            range(len(m[sigma]))],m[sigma],xerr=0,yerr=s[sigma],
            label=("sigma-%s"%sigma))
    plt.legend(loc='best')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(directory, "graph.png"))
    return data, m, s

def run_all():
    part1_dir = os.path.join(utils.get_results_directory(),__name__,"part1")
    part2_dir = os.path.join(utils.get_results_directory(),__name__,"part2")

    #run(directory=part1_dir)
    #parse_results(directory=part1_dir)
    run2(directory=part2_dir)
    parse_results2(directory=part2_dir)

if __name__ == "__main__":
    run_all()
