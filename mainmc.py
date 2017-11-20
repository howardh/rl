import gym
import numpy as np
import datetime
import dill
import os
import itertools
from tqdm import tqdm
import time
import traceback


from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent
from agent.rbf_agent import RBFAgent

import mountaincar 
import mountaincar.features
import mountaincar.utils

#from mountaincar import exp1
import utils

def tabular_control(discount_factor, learning_rate, initial_value, num_pos,
        num_vel, behaviour_eps, target_eps, epoch, max_iters, test_iters):
    env_name = 'MountainCar-v0'
    e = gym.make(env_name)
    start_time = datetime.datetime.now()

    action_space = np.array([0,1,2])
    agent = TabularAgent(
            action_space=action_space,
            features=mountaincar.features.get_one_hot(num_pos,num_vel),
            discount_factor=discount_factor,
            learning_rate=learning_rate,
            initial_value=initial_value
    )
    agent.set_behaviour_policy("%f-epsilon" % behaviour_eps)
    #agent.set_behaviour_policy(mountaincar.utils.get_one_hot_optimal_policy(num_pos, num_vel, 0.75))
    #agent.set_behaviour_policy(utils.less_optimal_policy)
    agent.set_target_policy("%f-epsilon" % target_eps)

    rewards = []
    steps_to_learn = None
    iters = 0
    while True:
        iters += 1
        r,s = agent.run_episode(e)
        rewards.append(r)
        print("... %d\r" % iters, end='')
        if r != -200:
            print("Iteration: %d\t Reward: %d"%(iters, r))
        if epoch is not None and iters % epoch == 0:
            print("Testing...")
            #print(agent.learner.weights.transpose())
            rewards = agent.test(e, test_iters, render=False, processors=1)
            print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
            if np.mean(rewards) >= -110:
                if steps_to_learn is None:
                    steps_to_learn = iters
        if iters > max_iters:
            break
    return rewards,steps_to_learn

def parse_results(directory):
    results = []
    file_names = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]
    for file_name in file_names:
        with open(os.path.join(directory,file_name), 'rb') as f:
            x = dill.load(f)
            diverged = None in x[1]
            if diverged:
                x[1] = [a for a in x[1] if a is not None]
            if x[2] is None:
                results.append((x[0], file_name, np.mean(x[1]), np.inf, diverged))
            else:
                results.append((x[0], file_name, np.mean(x[1]), x[2], diverged))
    print("\nSorting by mean reward...")
    results.sort(key=lambda x: x[2], reverse=True)
    output1 = results[0][0]
    for i in range(min(10,len(results))):
        print(results[i][2])
        print("\t%s" % results[i][0])
        print("\t%s" % results[i][1])
        print("\t%s" % results[i][3])
        print("\tDiverged?: %s" % results[i][4])

    print("\nSorting by time to learn...")
    results.sort(key=lambda x: x[3], reverse=False)
    output2 = results[0][0]
    for i in range(min(10,len(results))):
        print(results[i][3])
        print("\t%s" % results[i][0])
        print("\t%s" % results[i][1])
        print("\t%s" % results[i][2])
        print("\tDiverged?: %s" % results[i][4])

    return output1, output2

def rbf_control(discount_factor, learning_rate, initial_value, num_pos,
        num_vel, behaviour_eps, target_eps, epoch, max_iters, test_iters,
        results_dir):
    args = locals()
    print(args)
    env_name = 'MountainCar-v0'
    e = gym.make(env_name)
    start_time = datetime.datetime.now()

    action_space = np.array([0,1,2])
    obs_space = np.array([[-1.2, .6], [-0.07, 0.07]])
    def norm(x):
        return np.array([(s-r[0])/(r[1]-r[0]) for s,r in zip(x, obs_space)])
    agent = RBFAgent(
            action_space=action_space,
            observation_space=np.array([[0,1],[0,1]]),
            discount_factor=discount_factor,
            learning_rate=learning_rate,
            initial_value=initial_value,
            features=norm
    )
    agent.set_behaviour_policy("%f-epsilon" % behaviour_eps)
    agent.set_target_policy("%f-epsilon" % target_eps)

    rewards = []
    steps_to_learn = None
    iters = 0
    try:
        while iters < max_iters:
            iters += 1
            r,s = agent.run_episode(e)
            rewards.append(r)
            if epoch is not None:
                if iters % epoch == 0:
                    print("Testing...")
                    rewards = agent.test(e, test_iters, render=False, processors=1)
                    if np.mean(rewards) >= -110:
                        if steps_to_learn is None:
                            steps_to_learn = iters
            else:
                if r >= -110:
                    if steps_to_learn is None:
                        steps_to_learn = iters
    except ValueError:
        print("Diverged")
        pass # Diverged weights

    while iters < max_iters: # Means it diverged at some point
        rewards.append(None)

    data = (args, rewards, steps_to_learn)
    file_name, file_num = utils.find_next_free_file("results", "pkl", results_dir)
    with open(file_name, "wb") as f:
        dill.dump(data, f)

    return rewards,steps_to_learn

def cc(fn, params, proc=10, keyworded=False):
    futures = []
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=proc) as executor:
        for i in tqdm(params, desc="Adding jobs"):
            if keyworded:
                future = [executor.submit(fn, **i)]
            else:
                future = [executor.submit(fn, *i)]
            futures += future
        pbar = tqdm(total=len(futures), desc="Job completion")
        while len(futures) > 0:
            count = [f.done() for f in futures].count(True)
            pbar.update(count)
            futures = [f for f in futures if not f.done()]
            time.sleep(1)

def gs_rbf(proc=10, results_directory="./results-rbf"):
    #def rbf_control(discount_factor, learning_rate, initial_value, num_pos,
    #        num_vel, behaviour_eps, target_eps, epoch, max_iters, test_iters):
    d = [1]
    lr = [2.5, 2, 1.5, 1, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01]
    #lr = [1, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01]
    iv = [0]
    np = [8]
    nv = [8]
    be = [0]
    te = [0]
    e = [None]
    mi = [3000]
    ti = [0]
    rd = [results_directory]
    indices = itertools.product(d,lr,iv,np,nv,be,te,e,mi,ti,rd)

    cc(rbf_control, indices, proc)

def lstd_rbf_control(discount_factor, initial_value, num_pos,
        num_vel, behaviour_eps, target_eps, update_freq, epoch, max_iters, test_iters,
        results_dir):
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
        global agent
        agent = LSTDAgent(
                action_space=action_space,
                discount_factor=discount_factor,
                #initial_value=initial_value,
                features=rbf,
                num_features=num_pos*num_vel
        )
        agent.set_behaviour_policy("%f-epsilon" % behaviour_eps)
        agent.set_target_policy("%f-epsilon" % target_eps)

        rewards = []
        steps_to_learn = None
        iters = 0
        while iters < max_iters:
            iters += 1
            r,s = agent.run_episode(e)
            #print("Iteration %d, Reward: %d" % (iters, r))
            rewards.append(r)
            if epoch is not None:
                if iters % epoch == 0:
                    print("Testing...")
                    rewards = agent.test(e, test_iters, render=False, processors=1)
                    if np.mean(rewards) >= -110:
                        if steps_to_learn is None:
                            steps_to_learn = iters
            else:
                if r >= -110:
                    if steps_to_learn is None:
                        steps_to_learn = iters
            if iters % update_freq == 0:
                #print("Updating Weights")
                agent.update_weights()

        while iters < max_iters: # Means it diverged at some point
            rewards.append(None)

        data = (args, rewards, steps_to_learn)
        file_name, file_num = utils.find_next_free_file("results", "pkl", results_dir)
        with open(file_name, "wb") as f:
            dill.dump(data, f)

        return rewards,steps_to_learn
    except Exception as e:
        #print(e)
        traceback.print_exc()

def gs_lstd_rbf(proc=10, results_directory="./results-lstd-rbf"):
    #def lstd_rbf_control(discount_factor, initial_value, num_pos,
    #        num_vel, behaviour_eps, target_eps, update_freq, epoch, max_iters, test_iters,
    #        results_dir):
    d = [1]
    iv = [0]
    np = [8]
    nv = [8]
    be = [0, 0.1, 0.2, 0.3, 0.5, 0.7]
    te = [0]
    uf = [1,5,10,20,50,100]
    e = [None]
    mi = [3000]
    ti = [0]
    rd = [results_directory]
    indices = itertools.product(d,iv,np,nv,be,te,uf,e,mi,ti,rd)

    futures = []
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=proc) as executor:
        for i in tqdm(indices, desc="Adding jobs"):
            print(i)
            future = [executor.submit(lstd_rbf_control, *i)]
            futures += future
        pbar = tqdm(total=len(futures), desc="Job completion")
        while len(futures) > 0:
            count = [f.done() for f in futures].count(True)
            pbar.update(count)
            futures = [f for f in futures if not f.done()]
            time.sleep(1)

def graph(file_names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def get_data(file_name):
        with open(file_name, 'rb') as f:
            x = dill.load(f)
            diverged = None in x[1]
            if diverged:
                x[1] = [a for a in x[1] if a is not None]
            if x[2] is None:
                return (x[0], file_name, x[1], np.inf, diverged)
            else:
                return (x[0], file_name, x[1], x[2], diverged)
    fig = plt.figure()
    for file_name in file_names:
        x = get_data(file_name)
        plt.plot(x[2])
    plt.savefig("mf.png")

def graph_dirs(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if type(directory) is str:
        directory = [directory]

    fig = plt.figure()
    for d in directory:
        file_names = [f for f in tqdm(os.listdir(d)) if os.path.isfile(os.path.join(d,f))]
        data = []
        for file_name in tqdm(file_names):
            with open(os.path.join(d,file_name), 'rb') as f:
                x = dill.load(f)
                diverged = None in x[1]
                if diverged:
                    print("Diverged")
                    raise NotImplementedError("Did not implement handling of data of different lengths")
                else:
                    data.append(x[1])
        data = np.array(data)
        mean = np.mean(data,0)
        std = np.std(data,0)
        plt.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.5)
        plt.plot(mean)
    plt.savefig("mf.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Boo")
    parser.add_argument("--parse-results",
            action="store_true")
    parser.add_argument("--grid-search",
            action="store_true")
    parser.add_argument("--graph",
            action="store_true")
    parser.add_argument("--model",
            type=str, default="rbf", help="Model to use (tabular|rbf)")
    parser.add_argument("--learning-rate",
            type=float, default=0.5, help="Learning rate")
    parser.add_argument("--discount",
            type=float, default=0.9, help="")
    parser.add_argument("--behaviour-epsilon",
            type=float, default=0.1, help="")
    parser.add_argument("--target-epsilon",
            type=float, default=0, help="")
    parser.add_argument("--update-freq",
            type=float, default=10, help="")
    parser.add_argument("--initial-value",
            type=float, default=0, help="")
    parser.add_argument("--num-pos",
            type=int, default=10, help="")
    parser.add_argument("--num-vel",
            type=int, default=10, help="")
    parser.add_argument("--epoch",
            type=int, default=None, help="Number of episodes to complete before testing")
    parser.add_argument("--max-iters",
            type=int, default=3000, help="Maximum number of episodes to run while learning")
    parser.add_argument("--test-iters",
            type=int, default=10, help="Number of episodes to run when testing")
    parser.add_argument("--trials",
            type=int, default=1,
            help="Number of trials to run with the given parameters")
    parser.add_argument("--results-dir",
            type=str, default="./results",
            help="Directory in which to place the pickled results")
    parser.add_argument("--results-dirs",
            type=str, default="./results", nargs='+',
            help="Directory in which to place the pickled results")
    parser.add_argument("--best-params-from",
            type=str, default=None,
            help="Look in the provided directory for the best set of parameters")
    args = parser.parse_args()

    if args.parse_results:
        parse_results(args.results_dir)
    elif args.grid_search:
        if args.model == "rbf":
            gs_rbf(results_directory=args.results_dir)
        elif args.model == "lstd-rbf":
            gs_lstd_rbf(results_directory=args.results_dir)
    elif args.graph:
        #graph(["./results-lstd-rbf/results-8.pkl",
        #    "./results-lstd-rbf/results-0.pkl"])
        #graph(["./results-lstd-rbf/results-8.pkl",
        #    "./results-lstd-rbf/results-0.pkl"])
        print(args.results_dirs)
        graph_dirs(args.results_dirs)
    else:
        if args.model == "tabular":
            params={"discount_factor":args.discount,
                    "learning_rate":args.learning_rate,
                    "initial_value":args.initial_value,
                    "num_pos":args.num_pos,
                    "num_vel":args.num_vel,
                    "behaviour_eps":args.behaviour_epsilon,
                    "target_eps":args.target_epsilon,
                    "epoch":args.epoch,
                    "test_iters":args.test_iters,
                    "max_iters":args.max_iters,
                    "results_dir":args.results_dir}
            fn = tabular_control
        elif args.model == "rbf":
            params={"discount_factor":args.discount,
                    "learning_rate":args.learning_rate,
                    "initial_value":args.initial_value,
                    "num_pos":args.num_pos,
                    "num_vel":args.num_vel,
                    "behaviour_eps":args.behaviour_epsilon,
                    "target_eps":args.target_epsilon,
                    "epoch":args.epoch,
                    "test_iters":args.test_iters,
                    "max_iters":args.max_iters,
                    "results_dir":args.results_dir}
            fn = rbf_control
        elif args.model == "lstd-rbf":
            params={"discount_factor":args.discount,
                    "initial_value":args.initial_value,
                    "num_pos":args.num_pos,
                    "num_vel":args.num_vel,
                    "behaviour_eps":args.behaviour_epsilon,
                    "target_eps":args.target_epsilon,
                    "update_freq":args.update_freq,
                    "epoch":args.epoch,
                    "test_iters":args.test_iters,
                    "max_iters":args.max_iters,
                    "results_dir":args.results_dir}
            fn = lstd_rbf_control
        else:
            print("Invalid model")
        if args.best_params_from is not None:
            p1, p2 = parse_results(args.best_params_from)
            params = p1
        if args.trials == 1:
            fn(**params)
        else:
            cc(fn, itertools.repeat(params, args.trials), keyworded=True)
        # --initial_value 0 --update-freq 1 --target-eps 0 --num-pos 8
        # --behaviour_eps 0 --max-iters 3000 --num-vel 8 --test-iters 0
        # --results-dir ./results-lstd-rbf --discount_factor 1

        # --initial-value 0 update-freq 20 --target-eps 0 --num-pos 8
        # --behaviour-eps 0.1 --max-iters 3000 --num-vel 8 --test-iters 0
        # --results-dir './results-lstd-rbf' --discount-factor 1

        # {'initial_value': 0, 'update_freq': 1, 'target_eps': 0, 'num_pos': 8,
        # 'epoch': None, 'behaviour_eps': 0, 'max_iters': 3000, 'num_vel': 8,
        # 'test_iters': 0, 'results_dir': './results-lstd-rbf',
        # 'discount_factor': 1}
