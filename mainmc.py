import gym
import numpy as np
import datetime
import dill
import os
import itertools
from tqdm import tqdm
import time
import traceback
import threading

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent
from agent.rbf_agent import RBFAgent
from agent.rbf_agent import RBFTracesAgent

import mountaincar 
import mountaincar.features
import mountaincar.utils

from mountaincar import exp3
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
        tqdm.write("... %d\r" % iters, end='')
        if r != -200:
            tqdm.write("Iteration: %d\t Reward: %d"%(iters, r))
        if epoch is not None and iters % epoch == 0:
            tqdm.write("Testing...")
            #print(agent.learner.weights.transpose())
            rewards = agent.test(e, test_iters, render=False, processors=1)
            tqdm.write("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
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
            data = x[1]
            if diverged:
                data = [a for a in x[1] if a is not None]
            if x[2] is None:
                results.append((x[0], file_name, np.mean(data), np.inf, diverged))
            else:
                results.append((x[0], file_name, np.mean(data), x[2], diverged))
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
                    tqdm.write("Testing...")
                    rewards = agent.test(e, test_iters, render=False, processors=1)
                    if np.mean(rewards) >= -110:
                        if steps_to_learn is None:
                            steps_to_learn = iters
            else:
                if r >= -110:
                    if steps_to_learn is None:
                        steps_to_learn = iters
    except ValueError:
        tqdm.write("Diverged")
        pass # Diverged weights

    while iters < max_iters: # Means it diverged at some point
        rewards.append(None)

    data = (args, rewards, steps_to_learn)
    file_name, file_num = utils.find_next_free_file("results", "pkl", results_dir)
    with open(file_name, "wb") as f:
        dill.dump(data, f)

    return rewards,steps_to_learn

def rbft_control(discount_factor, learning_rate, trace_factor, initial_value, num_pos,
        num_vel, behaviour_eps, target_eps, epoch, max_iters, test_iters,
        results_dir):
    args = locals()
    #print('a %s' % threading.current_thread())
    env_name = 'MountainCar-v0'
    e = gym.make(env_name)
    start_time = datetime.datetime.now()

    #print('b %s' % threading.current_thread())
    action_space = np.array([0,1,2])
    obs_space = np.array([[-1.2, .6], [-0.07, 0.07]])
    def norm(x):
        return np.array([(s-r[0])/(r[1]-r[0]) for s,r in zip(x, obs_space)])
    #print('c %s' % threading.current_thread())
    agent = RBFTracesAgent(
            action_space=action_space,
            observation_space=np.array([[0,1],[0,1]]),
            discount_factor=discount_factor,
            learning_rate=learning_rate,
            initial_value=initial_value,
            features=norm,
            trace_factor=trace_factor
    )
    #print('d %s' % threading.current_thread())
    agent.set_behaviour_policy("%f-epsilon" % behaviour_eps)
    agent.set_target_policy("%f-epsilon" % target_eps)

    rewards = []
    steps_to_learn = None
    iters = 0
    #print('e %s' % threading.current_thread())
    try:
        while iters < max_iters:
            iters += 1
            r,s = agent.run_episode(e)
            rewards.append(r)
            if epoch is not None:
                if iters % epoch == 0:
                    tqdm.write("Testing...")
                    rewards = agent.test(e, test_iters, render=False, processors=1)
                    if np.mean(rewards) >= -110:
                        if steps_to_learn is None:
                            steps_to_learn = iters
            else:
                if r >= -110:
                    if steps_to_learn is None:
                        steps_to_learn = iters
    except ValueError:
        tqdm.write("Diverged")
        pass # Diverged weights
    except KeyboardInterrupt:
        print("kbi")

    #print('f %s' % threading.current_thread())
    while iters < max_iters: # Means it diverged at some point
        iters += 1
        rewards.append(None)

    #print('g %s' % threading.current_thread())
    data = (args, rewards, steps_to_learn)
    file_name, file_num = utils.find_next_free_file("results", "pkl", results_dir)
    with open(file_name, "wb") as f:
        dill.dump(data, f)

    del agent
    del rewards

    #print('h %s' % threading.current_thread())
    #return rewards,steps_to_learn

def gs_rbf(proc=10, results_directory="./results-rbf"):
    #def rbf_control(discount_factor, learning_rate, initial_value, num_pos,
    #        num_vel, behaviour_eps, target_eps, epoch, max_iters, test_iters):
    d = [1]
    lr = [2.5, 2, 1.5, 1, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01]
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

    utils.cc(rbf_control, indices, proc)

def gs_rbft(proc=10, results_directory="./results-rbft"):
    #def rbf_control(discount_factor, learning_rate, initial_value, num_pos,
    #        num_vel, behaviour_eps, target_eps, epoch, max_iters, test_iters):
    d = [1]
    lr = [2.5, 2, 1.5, 1, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01]
    l = [0,0.2,0.5,0.7,0.9,1]
    iv = [0]
    np = [8]
    nv = [8]
    be = [0]
    te = [0]
    e = [None]
    mi = [3000]
    ti = [0]
    rd = [results_directory]
    indices = itertools.product(d,lr,l,iv,np,nv,be,te,e,mi,ti,rd)

    utils.cc(rbft_control, indices, proc)

def lstd_rbft_control(discount_factor, initial_value, num_pos,
        num_vel, behaviour_eps, target_eps, trace_factor, update_freq, epoch, max_iters, test_iters,
        results_dir):
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
                trace_factor=trace_factor
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
                    tqdm.write("Testing...")
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
            iters += 1
            rewards.append(None)

        data = (args, rewards, steps_to_learn)
        file_name, file_num = utils.find_next_free_file("results", "pkl", results_dir)
        with open(file_name, "wb") as f:
            dill.dump(data, f)

        return rewards,steps_to_learn
    except Exception as e:
        #print(e)
        traceback.print_exc()

def lstd_rbf_control(discount_factor, initial_value, num_pos,
        num_vel, behaviour_eps, target_eps, update_freq, epoch, max_iters, test_iters,
        results_dir, trace_factor=0):
    lstd_rbft_control(discount_factor, initial_value, num_pos,
        num_vel, behaviour_eps, target_eps, trace_factor, update_freq, epoch, max_iters, test_iters,
        results_dir)

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

    params = itertools.product(d,iv,np,nv,be,te,uf,e,mi,ti,rd)
    utils.cc(lstd_rbf_control, params, proc, keyworded=False)

def gs_lstd_rbft(proc=10, results_directory="./results-lstd-rbft"):
    #def lstd_rbft_control(discount_factor, initial_value, num_pos,
    #        num_vel, behaviour_eps, target_eps, trace_factor, update_freq, epoch, max_iters, test_iters,
    #        results_dir):
    d = [1]
    iv = [0]
    np = [8]
    nv = [8]
    be = [0, 0.1, 0.2, 0.3, 0.5, 0.7]
    te = [0]
    tf = [0,0.25,0.5,0.75,1]
    uf = [1,5,10,20,50,100]
    e = [None]
    mi = [3000]
    ti = [0]
    rd = [results_directory]

    indices = itertools.product(d,iv,np,nv,be,te,tf,uf,e,mi,ti,rd)
    utils.cc(lstd_rbft_control, indices, proc=proc, keyworded=False)

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

def graph_dirs(directory, labels=None, output="graph.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if type(directory) is str:
        directory = [directory]
    if labels is None:
        labels = directory

    # Load data
    data_dict = dict()
    for d in directory:
        file_names = [f for f in tqdm(os.listdir(d)) if os.path.isfile(os.path.join(d,f))]
        data = []
        for file_name in tqdm(file_names):
            with open(os.path.join(d,file_name), 'rb') as f:
                try:
                    x = dill.load(f)
                except Exception:
                    tqdm.write(file_name)
                    return
                diverged = None in x[1]
                if diverged:
                    tqdm.write("Diverged %s" % file_name)
                    #raise NotImplementedError("Did not implement handling of data of different lengths")
                    continue
                else:
                    data.append(x[1])
        data = np.array(data)
        mean = np.mean(data,0)
        std = np.std(data,0)
        data_dict[d] = (mean,std)

    # Plot figure
    fig = plt.figure()
    for d,l in zip(directory,labels):
        mean = data_dict[d][0]
        std = data_dict[d][1]
        plt.fill_between(range(len(mean)), mean-std/2, mean+std/2, alpha=0.5)
        plt.plot(mean, label=l)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.savefig(output)

    fig = plt.figure()
    for d,l in zip(directory,labels):
        mean = data_dict[d][0][:500]
        std = data_dict[d][1][:500]
        plt.fill_between(range(len(mean)), mean-std/2, mean+std/2, alpha=0.5)
        plt.plot(mean, label=l)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.savefig("2"+output)

    return data_dict

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Boo")
    parser.add_argument("--parse-results",
            action="store_true")
    parser.add_argument("--grid-search",
            action="store_true")
    parser.add_argument("--graph",
            action="store_true")
    parser.add_argument("--exps",
            action="store_true")
    parser.add_argument("--output-file",
            type=str, default="graph.png",
            help="Name of the file in which to store the graph")
    parser.add_argument("--model",
            type=str, default="rbf",
            help="Model to use (tabular|rbf|rbft|lstd-rbf|lstd-rbft)")
    parser.add_argument("--learning-rate",
            type=float, default=0.5, help="Learning rate")
    parser.add_argument("--discount",
            type=float, default=0.9, help="")
    parser.add_argument("--trace-factor",
            type=float, default=0, help="Trace factor (lambda)")
    parser.add_argument("--behaviour-epsilon",
            type=float, default=0.1, help="")
    parser.add_argument("--target-epsilon",
            type=float, default=0, help="")
    parser.add_argument("--update-freq",
            type=float, default=10, help="")
    parser.add_argument("--initial-value",
            type=float, default=0, help="")
    parser.add_argument("--num-pos",
            type=int, default=8, help="")
    parser.add_argument("--num-vel",
            type=int, default=8, help="")
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
    parser.add_argument("--threads",
            type=int, default=10,
            help="Maximum number of threads to use")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.parse_results:
        parse_results(args.results_dir)
    elif args.grid_search:
        if args.model == "rbf":
            gs_rbf(results_directory=args.results_dir)
        elif args.model == "rbft":
            gs_rbft(results_directory=args.results_dir)
        elif args.model == "lstd-rbf":
            gs_lstd_rbf(results_directory=args.results_dir)
        elif args.model == "lstd-rbft":
            gs_lstd_rbft(results_directory=args.results_dir)
    elif args.graph:
        #graph(["./results-lstd-rbf/results-8.pkl",
        #    "./results-lstd-rbf/results-0.pkl"])
        #graph(["./results-lstd-rbf/results-8.pkl",
        #    "./results-lstd-rbf/results-0.pkl"])
        print(args.results_dirs)
        graph_dirs(args.results_dirs,output=args.output_file)
    elif args.exps:
        # The old school way
        #exp3.run3(proc=3)
        exp3.parse_results3('/NOBACKUP/hhuang63/results3/2018-03-20_19-17-32/mountaincar.exp3/part3')
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
        elif args.model == "rbft":
            params={"discount_factor":args.discount,
                    "learning_rate":args.learning_rate,
                    "trace_factor": args.trace_factor,
                    "initial_value":args.initial_value,
                    "num_pos":args.num_pos,
                    "num_vel":args.num_vel,
                    "behaviour_eps":args.behaviour_epsilon,
                    "target_eps":args.target_epsilon,
                    "epoch":args.epoch,
                    "test_iters":args.test_iters,
                    "max_iters":args.max_iters,
                    "results_dir":args.results_dir}
            fn = rbft_control
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
        elif args.model == "lstd-rbft":
            params={"discount_factor":args.discount,
                    "initial_value":args.initial_value,
                    "num_pos":args.num_pos,
                    "num_vel":args.num_vel,
                    "behaviour_eps":args.behaviour_epsilon,
                    "target_eps":args.target_epsilon,
                    "trace_factor":args.trace_factor,
                    "update_freq":args.update_freq,
                    "epoch":args.epoch,
                    "test_iters":args.test_iters,
                    "max_iters":args.max_iters,
                    "results_dir":args.results_dir}
            fn = lstd_rbft_control
        else:
            print("Invalid model")
        if args.best_params_from is not None:
            p1, p2 = parse_results(args.best_params_from)
            params = p1
            params['results_dir'] = args.results_dir
        if args.trials == 1:
            fn(**params)
        else:
            utils.cc(fn, itertools.repeat(params, args.trials), proc=args.threads, keyworded=True)
