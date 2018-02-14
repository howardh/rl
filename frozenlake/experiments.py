import numpy as np
import gym
import itertools
import pandas
from pathos.multiprocessing import ProcessPool
from pathos.threading import ThreadPool
import multiprocessing
import dill
import csv
import os
from tqdm import tqdm

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent

import frozenlake
from frozenlake import features
from frozenlake import utils

"""
FrozenLake

Tabular gridsearch
LSTD gridsearch
"""

def exp1():
    print("Gridsearch")
    print("Environment: FrozenLake4x4")
    print("Parameter space:")
    print("""
            \tDiscount factor: 
            \tLearning rate:
            \tOptimizer:
    """)
    print("Determines the best combination of parameters by comparing the performance over time of several algorithms.")
    discount_factors = ['1', '0.99', '0.9']
    learning_rates = ['0.1', '0.01', '0.001']
    optimizers = ['Optimizer.RMS_PROP', 'Optimizer.NONE']
    indices = pandas.MultiIndex.from_product(
            [discount_factors, learning_rates, optimizers],
            names=["Discount Factor", "Learning Rate", "Optimizer"])
    data = pandas.DataFrame(
            np.zeros([len(discount_factors)*len(learning_rates)*len(optimizers),1]),
            index=indices)

    def run_trial(gamma, alpha, op):
        env_name = 'FrozenLake-v0'
        e = gym.make(env_name)

        action_space = np.array([0,1,2,3])
        agent = TabularAgent(
                action_space=action_space,
                discount_factor=gamma,
                learning_rate=alpha,
                optimizer=op)

        for iters in range(1,10000):
            agent.run_episode(e)
            if iters % 500 == 0:
                rewards = agent.test(e, 100, max_steps=1000)
                if np.mean(rewards) >= 0.78:
                    break
        return iters

    def foo(i):
        g,a,o = indices[i]
        g = float(g)
        a = float(a)
        o = eval(o)
        results = [run_trial(g,a,o) for _ in range(10)]
        print(indices[i])
        print(results)
        return np.mean(results)
    pool = ProcessPool(3)
    output = pool.imap(foo, range(len(indices)))
    for i,x in enumerate(output):
        g,a,o = indices[i]
        data.loc[g,a,o] = x
    print(data)
    return data

def exp1_2():
    print("Environment: FrozenLake4x4")
    print("Params:")
    print("""
            \tDiscount factor: 1
            \tLearning rate: 0.1
            \tOptimizer: RMS Prop
    """)

    env_name = 'FrozenLake-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = TabularAgent(action_space=action_space, discount_factor=1, learning_rate=0.1, optimizer=Optimizer.RMS_PROP)

    file_num = 0
    while os.path.isfile("data/%d.csv" % file_num):
        file_num += 1
    with open('data/%d.csv' % file_num, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        iters = 0
        all_rewards = []
        while iters < 100000:
            iters += 1
            agent.run_episode(e)
            if iters % 500 == 0:
                print(iters)
                rewards = agent.test(e, 100, max_steps=1000)
                all_rewards.append(rewards)
                csvwriter.writerow(rewards)
                csvfile.flush()
                print(rewards)
    return data

def exp2():
    print("Gridsearch")
    print("Environment: FrozenLake4x4")
    print("Parameter space:")
    print("""
            \tDiscount factor 
            \tUpdate frequency
            \tBehaviour epsilon
            \tTarget epsilon
    """)
    discount_factors = ['1', '0.99', '0.9']
    update_frequencies = ['100', '300', '500', '1000']
    behaviour_epsilon = ['0', '0.05', '0.1', '0.2', '0.5', '1']
    target_epsilon = ['0', '0.01', '0.05', '0.1']
    indices = pandas.MultiIndex.from_product(
            [discount_factors, update_frequencies, behaviour_epsilon, target_epsilon],
            names=["Discount Factor", "Update Frequency", "Behaviour Epsilon", "Target Epsilon"])
    data = pandas.DataFrame(
            np.zeros([len(discount_factors)*len(update_frequencies)*len(behaviour_epsilon)*len(target_epsilon),1]),
            index=indices)

    def run_trial(gamma, upd_freq, eps_b, eps_t):
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
        agent.set_behaviour_policy("%s-epsilon" % eps_b)
        agent.set_target_policy("%s-epsilon" % eps_t)

        for iters in range(1,1000):
            agent.run_episode(e)
            if iters % upd_freq == 0:
                agent.update_weights()
                rewards = agent.test(e, 100)
                if np.mean(rewards) >= 0.78:
                    break
        return iters

    def foo(i):
        g,n,eb,et = indices[i]
        g = float(g)
        n = int(n)
        eb = float(eb)
        et = float(et)
        results = [run_trial(g,n,eb,et) for _ in range(10)]
        return np.mean(results)
    pool = ProcessPool(processes=3)
    output = pool.imap(foo, range(len(indices)))
    for i,x in enumerate(output):
        g,n,eb,et = indices[i]
        data.loc[g,n,eb,et] = x
    print(data)
    return data

def run_all():
    #gridsearch_results = exp1()
    #gridsearch_results.to_csv("gridsearch.csv")
    exp1_2()
    #exp2()

if __name__ == "__main__":
    run_all()
