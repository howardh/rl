import numpy as np
import gym
import itertools
import pandas
from pathos.multiprocessing import ProcessPool

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent
from learner.learner import Optimizer

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

        iters = 0
        while True:
            iters += 1
            agent.run_episode(e)
            if iters % 500 == 0:
                rewards = agent.test(e, 100)
                print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
                if np.mean(rewards) >= 0.78:
                    break
            if iters >= 100000:
                break
        return iters

    def foo(i):
        g,a,o = indices[i]
        g = float(g)
        a = float(a)
        o = eval(o)
        results = [run_trial(g,a,o) for _ in range(10)]
        return np.mean(results)
    pool = ProcessPool(processes=3)
    output = pool.map(foo, range(len(indices)))
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

    iters = 0
    while True:
        iters += 1
        agent.run_episode(e)
        if iters % 500 == 0:
            rewards = agent.test(e, 100)
            print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
            frozenlake.utils.print_policy(agent)
            if np.mean(rewards) >= 0.78:
                break

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

        iters = 0
        while True:
            iters += 1
            agent.run_episode(e)
            if iters % upd_freq == 0:
                agent.update_weights()
                rewards = agent.test(e, 100)
                print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
                if np.mean(rewards) >= 0.78:
                    break
            if iters > 100000:
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
    output = pool.map(foo, range(len(indices)))
    for i,x in enumerate(output):
        g,n,eb,et = indices[i]
        data.loc[g,n,eb,et] = x
    print(data)
    return data

def run_all():
    #gridsearch_results = exp1()
    exp2()

if __name__ == "__main__":
    run_all()
