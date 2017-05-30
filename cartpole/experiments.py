import numpy as np
import gym
import itertools
import pandas
from pathos.multiprocessing import ProcessPool

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent
from learner.learner import Optimizer

import cartpole 
from cartpole import features
from cartpole import utils

"""
FrozenLake

Tabular gridsearch
LSTD gridsearch
"""

def exp1():
    print("Gridsearch")
    print("Environment: CartPole")
    print("Parameter space:")
    print("""
            \tDiscount factor 
            \tUpdate frequency
            \tBehaviour epsilon
            \tTarget epsilon
    """)
    discount_factors = ['1', '0.9', '0.8', '0.5']
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
        env_name = 'CartPole-v0'
        e = gym.make(env_name)

        action_space = np.array([0,1])
        agent = LSTDAgent(
                action_space=action_space,
                num_features=cartpole.features.IDENTITY_NUM_FEATURES,
                discount_factor=gamma,
                features=cartpole.features.identity2,
                use_importance_sampling=False,
                use_traces=False,
                sigma=None,
                trace_factor=None,
        )
        agent.set_behaviour_policy("%s-epsilon" % eps_b)
        agent.set_target_policy("%s-epsilon" % eps_t)

        iters = 0
        while True:
            iters += 1
            agent.run_episode(e)
            if iters % upd_freq == 0:
                agent.update_weights()
                rewards = agent.test(e, 100, render=False)
                print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
                if np.mean(rewards) >= 190:
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
    print(output)
    print(data)
    return data

def run_all():
    gridsearch_results = exp1()

if __name__ == "__main__":
    run_all()
