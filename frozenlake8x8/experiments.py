import numpy as np
import gym
import itertools

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent
from learner.learner import Optimizer

import frozenlake
from frozenlake import features
from frozenlake import utils

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
    discount_factors = [1, 0.99, 0.9]
    learning_rates = [0.1, 0.01, 0.001]
    optimizers = [Optimizer.RMS_PROP, Optimizer.NONE]
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

    highest_val = 0
    best_params = None
    for (gamma, alpha, op) in itertools.product(discount_factors, learning_rates, optimizers):
        print("Running discount factor %f, learning rate %f, optimizer %s" % (gamma, alpha, op))
        val = 0
        for _ in range(10):
            val += run_trial(gamma, alpha, op)
        if highest_val < val:
            highest_val = val
            best_params = (gamma, alpha, op)

def exp2():
    print("Environment: FrozenLake4x4")
    print("Params:")
    print("""
            \tDiscount factor: 
            \tLearning rate:
            \tOptimizer:
    """)

    env_name = 'FrozenLake-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = TabularAgent(action_space=action_space, discount_factor=0.99, learning_rate=0.1, optimizer=Optimizer.RMS_PROP)

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

def run_all():
    exp1()

if __name__ == "__main__":
    run_all()
