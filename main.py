import torch
import gym
import numpy as np
import pprint
from tqdm import tqdm

from agent.discrete_agent import TabularAgent
from agent.linear_agent import LinearAgent
from agent.lstd_agent import LSTDAgent

import frozenlake
#from frozenlake import exp1
from frozenlake import exp2
from frozenlake import exp3
from frozenlake import graph

import frozenlake8x8
import frozenlake8x8.features
import frozenlake8x8.utils

import experiments
import utils

# When taking an action, there's an equal probability of moving in any direction that isn't the opposite direction
# e.g. If you choose up, there's a 1/3 chance of going up, 1/3 of going left, 1/3 of going right

def control():
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
            #frozenlake.utils.print_policy(agent)
            if np.mean(rewards) >= 0.78:
                break

def policy_evaluation():
    env_name = 'FrozenLake-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = TabularAgent(action_space=action_space, discount_factor=0.99, learning_rate=0.1, optimizer=Optimizer.RMS_PROP)
    agent.set_target_policy(frozenlake.utils.optimal_policy)
    agent.set_behaviour_policy(frozenlake.utils.optimal_policy)

    iters = 0
    while True:
        iters += 1

        agent.run_episode(e)
        if iters % 500 == 0:
            weight_diff = agent.get_weight_change()
            if weight_diff < 0.01:
                break
            agent.reset_weight_change()
            print("Iteration %d\t Weight Change: %f" % (iters, weight_diff))

def linear_control():
    env_name = 'FrozenLake-v0'
    e = gym.make(env_name)

    def features(x):
        output = np.zeros(16)
        output[x] = 1
        return output

    action_space = np.array([0,1,2,3])
    agent = LinearAgent(action_space=action_space, discount_factor=0.99,
            learning_rate=0.1, num_features=16, features=features)

    iters = 0
    while True:
        iters += 1
        agent.run_episode(e)
        if iters % 500 == 0:
            rewards = agent.test(e, 100)
            print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
            if np.mean(rewards) >= 0.78:
                break

def lstd_control():
    env_name = 'FrozenLake-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = LSTDAgent(
            action_space=action_space,
            num_features=frozenlake.features.ONE_HOT_NUM_FEATURES,
            discount_factor=0.99,
            features=frozenlake.features.one_hot,
            use_importance_sampling=False,
            sigma=1
    )

    iters = 0
    while True:
        iters += 1
        agent.run_episode(e)
        if iters % 500 == 0:
            agent.update_weights()
            rewards = agent.test(e, 100)
            print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
            frozenlake.utils.print_policy(agent, f=frozenlake.features.one_hot)
            frozenlake.utils.print_values(agent, f=frozenlake.features.one_hot)
            if np.mean(rewards) >= 0.78:
                break

def lstd_policy_evaluation():
    env_name = 'FrozenLake-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = LSTDAgent(
            action_space=action_space,
            num_features=frozenlake.features.ONE_HOT_NUM_FEATURES,
            discount_factor=0.9,
            features=frozenlake.features.one_hot
    )
    agent.set_target_policy(frozenlake.utils.optimal_policy)
    agent.set_behaviour_policy(frozenlake.utils.optimal_policy)

    iters = 0
    while True:
        iters += 1

        agent.run_episode(e)
        if iters % 500 == 0:
            agent.update_weights()
            weight_diff = agent.get_weight_change()
            print("Iteration %d\t Weight Change: %f" % (iters, weight_diff))
            #if weight_diff < 0.0001:
            #    break
            agent.reset_weight_change()
            frozenlake.utils.print_values(agent, f=frozenlake.features.one_hot)

def lstd_tile_coding_control():
    env_name = 'FrozenLake-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = LSTDAgent(
            action_space=action_space,
            num_features=frozenlake.features.TILE_CODING_NUM_FEATURES,
            discount_factor=0.99,
            features=frozenlake.features.tile_coding)

    iters = 0
    while True:
        iters += 1
        agent.run_episode(e)
        if iters % 500 == 0:
            agent.update_weights()
            rewards = agent.test(e, 100)
            print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
            frozenlake.utils.print_policy(agent, f=frozenlake.features.tile_coding)
            frozenlake.utils.print_values(agent, f=frozenlake.features.tile_coding)
            if np.mean(rewards) >= 0.78:
                break

def lstd_trace_control():
    env_name = 'FrozenLake-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = LSTDAgent(
            action_space=action_space,
            num_features=frozenlake.features.ONE_HOT_NUM_FEATURES,
            discount_factor=0.99,
            features=frozenlake.features.one_hot,
            use_traces=True,
            trace_factor=0.5
    )

    iters = 0
    while True:
        iters += 1
        agent.run_episode(e)
        if iters % 500 == 0:
            agent.update_weights()
            rewards = agent.test(e, 100)
            print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
            frozenlake.utils.print_policy(agent, f=frozenlake.features.one_hot)
            frozenlake.utils.print_values(agent, f=frozenlake.features.one_hot)
            if np.mean(rewards) >= 0.78:
                break

def lstd_policy_evaluation_8x8():
    env_name = 'FrozenLake8x8-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = LSTDAgent(
            action_space=action_space,
            num_features=frozenlake8x8.features.ONE_HOT_NUM_FEATURES,
            discount_factor=0.9,
            features=frozenlake8x8.features.one_hot
    )
    agent.set_target_policy(frozenlake8x8.utils.optimal_policy)
    agent.set_behaviour_policy(frozenlake8x8.utils.optimal_policy)

    iters = 0
    while True:
        iters += 1

        agent.run_episode(e)
        if iters % 500 == 0:
            agent.update_weights()
            weight_diff = agent.get_weight_change()
            print("Iteration %d\t Weight Change: %f" % (iters, weight_diff))
            if weight_diff < 0.0001:
                break
            agent.reset_weight_change()
            frozenlake8x8.utils.print_values(agent, f=frozenlake8x8.features.one_hot)

def lstd_control_8x8():
    env_name = 'FrozenLake8x8-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = LSTDAgent(
            action_space=action_space,
            num_features=frozenlake8x8.features.ONE_HOT_NUM_FEATURES,
            discount_factor=0.99,
            features=frozenlake8x8.features.one_hot,
            use_importance_sampling=False,
            use_traces=True,
            trace_factor=0.5,
            sigma=0.5
    )

    iters = 0
    
    # Wait until receiving the first reward before we start updating the weights
    r = 0
    while r == 0:
        iters += 1
        r,s = agent.run_episode(e)
    agent.update_weights()
    frozenlake8x8.utils.print_policy(agent, f=frozenlake8x8.features.one_hot)
    frozenlake8x8.utils.print_values(agent, f=frozenlake8x8.features.one_hot)
    print("First reward encountered on iteration %d" % iters)

    while True:
        iters += 1
        agent.run_episode(e)
        if iters % 500 == 0:
            agent.update_weights()
            rewards = agent.test(e, 100)
            print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
            frozenlake8x8.utils.print_policy(agent, f=frozenlake8x8.features.one_hot)
            frozenlake8x8.utils.print_values(agent, f=frozenlake8x8.features.one_hot)
            if np.mean(rewards) >= 0.99:
                break

def run_all(proc=20):
    # SGD
    experiments.run1(exp2, n=1, proc=proc)
    for _ in range(1):
        experiments.run2(exp2, n=1, m=100, proc=proc)
    experiments.run3(exp2, n=15, proc=proc)
    exp2.plot_best()
    exp2.plot_final_rewards()

    # LSTD
    experiments.run1(exp3, n=1, proc=proc)
    for _ in range(100):
        experiments.run2(exp3, n=1, m=100, proc=proc)
    experiments.run3(exp3, n=15, proc=proc)
    exp2.plot_best()
    exp3.plot_final_rewards()

    graph.graph_sgd()
    graph.graph_lstd()
    graph.graph_all()

if __name__ == "__main__":
    utils.set_results_directory("/home/ml/hhuang63/results/final")
    run_all(proc=15)
