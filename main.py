import gym
import numpy as np
import pprint

from agent.discrete_agent import TabularAgent
from agent.linear_agent import LinearAgent
from agent.lstd_agent import LSTDAgent
from learner.learner import Optimizer

import frozenlake
from frozenlake import features
from frozenlake import utils
from frozenlake import experiments
from frozenlake import exp1
from frozenlake import exp2
from frozenlake import exp3
from frozenlake import exp4
from frozenlake import graph

import frozenlake8x8
import frozenlake8x8.features
import frozenlake8x8.utils

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

if __name__ == "__main__":
    #utils.set_results_directory("/NOBACKUP/hhuang63/results3/final")
    utils.set_results_directory("/NOBACKUP/hhuang63/results3/test")
    #data = exp3.parse_results()
    #exp1.run(n=2,proc=10)
    exp1.run2(n=2)
    #exp3.run2()
    #graph.graph_all()

