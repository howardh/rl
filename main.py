import gym
import numpy as np

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent
from learner.learner import TabularLearner
from learner.learner import Optimizer

from frozenlake import features
from frozenlake import utils

# When taking an action, there's an equal probability of moving in any direction that isn't the opposite direction
# e.g. If you choose up, there's a 1/3 chance of going up, 1/3 of going left, 1/3 of going right


def control():
    env_name = 'FrozenLake-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = TabularAgent(action_space=action_space, discount_factor=0.9, learning_rate=0.1, optimizer=Optimizer.RMS_PROP)

    iters = 0
    while True:
        iters += 1
        agent.run_episode(e)
        if iters % 500 == 0:
            rewards = agent.test(e, 100)
            print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
            utils.print_policy(agent)
            #if np.mean(rewards) >= 0.78:
            #    break

def policy_evaluation():
    env_name = 'FrozenLake-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = TabularAgent(action_space=action_space, discount_factor=0.9, learning_rate=0.1, optimizer=Optimizer.RMS_PROP)
    agent.set_target_policy(frozen_lake_policy)
    agent.set_behaviour_policy(frozen_lake_policy)

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

def lstd_control():
    env_name = 'FrozenLake-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = LSTDAgent(
            action_space=action_space,
            num_features=features.ONE_HOT_NUM_FEATURES,
            discount_factor=0.99,
            features=features.one_hot
    )

    iters = 0
    while True:
        iters += 1
        agent.run_episode(e)
        if iters % 500 == 0:
            agent.update_weights()
            rewards = agent.test(e, 100)
            print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
            utils.print_policy(agent, f=features.one_hot)
            utils.print_values(agent, f=features.one_hot)
            if np.mean(rewards) >= 0.78:
                break

def lstd_policy_evaluation():
    env_name = 'FrozenLake-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = LSTDAgent(
            action_space=action_space,
            num_features=features.ONE_HOT_NUM_FEATURES,
            discount_factor=0.9,
            features=features.one_hot
    )
    agent.set_target_policy(frozen_lake_policy)
    agent.set_behaviour_policy(frozen_lake_policy)

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
            utils.print_values(agent, f=features.one_hot)

def lstd_tile_coding_control():
    env_name = 'FrozenLake-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = LSTDAgent(
            action_space=action_space,
            num_features=features.TILE_CODING_NUM_FEATURES,
            discount_factor=0.99,
            features=features.tile_coding)

    iters = 0
    while True:
        iters += 1
        agent.run_episode(e)
        if iters % 500 == 0:
            agent.update_weights()
            rewards = agent.test(e, 100)
            print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
            utils.print_policy(agent, f=features.tile_coding)
            utils.print_values(agent, f=features.tile_coding)
            if np.mean(rewards) >= 0.78:
                break

def lstd_trace_control():
    env_name = 'FrozenLake-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = LSTDAgent(
            action_space=action_space,
            num_features=features.ONE_HOT_NUM_FEATURES,
            discount_factor=0.99,
            features=features.one_hot,
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
            utils.print_policy(agent, f=features.one_hot)
            utils.print_values(agent, f=features.one_hot)
            if np.mean(rewards) >= 0.78:
                break

if __name__ == "__main__":
    #control()
    #policy_evaluation()
    #lstd_policy_evaluation()
    #lstd_control()
    #lstd_tile_coding_control()
    lstd_trace_control()
