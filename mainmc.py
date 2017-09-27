import gym
import numpy as np
import datetime

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent

import mountaincar 
import mountaincar.features
import mountaincar.utils

#from mountaincar import exp1
import utils

def tabular_control():
    env_name = 'MountainCar-v0'
    e = gym.make(env_name)
    start_time = datetime.datetime.now()

    action_space = np.array([0,1,2])
    agent = TabularAgent(
            action_space=action_space,
            features=mountaincar.features.one_hot,
            discount_factor=0.95,
            learning_rate=0.05
    )
    #agent.set_behaviour_policy("1.0-epsilon")
    agent.set_behaviour_policy(mountaincar.utils.less_optimal_policy2)
    #agent.set_behaviour_policy(utils.less_optimal_policy)
    agent.set_target_policy("0.01-epsilon")

    iters = 0
    #for _ in range(100):
    while True:
        iters += 1
        r,s = agent.run_episode(e)
        if r != -200:
            break
        if iters % 100 == 0:
            print("...")
    print("Yay!")
    #agent.set_behaviour_policy("0.1-epsilon")
    while True:
        iters += 1
        r,s = agent.run_episode(e)
        #if r != -200:
        #    print("%d %d" % (iters, r))
        if iters % 100 == 0:
            #agent.update_weights()
            print("Testing...")
            #print(agent.learner.weights.transpose())
            rewards = agent.test(e, 100, render=True, processors=1)
            print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
            if np.mean(rewards) >= -110:
                break
        if iters > 3000:
            break

def lstd_control():
    env_name = 'MountainCar-v0'
    e = gym.make(env_name)
    start_time = datetime.datetime.now()

    action_space = np.array([0,1,2])
    #agent = LSTDAgent(
    #        action_space=action_space,
    #        num_features=mountaincar.features.ONE_HOT_NUM_FEATURES,
    #        features=mountaincar.features.one_hot,
    #        #num_features=mountaincar.features.IDENTITY_NUM_FEATURES,
    #        #features=mountaincar.features.identity,
    #        discount_factor=1,
    #        use_importance_sampling=False,
    #        use_traces=False,
    #        sigma=None,
    #        trace_factor=None,
    #)
    agent = TabularAgent(
            action_space=action_space,
            features=mountaincar.features.one_hot,
            discount_factor=1,
            learning_rate=0.05
    )
    #agent.set_behaviour_policy("1.0-epsilon")
    agent.set_behaviour_policy(mountaincar.utils.less_optimal_policy2)
    #agent.set_behaviour_policy(utils.less_optimal_policy)
    agent.set_target_policy("0.1-epsilon")

    iters = 0
    #for _ in range(100):
    while True:
        iters += 1
        r,s = agent.run_episode(e)
        if r != -200:
            break
        if iters % 100 == 0:
            print("...")
    print("Yay!")
    #agent.set_behaviour_policy("0.1-epsilon")
    while True:
        iters += 1
        r,s = agent.run_episode(e)
        #if r != -200:
        #    print("%d %d" % (iters, r))
        if iters % 100 == 0:
            #agent.update_weights()
            print("Testing...")
            #print(agent.learner.weights.transpose())
            rewards = agent.test(e, 100, render=False, processors=3)
            print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
            if np.mean(rewards) >= -110:
                break
        if iters > 3000:
            break

def lstd_control_steps():
    env_name = 'CartPole-v0'
    e = gym.make(env_name)
    start_time = datetime.datetime.now()

    action_space = np.array([0,1])
    agent = LSTDAgent(
            action_space=action_space,
            num_features=cartpole.features.IDENTITY_NUM_FEATURES,
            discount_factor=0.99,
            features=cartpole.features.identity,
            use_importance_sampling=False,
            use_traces=False,
            sigma=0,
            trace_factor=0.5,
    )
    agent.set_behaviour_policy(utils.optimal_policy)
    agent.set_target_policy("0-epsilon")

    steps = 0
    while True:
        steps += 1
        agent.run_step(e)
        if steps % 5000 == 0:
            agent.update_weights()
            rewards = agent.test(e, 100, render=False, processors=3)
            print("Steps %d\t Rewards: %f" % (steps, np.mean(rewards)))
            print(agent.learner.weights.transpose())
            if np.mean(rewards) >= 190:
                break

if __name__ == "__main__":
    #experiments.run_all()
    tabular_control()
    #lstd_control()
    #lstd_control_steps()
    #utils.set_results_directory("/NOBACKUP/hhuang63/results3/2017-06-20_20-34-59")

    #import shutil
    #shutil.rmtree("/NOBACKUP/hhuang63/results3/test")
    #utils.set_results_directory("/NOBACKUP/hhuang63/results3/test")
    #utils.set_results_directory("/NOBACKUP/hhuang63/results3/2017-09-14_15-30-47")
    #exp3.run_all(20)
