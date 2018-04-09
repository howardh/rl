import gym
import numpy as np
import datetime

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent

import cartpole
#import cartpole.experiments
import cartpole.features
import cartpole.utils

from cartpole import exp1
#from cartpole import exp2
from cartpole import exp3
import utils

# When taking an action, there's an equal probability of moving in any direction that isn't the opposite direction
# e.g. If you choose up, there's a 1/3 chance of going up, 1/3 of going left, 1/3 of going right

def lstd_control():
    env_name = 'CartPole-v0'
    e = gym.make(env_name)
    start_time = datetime.datetime.now()

    action_space = np.array([0,1])
    agent = LSTDAgent(
            action_space=action_space,
            num_features=cartpole.features.IDENTITY_NUM_FEATURES,
            discount_factor=0.99,
            features=cartpole.features.identity2,
            use_importance_sampling=False,
            use_traces=False,
            sigma=None,
            trace_factor=None,
    )
    agent.set_behaviour_policy("0.1-epsilon")
    #agent.set_behaviour_policy(utils.optimal_policy)
    #agent.set_behaviour_policy(utils.less_optimal_policy)
    agent.set_target_policy("0-epsilon")

    iters = 0
    while True:
        iters += 1
        r,s = agent.run_episode(e)
        if iters % 100 == 0:
            agent.update_weights()
            print("Testing...")
            print(agent.learner.weights.transpose())
            rewards = agent.test(e, 100, render=False, processors=3)
            print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
            if np.mean(rewards) >= 190:
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
    #lstd_control()
    #lstd_control_steps()
    #utils.set_results_directory("/NOBACKUP/hhuang63/results3/2018-03-27_17-59-35")

    #import shutil
    #shutil.rmtree("/NOBACKUP/hhuang63/results3/test")
    #utils.set_results_directory("/NOBACKUP/hhuang63/results3/test2")
    #exp3.run_all(20)
    utils.set_results_directory("/NOBACKUP/hhuang63/results3/2018-03-28_16-58-42")
    #exp1.run(n=1,proc=10)
    #d = exp1.parse_results()
    while True:
        exp1.run2(n=5,proc=10)
        #d = exp1.parse_results2(labels=['ttl','mr','fr'])
        d = exp1.parse_results2(labels=['mr','fr'])
    #exp3.run(n=1,proc=15)
    #exp3.parse_results()
    #exp3.run2(n=100,proc=10)
    #exp3.parse_results2()
    #exp3.run3(n=100,proc=10)
    #exp3.parse_results3()
