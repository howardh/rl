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
from cartpole import exp3
from cartpole import exp4
from cartpole import exp5
from cartpole import graph
import experiments
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

def run_all(proc=20):
    # SGD
    experiments.run1(exp1, n=1, proc=proc)
    for _ in tqdm(range(10)):
        experiments.run2(exp1, n=10, m=100, proc=proc)
    experiments.run3(exp1, n=15, proc=proc)
    experiments.plot_best(exp1)
    experiments.plot_final_rewards(exp1)

    # LSTD
    experiments.run1(exp3, n=1, proc=proc)
    for _ in range(100):
        experiments.run2(exp3, n=1, m=100, proc=proc)
    experiments.run3(exp3, n=100, proc=proc)
    experiments.plot_best(exp3)
    experiments.plot_final_rewards(exp3)

    experiments.run3(exp4, n=100, proc=proc)
    experiments.plot_best(exp4)

    experiments.run3(exp5, n=100, proc=proc)
    experiments.plot_best(exp5)

    graph.graph_sgd()
    graph.graph_lstd()
    graph.graph_sarsa_tb()
    graph.graph_all()

if __name__ == "__main__":
    utils.set_results_directory("/home/ml/hhuang63/results/final")
    utils.skip_new_files(True)
    #run_all(20)
    experiments.plot_best_trials(exp1,3)
    experiments.plot_best_trials(exp3,3)
