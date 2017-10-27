import gym
import numpy as np
import datetime
import dill
import os

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent

import mountaincar 
import mountaincar.features
import mountaincar.utils

#from mountaincar import exp1
import utils

def tabular_control(discount_factor, learning_rate, initial_value, num_pos,
        num_vel, behaviour_eps, target_eps, epoch, max_iters, test_iters):
    env_name = 'MountainCar-v0'
    e = gym.make(env_name)
    start_time = datetime.datetime.now()

    action_space = np.array([0,1,2])
    agent = TabularAgent(
            action_space=action_space,
            features=mountaincar.features.get_one_hot(num_pos,num_vel),
            discount_factor=discount_factor,
            learning_rate=learning_rate,
            initial_value=initial_value
    )
    agent.set_behaviour_policy("%f-epsilon" % behaviour_eps)
    #agent.set_behaviour_policy(mountaincar.utils.get_one_hot_optimal_policy(num_pos, num_vel, 0.75))
    #agent.set_behaviour_policy(utils.less_optimal_policy)
    agent.set_target_policy("%f-epsilon" % target_eps)

    rewards = []
    steps_to_learn = None
    iters = 0
    while True:
        iters += 1
        r,s = agent.run_episode(e)
        rewards.append(r)
        print("... %d\r" % iters, end='')
        if r != -200:
            print("Iteration: %d\t Reward: %d"%(iters, r))
        if epoch is not None and iters % epoch == 0:
            print("Testing...")
            #print(agent.learner.weights.transpose())
            rewards = agent.test(e, test_iters, render=False, processors=1)
            print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
            if np.mean(rewards) >= -110:
                if steps_to_learn is None:
                    steps_to_learn = iters
        if iters > max_iters:
            break
    return rewards,steps_to_learn

def parse_results(directory):
    # List files in directory
    # Load all files
    # From each file...
    #   Compute the mean of rewards
    #   Add to dictionary
    # Sort dictionary by mean rewards and print top 10 along with scores
    # Sort dictionary by time to learn and print top 10 along with scores

    results = []
    file_names = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]
    for file_name in file_names:
        with open(os.path.join(directory,file_name), 'rb') as f:
            x = dill.load(f)
            if x[2] is None:
                results.append((x[0], file_name, np.mean(x[1]), np.inf))
            else:
                results.append((x[0], file_name, np.mean(x[1]), x[2]))
    print("\nSorting by mean reward...")
    results.sort(key=lambda x: x[2], reverse=True)
    for i in range(min(10,len(results))):
        print(results[i][2])
        print("\t%s" % results[i][0])
        print("\t%s" % results[i][1])
        print("\t%s" % results[i][3])

    print("\nSorting by time to learn...")
    results.sort(key=lambda x: x[3], reverse=True)
    for i in range(min(10,len(results))):
        print(results[i][3])
        print("\t%s" % results[i][0])
        print("\t%s" % results[i][1])
        print("\t%s" % results[i][2])

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
    import argparse
    parser = argparse.ArgumentParser(description="Boo")
    parser.add_argument("--parse-results",
            action="store_true")
    parser.add_argument("--learning-rate",
            type=float, default=0.5, help="Learning rate")
    parser.add_argument("--discount",
            type=float, default=0.9, help="")
    parser.add_argument("--behaviour-epsilon",
            type=float, default=0.1, help="")
    parser.add_argument("--target-epsilon",
            type=float, default=0, help="")
    parser.add_argument("--initial-value",
            type=float, default=0, help="")
    parser.add_argument("--num-pos",
            type=int, default=10, help="")
    parser.add_argument("--num-vel",
            type=int, default=10, help="")
    parser.add_argument("--epoch",
            type=int, default=None, help="Number of episodes to complete before testing")
    parser.add_argument("--max-iters",
            type=int, default=3000, help="Maximum number of episodes to run while learning")
    parser.add_argument("--test-iters",
            type=int, default=10, help="Number of episodes to run when testing")
    parser.add_argument("--results-dir",
            type=str, default="./results",
            help="Directory in which to place the pickled results")
    args = parser.parse_args()
    print(args)

    if args.parse_results:
        parse_results(args.results_dir)
    else:
        r,s = tabular_control(discount_factor=args.discount,
                learning_rate=args.learning_rate,
                initial_value=args.initial_value,
                num_pos=args.num_pos,
                num_vel=args.num_vel,
                behaviour_eps=args.behaviour_epsilon,
                target_eps=args.target_epsilon,
                epoch=args.epoch,
                test_iters=args.test_iters,
                max_iters=args.max_iters)
        data = (args, r, s)
        file_name, file_num = utils.find_next_free_file("results", "pkl", args.results_dir)
        with open(file_name, "wb") as f:
            dill.dump(data, f)
