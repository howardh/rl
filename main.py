import gym
import numpy as np

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent
from learner.learner import TabularLearner
from learner.learner import Optimizer

# When taking an action, there's an equal probability of moving in any direction that isn't the opposite direction
# e.g. If you choose up, there's a 1/3 chance of going up, 1/3 of going left, 1/3 of going right

def frozen_lake_features(x):
    """Return a feature vector representing the given state and action"""
    output = [0] * 16
    output[x] = 1
    return np.array([output]).transpose()

def frozen_lake_policy(state):
    """
    ?Optimal? policy for frozen lake

    SFFF
    FHFH
    FFFH
    HFFG
    """

    if not isinstance(state,int):
        state = state.tolist().index([1])

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    NONE = 0
    policy = [
            LEFT, UP, LEFT, UP,
            LEFT, NONE, RIGHT, NONE,
            UP, DOWN, LEFT, NONE,
            NONE, RIGHT, DOWN, NONE
    ]
    result = [0,0,0,0]
    result[policy[state]] = 1
    return result

def print_frozen_lake_policy(agent, f=lambda x: x):
    dirs = "<v>^"
    holes = [5,7,11,12]
    x = ''
    for i in range(4*4):
        if i%4 == 0:
            x += '\n'
        if i in holes:
            x = x+' '
        else:
            x = x+dirs[np.argmax(agent.learner.get_target_policy(f(i)))]
    print(x)

def print_frozen_lake_values(agent):
    vals = [agent.learner.get_state_value(frozen_lake_features(s)) for s in range(16)]
    vals = np.reshape(vals, (4,4))
    np.set_printoptions(precision=5, suppress=True)
    print(vals)

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
            print_frozen_lake_policy(agent)
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
    agent = LSTDAgent(action_space=action_space, num_features=16, discount_factor=0.99, features=frozen_lake_features)

    iters = 0
    while True:
        iters += 1
        agent.run_episode(e)
        if iters % 500 == 0:
            agent.update_weights()
            rewards = agent.test(e, 100)
            print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
            print_frozen_lake_policy(agent, f=frozen_lake_features)
            print_frozen_lake_values(agent)
            if np.mean(rewards) >= 0.78:
                break

def lstd_policy_evaluation():
    env_name = 'FrozenLake-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = LSTDAgent(action_space=action_space, num_features=16, discount_factor=0.9, features=frozen_lake_features)
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
            print_frozen_lake_values(agent)

if __name__ == "__main__":
    #control()
    #policy_evaluation()
    #lstd_policy_evaluation()
    lstd_control()
