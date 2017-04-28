import gym
import numpy as np

from discrete_agent import TabularAgent
from learner import TabularLearner

def features(x, a):
    """Return a feature vector representing the given state and action"""
    output = [0] * (16*4)
    output[x+a*16] = 1
    return np.array([output])

def frozen_lake_policy(state):
    """
    ?Optimal? policy for frozen lake

    SFFF
    FHFH
    FFFH
    HFFG
    """

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

def print_frozen_lake_policy(agent):
    dirs = "<v>^"
    holes = [5,7,11,12]
    x = ''
    for i in range(4*4):
        if i%4 == 0:
            x += '\n'
        if i in holes:
            x = x+' '
        else:
            x = x+dirs[np.argmax(agent.learner.get_target_policy(i))]
    print(x)

def main():
    env_name = 'FrozenLake-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1,2,3])
    agent = TabularAgent(action_space=action_space, discount_factor=0.9, learning_rate=0.1)
    #agent.set_target_policy(frozen_lake_policy)
    #agent.set_behaviour_policy(frozen_lake_policy)

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
    agent = TabularAgent(action_space=action_space, discount_factor=0.9, learning_rate=0.1)
    agent.set_target_policy(frozen_lake_policy)
    agent.set_behaviour_policy(frozen_lake_policy)

    iters = 0
    while True:
        iters += 1

        agent.run_episode(e)
        if iters % 500 == 0:
            # TODO: Check value difference
            print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))

if __name__ == "__main__":
    main()
