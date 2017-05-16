import gym
import numpy as np

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent
from learner.learner import Optimizer

import cartpole
from cartpole import features
from cartpole import utils

# When taking an action, there's an equal probability of moving in any direction that isn't the opposite direction
# e.g. If you choose up, there's a 1/3 chance of going up, 1/3 of going left, 1/3 of going right

def lstd_control():
    env_name = 'CartPole-v0'
    e = gym.make(env_name)

    action_space = np.array([0,1])
    agent = LSTDAgent(
            action_space=action_space,
            num_features=cartpole.features.IDENTITY_NUM_FEATURES,
            discount_factor=0.99,
            features=cartpole.features.identity,
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
            if np.mean(rewards) >= 190:
                break

if __name__ == "__main__":
    lstd_control()
