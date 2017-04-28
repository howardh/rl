import numpy as np

from agent import Agent
from learner import TabularLearner

class TabularAgent(Agent):

    def __init__(self, action_space, discount_factor, learning_rate):
        self.learner = TabularLearner(
                action_space=action_space,
                discount_factor=discount_factor,
                learning_rate=learning_rate
        )

    def run_episode(self, env):
        obs = env.reset()
        action = self.act(obs)

        obs2 = None
        done = False
        step_count = 0
        reward_sum = 0
        # Run an episode
        while not done:
            step_count += 1

            obs2, reward, done, _ = env.step(action)
            action2 = self.act(obs2)
            reward_sum += reward

            self.learner.observe_step(obs, action, reward, obs2, terminal=done)

            # Next time step
            obs = obs2
            action = action2
        return reward, step_count
