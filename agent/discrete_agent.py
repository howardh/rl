import numpy as np

from agent.agent import Agent
from learner.learner import TabularLearner
from learner.learner import Optimizer

class TabularAgent(Agent):

    def __init__(self, action_space, discount_factor, learning_rate,
            initial_value=0, optimizer=Optimizer.NONE, features=lambda x: x):
        self.learner = TabularLearner(
                action_space=action_space,
                discount_factor=discount_factor,
                learning_rate=learning_rate,
                optimizer=optimizer,
                initial_value=initial_value
        )
        self.features = features

    def run_episode(self, env):
        obs = env.reset()
        obs = self.features(obs)
        action = self.act(obs)

        obs2 = None
        done = False
        step_count = 0
        reward_sum = 0
        # Run an episode
        while not done:
            step_count += 1

            obs2, reward, done, _ = env.step(action)
            obs2 = self.features(obs2)
            action2 = self.act(obs2)
            reward_sum += reward

            self.learner.observe_step(obs, action, reward, obs2, terminal=done)

            # Next time step
            obs = obs2
            action = action2
        return reward_sum, step_count

    def run_steps(self, env, num_steps):
        obs = self.features(obs)
        action = self.act(obs)

        reward_sum = 0
        # Run steps
        for step_count in range(1,num_steps+1):
            obs2, reward, done, _ = env.step(action)
            obs2 = self.features(obs2)
            action2 = self.act(obs2)
            reward_sum += reward

            self.learner.observe_step(obs, action, reward, obs2, terminal=done)

            # Next time step
            if done:
                obs = env.reset()
                obs = self.features(obs)
                action = self.act(obs)
                obs2 = None
                done = False
            else:
                obs = obs2
                action = action2

        return reward_sum, num_steps
