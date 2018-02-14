import numpy as np
from tqdm import tqdm

from agent.agent import Agent
from learner.linear_learner import LinearLearner

class LinearAgent(Agent):

    def __init__(self, action_space, discount_factor, learning_rate,
            num_features, features=lambda x: x, trace_factor=0, sigma=0):
        self.learner = LinearQsLearner(
                action_space=action_space,
                discount_factor=discount_factor,
                learning_rate=learning_rate,
                num_features=num_features,
                trace_factor=trace_factor,
                sigma=sigma
        )
        self.features = features

    def test_once(self, env, max_steps=np.inf, render=False):
        """
        Run an episode on the environment by following the target behaviour policy (Probably using a greedy deterministic policy).

        env
            Environment on which to run the test
        """
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

            # Next time step
            obs = obs2
            action = action2
        return reward_sum

    def run_episode(self, env):
        obs = env.reset()
        obs = self.features(obs)
        action = self.act(obs)

        obs2 = None
        done = False
        step_count = 0
        reward_sum = 0
        # Run an episode
        with tqdm() as pbar:
            while not done:
                step_count += 1
                pbar.update(1)

                obs2, reward, done, _ = env.step(action)
                obs2 = self.features(obs2)
                action2 = self.act(obs2)
                reward_sum += reward

                self.learner.observe_step(obs, action, reward, obs2, terminal=done)

                # Next time step
                obs = obs2
                action = action2
        return reward_sum, step_count
