import numpy as np
from tqdm import tqdm

from agent.agent import Agent
from learner.linear_learner import LinearQsLearner

class LinearAgent(Agent):

    def __init__(self, action_space, discount_factor, learning_rate,
            num_features, features=lambda x: x, trace_factor=0, sigma=0,
            trace_type='accumulating'):
        self.learner = LinearQsLearner(
                action_space=action_space,
                discount_factor=discount_factor,
                learning_rate=learning_rate,
                num_features=num_features,
                trace_factor=trace_factor,
                sigma=sigma,
                trace_type=trace_type
        )
        self.features = features
        self.running_episode = False
        self.prev_obs = None

    def test_once(self, env, max_steps=np.inf, render=False):
        """
        Run an episode on the environment by following the target behaviour policy (Probably using a greedy deterministic policy).

        env
            Environment on which to run the test
        """
        obs = env.reset()
        obs = self.features(obs)
        action = self.act(obs, testing=True)

        obs2 = None
        done = False
        step_count = 0
        reward_sum = 0
        # Run an episode
        while not done:
            step_count += 1

            obs2, reward, done, _ = env.step(action)
            obs2 = self.features(obs2)
            action2 = self.act(obs2, testing=True)
            reward_sum += reward

            # Next time step
            obs = obs2
            action = action2
        return reward_sum

    def run_episode(self, env):
        if self.running_episode:
            raise ValueError("Cannot start new episode while one is already running through Agent.run_step()")
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

    def run_step(self, env):
        if not self.running_episode:
            obs = env.reset()
            obs = self.features(obs)
        else:
            obs = self.prev_obs

        action = self.act(obs)

        obs2, reward, done, _ = env.step(action)
        obs2 = self.features(obs2)

        self.learner.observe_step(obs, action, reward, obs2, terminal=done)

        # Save data for next step
        self.running_episode = not done
        self.prev_obs = obs2
