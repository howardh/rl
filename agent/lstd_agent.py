import logging
import os, sys
import collections
import dill as pickle
import datetime
from multiprocessing import Pool
import concurrent.futures
import operator
import itertools
import timeit

import numpy as np

import gym
from gym import spaces

from agent.agent import Agent
from learner.lstd_learner import LSTDLearner
from learner.learner import LSTDTraceLearner
from learner.learner import LSTDTraceQsLearner
from learner.learner import SparseLSTDLearner

class LSTDAgent(Agent):

    def __init__(self, num_features, action_space, discount_factor,
            use_traces=False, trace_factor=None,
            use_importance_sampling=False, sigma=1, features=lambda x: x,
            tree_backup_policy = None, sparse=False):
        if use_traces:
            if sigma==1:
                #print("Initializing LSTD agent with traces")
                self.learner = LSTDTraceLearner(
                        num_features=num_features,
                        action_space=action_space,
                        discount_factor=discount_factor,
                        trace_factor=trace_factor
                )
            else:
                #print("Initializing LSTD agent with Q(sigma=%f)" % sigma)
                self.learner = LSTDTraceQsLearner(
                        num_features=num_features,
                        action_space=action_space,
                        discount_factor=discount_factor,
                        trace_factor=trace_factor,
                        sigma=sigma,
                        tree_backup_policy=tree_backup_policy
                )
        else:
            #print("Initializing LSTD agent")
            if sparse:
                self.learner = SparseLSTDLearner(
                        num_features=num_features,
                        action_space=action_space,
                        discount_factor=discount_factor,
                        use_importance_sampling=use_importance_sampling
                )
            else:
                self.learner = LSTDLearner(
                        num_features=num_features,
                        action_space=action_space,
                        discount_factor=discount_factor,
                        use_importance_sampling=use_importance_sampling
                )
        self.features = features
        self.prev_obs = None
        self.prev_done = True
        self.prev_reward = None

    def update_weights(self):
        self.learner.update_weights()

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
            time1 = timeit.default_timer()
            obs2 = self.features(obs2)
            time2 = timeit.default_timer()
            action2 = self.act(obs2)
            time3 = timeit.default_timer()
            reward_sum += reward

            start_time = time3
            self.learner.observe_step(obs, action, reward, obs2, terminal=done)
            end_time = timeit.default_timer()

            # Next time step
            obs = obs2
            action = action2
        return reward_sum, step_count

    def run_step(self, env):
        if self.prev_done:
            self.prev_obs = self.features(env.reset())

        obs = self.prev_obs
        action = self.act(obs)

        obs2, reward, done, _ = env.step(action)
        obs2 = self.features(obs2)

        self.learner.observe_step(obs, action, reward, obs2, terminal=done)

        self.prev_obs = obs2
        self.prev_reward = reward
        self.prev_done = done

    def test_once(self, env, render=False, max_steps=np.inf):
        """
        Run an episode on the environment by following the target behaviour policy (Probably using a greedy deterministic policy).

        env
            Environment on which to run the test
        """
        reward_sum = 0
        obs = env.reset()
        obs = self.features(obs)
        done = False
        while not done:
            action = self.act(obs, testing=True)
            obs, reward, done, _ = env.step(action)
            obs = self.features(obs)
            reward_sum += reward
            if render:
                env.render()
        return reward_sum
