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
from learner.lstd_learner import LSPILearner
from learner.lstd_learner import LSTDTraceLearner
from learner.lstd_learner import LSTDTraceQsLearner
from learner.lstd_learner import SparseLSTDLearner

class LSTDAgent(Agent):

    def __init__(self, num_features, action_space, discount_factor,
            use_traces=False, trace_factor=None, trace_type='accumulating',
            use_importance_sampling=False, sigma=1, features=lambda x: x,
            sparse=False, cuda=False, lspi=False, decay=1):
        if lspi:
            self.learner = LSPILearner(
                    num_features=num_features,
                    action_space=action_space,
                    discount_factor=discount_factor,
            )
        else:
            if use_traces:
                #print("Initializing LSTD agent with Q(sigma=%f)" % sigma)
                self.learner = LSTDTraceQsLearner(
                        num_features=num_features,
                        action_space=action_space,
                        discount_factor=discount_factor,
                        trace_factor=trace_factor,
                        sigma=sigma,
                        trace_type=trace_type,
                        decay=decay
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
                            use_importance_sampling=use_importance_sampling,
                            cuda=cuda
                    )
        self.features = features
        self.prev_obs = None
        self.prev_done = True
        self.prev_reward = None

    def update_weights(self):
        self.learner.update_weights()
