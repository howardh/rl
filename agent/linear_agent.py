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
