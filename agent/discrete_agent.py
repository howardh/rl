import numpy as np 
from agent.agent import Agent
from learner.learner import TabularLearner
from learner.learner import TabularQsLearner

class TabularAgent(Agent):

    def __init__(self, action_space, discount_factor, learning_rate,
            trace_factor=0, sigma=0, initial_value=0, features=lambda x: x):
        self.learner = TabularQsLearner(
                action_space=action_space,
                discount_factor=discount_factor,
                learning_rate=learning_rate,
                initial_value=initial_value,
                trace_factor=trace_factor,
                sigma=sigma
        )
        self.features = features
