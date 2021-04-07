import torch
import numpy as np 
from learner.learner import TabularLearner
from learner.learner import TabularQsLearner

from .agent import Agent
from .policy import get_greedy_epsilon_policy, greedy_action

class TabularAgent(Agent):

    def __init__(self, action_space, discount_factor, learning_rate,
            trace_factor=0, sigma=0, initial_value=0,
            behaviour_policy=get_greedy_epsilon_policy(0.1),
            target_policy=get_greedy_epsilon_policy(0)):
        self.learner = TabularQsLearner(
                action_space=action_space,
                discount_factor=discount_factor,
                learning_rate=learning_rate,
                initial_value=initial_value,
                trace_factor=trace_factor,
                sigma=sigma,
                target_policy = target_policy
        )
        self.behaviour_policy = behaviour_policy
        self.target_policy = target_policy

    def act(self, observation, testing=False):
        """Return a random action according to the current behaviour policy"""
        #observation = torch.tensor(observation, dtype=torch.float).view(-1,4,84,84).to(self.device)
        vals = self.learner.get_all_state_action_values(observation)
        if testing:
            policy = self.target_policy
        else:
            policy = self.behaviour_policy
        dist = policy(vals)
        return dist.sample().item()
