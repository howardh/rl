from typing import Optional

import numpy as np
import gym
import gym.spaces

from rl.agent.agent import DeployableAgent
from rl.agent.smdp.hrl import HRLAgent

class DummyAgent(DeployableAgent):
    def __init__(self):
        self.reset()
    def reset(self):
        self.act_calls = []
        self.observe_calls = []
        self.next_action : Optional[int] = None
    def act(self, **kwargs):
        self.act_calls.append(kwargs)
        return self.next_action
    def observe(self, **kwargs):
        self.observe_calls.append(kwargs)
    # testing methods
    def assert_acted(self):
        assert len(self.act_calls) > 0
    def assert_did_not_act(self):
        assert len(self.act_calls) == 0
    def assert_observed(self, **kwargs):
        assert len(self.observe_calls) > 0, 'Expected `observe` to have been called, but it was never called.'
        assert self.observe_calls[-1] == kwargs, '`observe` was not called with the expected parameters.'
    def assert_did_not_observe(self):
        assert len(self.observe_calls) == 0, 'Expected `observe` to not be called, but it was called %d time(s)' % len(self.observe_calls)
    # Deployment Stuff
    def state_dict_deploy(self):
        return {}
    def load_state_dict_deploy(self, state):
        state = state
        pass

def test_0_delay():
    """ Check that the correct child agent is receiving the observations and acting. """
    parent_agent = DummyAgent()
    children_agent = [DummyAgent() for _ in range(2)]
    agent = HRLAgent(
            action_space=gym.spaces.Discrete(3),
            observation_space=gym.spaces.Box(low=np.array([0]),high=np.array([1])),
            agent=parent_agent,
            children=children_agent,
            children_discount=0.99,
            delay=0,
    )

    DEAULT_OBS = {
            'obs': np.array([0]),
            'reward': -1,
            'terminal': False,
            'testing': False,
            'discount': 0.99,
            'time': 1,
            'env_key': 'key'
    }

    obs = {
            **DEAULT_OBS,
            'obs': np.array([0]),
    }
    agent.observe(**obs)

    parent_agent.assert_did_not_act()
    parent_agent.assert_observed(**obs)
    parent_agent.next_action = 0

    agent.act(env_key='key')

    parent_agent.assert_acted()
    parent_agent.assert_observed(**obs)
    children_agent[0].assert_acted()
    children_agent[1].assert_did_not_act()
    children_agent[0].assert_observed(**obs)
    children_agent[1].assert_did_not_observe()

    parent_agent.reset()
    for ca in children_agent:
        ca.reset()

    obs = {
            **DEAULT_OBS,
            'obs': np.array([1]),
    }
    agent.observe(**obs)

    parent_agent.next_action = 0

    agent.act(env_key='key')
