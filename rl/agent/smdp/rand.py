from typing import Generic, TypeVar

from rl.agent.agent import Agent

ActionType = TypeVar('ActionType')

class RandomAgent(Agent, Generic[ActionType]):
    def __init__(self, action_space):
        self.action_space = action_space

    def observe(self, *args, **kwargs):
        args = args
        kwargs = kwargs
        return

    def act(self, *args, **kwargs) -> ActionType:
        args = args
        kwargs = kwargs
        return self.action_space.sample()

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        state = state # XXX: Get rid of unused variable warning
