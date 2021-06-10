from typing import Generic, TypeVar

from rl.agent.agent import Agent

ActionType = TypeVar('ActionType')

class ConstantAgent(Agent, Generic[ActionType]):
    def __init__(self, action):
        self.action = action

    def observe(self, **_):
        return

    def act(self, **_) -> ActionType:
        return self.action

    def state_dict(self):
        return {
                'action': self.action
        }

    def load_state_dict(self, state):
        self.action = state['action']
