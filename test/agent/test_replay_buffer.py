import pytest

import agent
import agent.replay_buffer
from agent.replay_buffer import ReplayBufferStackedObs

def test_rbso_init():
    rb = ReplayBufferStackedObs(10,2)
