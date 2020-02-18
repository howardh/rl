import pytest

import torch

import agent
import agent.replay_buffer
from agent.replay_buffer import ReplayBufferStackedObs

def test_rbso_init():
    rb = ReplayBufferStackedObs(10,2)

def test_rbso_state_dict():
    rb = ReplayBufferStackedObs(5,2)
    for _ in range(5):
        rb.add_transition(torch.rand([2]),0,0,torch.rand([2]),False)

    state = rb.state_dict()
    rb2 = ReplayBufferStackedObs(5,2)
    rb2.load_state_dict(state)

    assert rb.buffer == rb2.buffer
    assert rb.obs_stack == rb2.obs_stack
