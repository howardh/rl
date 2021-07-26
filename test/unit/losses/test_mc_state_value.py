import pytest
import torch
from timeit import default_timer as timer

from rl.agent.smdp.a2c import compute_mc_state_value_loss as loss_iterative
from rl.agent.smdp.a2c import compute_mc_state_value_loss_tensor as loss_tensor

@pytest.mark.parametrize('loss',[loss_iterative,loss_tensor])
def test_single_transition(loss):
    state_values = [torch.tensor(1),torch.tensor(2)]
    rewards = [None,1]
    terminals = [False,False]
    discounts = [0.9,0.9]

    predicted_val = 1
    target_val = 1+0.9*2
    expected_output = (predicted_val-target_val)**2

    output = loss(
            last_state_target_value=float(state_values[-1]),
            state_values=state_values,
            rewards=rewards,
            terminals=terminals,
            discounts=discounts,
    )
    assert pytest.approx(output.item()) == expected_output

@pytest.mark.parametrize('loss',[loss_iterative,loss_tensor])
def test_two_transitions(loss):
    state_values = [torch.tensor(1),torch.tensor(2),torch.tensor(3)]
    rewards = [None,1,5]
    terminals = [False,False,False]
    discounts = [0.9,0.8,0.7]

    predicted_val_1 = 1
    target_val_1 = 1+0.9*5+0.9*0.8*3
    loss_1 = (predicted_val_1-target_val_1)**2

    predicted_val_2 = 2
    target_val_2 = 5+0.8*3
    loss_2 = (predicted_val_2-target_val_2)**2

    print(loss_1,loss_2)

    output = loss(
            last_state_target_value=float(state_values[-1]),
            state_values=state_values,
            rewards=rewards,
            terminals=terminals,
            discounts=discounts,
    )
    assert pytest.approx(output[0].item()) == loss_1
    assert pytest.approx(output[1].item()) == loss_2

@pytest.mark.parametrize('loss',[loss_iterative,loss_tensor])
def test_terminal_after_one_step(loss):
    state_values = [torch.tensor(1),torch.tensor(2),torch.tensor(3)]
    rewards = [None,1,None]
    terminals = [False,True,False]
    discounts = [0.9,0.8,0.7]

    predicted_val = 1
    target_val = 1
    expected_output = (predicted_val-target_val)**2

    output = loss(
            last_state_target_value=float(state_values[-1]),
            state_values=state_values,
            rewards=rewards,
            terminals=terminals,
            discounts=discounts,
    )
    assert pytest.approx(output[0].item()) == expected_output
    assert pytest.approx(output[1].item()) == 0

@pytest.mark.parametrize('loss',[loss_iterative,loss_tensor])
def test_terminal_and_new_episode(loss):
    state_values = [torch.tensor(1),torch.tensor(2),torch.tensor(3),torch.tensor(4)]
    rewards = [None,1,None,1]
    terminals = [False,True,False,False]
    discounts = [0.9,0.8,0.7,0.6]

    predicted_val = 1
    target_val = 1
    loss_1 = (predicted_val-target_val)**2

    predicted_val = 3
    target_val = 1+0.7*4
    loss_2 = (predicted_val-target_val)**2

    output = loss(
            last_state_target_value=float(state_values[-1]),
            state_values=state_values,
            rewards=rewards,
            terminals=terminals,
            discounts=discounts,
    )
    assert pytest.approx(output[0].item()) == loss_1
    assert pytest.approx(output[1].item()) == 0
    assert pytest.approx(output[2].item()) == loss_2

@pytest.mark.skip(reason='The tensor version isn\'t actually faster than the iterative version. At least, not on CPU.')
def test_tensor_faster_than_iterative():
    state_values = [torch.tensor(1) for _ in range(5)]
    rewards = [1.,2.,None,4.,5.]
    terminals = [False,True,False,False,False]
    discounts = [0.9,0.8,0.7,0.6,0.5]
    
    t1 = timer()
    for _ in range(100):
        loss_iterative(
                last_state_target_value=float(state_values[-1]),
                state_values=state_values,
                rewards=rewards,
                terminals=terminals,
                discounts=discounts,
        )
    t2 = timer()
    for _ in range(100):
        loss_tensor(
                last_state_target_value=float(state_values[-1]),
                state_values=state_values,
                rewards=rewards,
                terminals=terminals,
                discounts=discounts,
        )
    t3 = timer()
    time_iterative = t2-t1
    time_tensor = t3-t2
    assert time_tensor < time_iterative
