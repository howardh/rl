import pytest
import torch

from rl.agent.smdp.a2c import compute_mc_state_value_loss as loss
#from rl.agent.smdp.a2c import compute_mc_state_value_loss_tensor as loss

def test_single_transition():
    state_values = [torch.tensor(1),torch.tensor(2)]
    rewards = [None,1]
    terminals = [False,False]
    discounts = [0.9,0.9]

    predicted_val = 1
    target_val = 1+0.9*2
    expected_output = (predicted_val-target_val)**2

    output = loss(
            target_state_values=state_values,
            state_values=state_values,
            rewards=rewards,
            terminals=terminals,
            discounts=discounts,
    )
    assert pytest.approx(output.item()) == expected_output

def test_two_transitions():
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
            target_state_values=state_values,
            state_values=state_values,
            rewards=rewards,
            terminals=terminals,
            discounts=discounts,
    )
    assert pytest.approx(output[0].item()) == loss_1
    assert pytest.approx(output[1].item()) == loss_2

def test_terminal_after_one_step():
    state_values = [torch.tensor(1),torch.tensor(2),torch.tensor(3)]
    rewards = [None,1,None]
    terminals = [False,True,False]
    discounts = [0.9,0.8,0.7]

    predicted_val = 1
    target_val = 1
    expected_output = (predicted_val-target_val)**2

    output = loss(
            target_state_values=state_values,
            state_values=state_values,
            rewards=rewards,
            terminals=terminals,
            discounts=discounts,
    )
    assert pytest.approx(output[0].item()) == expected_output
    assert pytest.approx(output[1].item()) == 0

def test_terminal_and_new_episode():
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
            target_state_values=state_values,
            state_values=state_values,
            rewards=rewards,
            terminals=terminals,
            discounts=discounts,
    )
    assert pytest.approx(output[0].item()) == loss_1
    assert pytest.approx(output[1].item()) == 0
    assert pytest.approx(output[2].item()) == loss_2
