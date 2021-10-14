import pytest
import torch

from rl.agent.smdp.a2c import compute_advantage_policy_gradient as loss

def test_single_transition():
    log_action_probs = [torch.tensor(1., requires_grad=True),torch.tensor(2., requires_grad=True)]
    target_state_values = [torch.tensor(1),torch.tensor(2)]
    rewards = [None,1]
    terminals = [False,False]
    discounts = [0.9,0.9]

    predicted_return = 1
    sampled_return = 1+0.9*2
    advantage = sampled_return-predicted_return
    expected_output = -(log_action_probs[0]*advantage).item()

    output = loss(
            log_action_probs=log_action_probs,
            target_state_values=target_state_values,
            rewards=rewards,
            terminals=terminals,
            discounts=discounts,
    )
    assert pytest.approx(output.item()) == expected_output

    # Check that the gradient is in the right direction
    # The sampled return is higher than the predicted return, so we expect the action probability to increase
    optimizer = torch.optim.SGD([log_action_probs[0]], lr=0.1)
    optimizer.zero_grad()
    output.backward()
    optimizer.step()

    assert log_action_probs[0].item() > 1
