import torch

from rl.agent.smdp.a2c import compute_entropy as entropy

def test_uniform_prob_vs_nonuniform():
    """
    The highest entropy distribution on a discrete support is a uniform distribution. Check that the entropy of a uniform distribution is higher than another arbitrarily-chosen distribution.
    """
    uniform_log_prob = torch.tensor([1,1,1])
    nonuniform_log_prob = torch.tensor([1,2,3])

    uniform_entropy = entropy(uniform_log_prob)
    nonuniform_entropy = entropy(nonuniform_log_prob)

    assert uniform_entropy > nonuniform_entropy
