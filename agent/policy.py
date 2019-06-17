import torch

def get_greedy_epsilon_policy(eps):
    def foo(values):
        batch_size, num_actions = values.size()
        probs = torch.ones_like(values)*(eps/(num_actions-1))
        probs[range(batch_size),values.argmax(1)] = 1-eps
        dist = torch.distributions.Categorical(probs)
        return dist
    return foo

def greedy_action(values):
    return values.argmax(1)
