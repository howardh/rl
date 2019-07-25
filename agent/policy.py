import torch

# State-Action Value

def get_greedy_epsilon_policy(eps):
    def foo(values):
        s = values.size()
        if len(s) == 1:
            num_actions = s[0]
            probs = torch.ones_like(values)*(eps/(num_actions-1))
            probs[values.argmax()] = 1-eps
        elif len(s) == 2:
            batch_size, num_actions = s
            probs = torch.ones_like(values)*(eps/(num_actions-1))
            probs[range(batch_size),values.argmax(1)] = 1-eps
        dist = torch.distributions.Categorical(probs)
        return dist
    return foo

def greedy_action(values):
    return values.argmax(1)

# Continuous action

# Continuous action distribution
