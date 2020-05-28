import torch

# State-Action Value

def get_greedy_epsilon_policy(eps):
    def foo(values):
        s = values.size()
        if len(s) == 1:
            num_actions = s[0]
            if num_actions == 1:
                probs = torch.tensor([1]).float()
            else:
                probs = torch.ones_like(values)*(eps/(num_actions-1))
                probs[values.argmax()] = 1-eps
        elif len(s) == 2:
            batch_size, num_actions = s
            if num_actions == 1:
                probs = torch.tensor([[1]]*batch_size).float()
            else:
                probs = torch.ones_like(values)*(eps/(num_actions-1))
                probs[range(batch_size),values.argmax(1)] = 1-eps
        dist = torch.distributions.Categorical(probs)
        return dist
    return foo

def greedy_action(values):
    return values.argmax(1)

def get_softmax_policy(temp):
    def foo(values):
        v = torch.exp(values/temp)
        probs = v/v.sum()
        dist = torch.distributions.Categorical(probs)
        return dist
    return foo

# Continuous action

# Continuous action distribution
