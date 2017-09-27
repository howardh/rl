import numpy as np

def optimal_policy(state):
    if state[1] < 0:
        return np.array([1,0,0])
    else:
        return np.array([0,0,1])

def less_optimal_policy(state, p=0.75):
    q = (1-p)/2
    if state[1] < 0:
        return np.array([p,q,q])
    else:
        return np.array([q,q,p])

def optimal_policy2(state):
    n = 16
    i=([True]*int(n/2)+[False]*int(n/2))*n
    if any(state[i]):
        return np.array([1,0,0])
    else:
        return np.array([0,0,1])

def less_optimal_policy2(state, p=0.75):
    q = (1-p)/2
    n = 16
    i=([True]*int(n/2)+[False]*int(n/2))*n
    if any(state[i]):
        return np.array([p,q,q])
    else:
        return np.array([q,q,p])

def print_policy(agent, f=lambda x: x):
    pass

def print_values(agent, f=lambda x: x):
    pass
