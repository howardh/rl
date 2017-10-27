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

def get_one_hot_optimal_policy(num_pos, num_vel, prob):
    p = prob
    q = (1-p)/2
    n = 10
    i=([True]*int(num_vel/2)+[False]*(num_vel-int(n/2)))*num_pos
    j=([False]*(num_vel-int(num_vel/2))+[True]*int(num_vel/2))*num_pos
    def policy(state):
        if any(state[i]):
            return np.array([p,q,q])
        elif any(state[j]):
            return np.array([q,q,p])
        else:
            return np.array([q,p,q])
    return policy

def print_policy(agent, f=lambda x: x):
    pass

def print_values(agent, f=lambda x: x):
    pass
