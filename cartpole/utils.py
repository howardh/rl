import numpy as np

def optimal_policy(state):
    w0 = np.matrix([[5.3122436, 0.20275837, -3.39513045, -0.50708204, 11.57365912]])
    w1 = np.matrix([[6.72158678, -1.7409665, 7.28781932, -0.31579945, 11.51384043]])
    v0 = w0*state
    v1 = w1*state
    if v0 > v1:
        return np.array([1,0])
    else:
        return np.array([0,1])

def less_optimal_policy(state):
    w0 = np.matrix([[5.3122436, 0.20275837, -3.39513045, -0.50708204, 11.57365912]])
    w1 = np.matrix([[6.72158678, -1.7409665, 7.28781932, -0.31579945, 11.51384043]])
    v0 = w0*state
    v1 = w1*state
    if v0 > v1:
        return np.array([0.75,0.25])
    else:
        return np.array([0.25,0.75])

def print_policy(agent, f=lambda x: x):
    pass

def print_values(agent, f=lambda x: x):
    pass
