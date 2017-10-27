import numpy as np
import itertools

IDENTITY_NUM_FEATURES = 3

def identity(x):
    return np.append(x,1).reshape([IDENTITY_NUM_FEATURES,1])

def get_one_hot(num_pos, num_vel):
    num_features = num_pos*num_vel
    def foo(x):
        pos = int((x[0]+1.2)/1.8*num_pos)
        vel = int((x[1]+.7)/1.4*num_vel)
        result = [0]*num_features
        result[vel+pos*num_vel] = 1
        return np.array(result).reshape([num_features,1])
    return foo
