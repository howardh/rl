import numpy as np
import itertools

IDENTITY_NUM_FEATURES = 3
LINEARLY_SEPARABLE_NUM_FEATURES = 4
ONE_HOT_NUM_POS = 16
ONE_HOT_NUM_VEL = 16
ONE_HOT_NUM_FEATURES = ONE_HOT_NUM_POS*ONE_HOT_NUM_VEL

def identity(x):
    return np.append(x,1).reshape([IDENTITY_NUM_FEATURES,1])

def linearly_separable(x):
    return np.array([x[0], x[1], (x[0]+0.5)*x[1],1]).reshape([LINEARLY_SEPARABLE_NUM_FEATURES,1])

def one_hot(x):
    pos = int((x[0]+1.2)/1.8*ONE_HOT_NUM_POS)
    vel = int((x[1]+.7)/1.4*ONE_HOT_NUM_VEL)
    result = [0]*ONE_HOT_NUM_FEATURES
    result[vel+pos*ONE_HOT_NUM_VEL] = 1
    return np.array(result).reshape([ONE_HOT_NUM_FEATURES,1])

def normalize(x):
    pass
