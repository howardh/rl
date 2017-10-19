import numpy as np
import itertools
import scipy.sparse

import atari
from atari import utils

IDENTITY_NUM_FEATURES = 160*210

def identity(x):
    return np.append(x,1).reshape([IDENTITY_NUM_FEATURES,1])

def generate_tiles(w,h,tw,th):
    return [int(i/(w*th))*int(w/tw)+int((i%w)/tw) for i in range(h*w)]

TILES = generate_tiles(160,210,10,15)
BACKGROUND = [0]*IDENTITY_NUM_FEATURES
BASIC_NUM_FEATURES = 16*14*128
def basic(x, background=BACKGROUND):
    NUM_COLOURS = 128
    results = scipy.sparse.lil_matrix((16*14*NUM_COLOURS,1))
    x = x.reshape((160*210,3))
    for i,c in enumerate(x):
        ci = atari.utils.rgb_to_index(c)
        if background[i] == ci:
            continue
        index = TILES[i]*NUM_COLOURS+ci
        results[index] = 1
    return results

def get_basic(env_name):
    bg = atari.utils.compute_background(env_name)
    def foo(x):
        return basic(x, background=bg)
    return foo
