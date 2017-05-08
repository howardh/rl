import numpy as np
import itertools

ONE_HOT_NUM_FEATURES = 8*8

def one_hot(x):
    """Return a feature vector representing the given state and action"""
    output = [0] * ONE_HOT_NUM_FEATURES
    output[x] = 1
    return np.array([output]).transpose()

def generate_tiles(dims, tile_size):
    def convert(x,y):
        return x+y*dims[1]
    results = []
    for y,x in itertools.product(range(dims[1]-tile_size[1]+1),range(dims[0]-tile_size[0]+1)):
        tile = []
        for dy,dx in itertools.product(range(tile_size[1]),range(tile_size[0])):
            tile.append(convert(x+dx,y+dy))
        results.append(tile)
    return results

tiles = generate_tiles((8,8),(2,2))
TILE_CODING_NUM_FEATURES = len(tiles)+1

def tile_coding(x):
    """
    Return a tile encoding of a FrozenLake 8x8 state.
    """
    output = np.array([1 if x in tiles[i] else 0 for i in range(len(tiles))])
    output = output/np.sum(output)
    return np.append(output,1).reshape((TILE_CODING_NUM_FEATURES,1))
