import numpy as np

def one_hot(x):
    """Return a feature vector representing the given state and action"""
    output = [0] * 16
    output[x] = 1
    return np.array([output]).transpose()

tiles = [
        [0,1,4,5],
        [1,2,5,6],
        [2,3,6,7],
        [4,5,8,9],
        [5,6,9,10],
        [6,7,10,11],
        [8,9,12,13],
        [9,10,13,14],
        [10,11,14,15]
]

def tile_coding(x):
    """
    Return a tile encoding of a FrozenLake 4x4 state.

    >>> tile_encoding(0)
    [[1], [0], [0], [0], [0], [0], [0], [0], [0]]
    >>> tile_encoding(1)
    [[0.5], [0.5], [0], [0], [0], [0], [0], [0], [0]]
    >>> tile_encoding(5)
    [[.25], [.25], [0], [.25], [.25], [0], [0], [0], [0]]
    """
    output = np.array([1 if x in tiles[i] else 0 for i in range(9)])
    output = output/np.sum(output)
    return np.append(output,1).reshape((10,1))
