import torch

import hyperparams.distributions

def sample_hyperparam(space):
    # Type checking
    if type(space) is not dict:
        raise ValueError('Argument "space" must be a dictionary.')
    # Sample
    output = {}
    for k,v in space.items():
        try:
            output[k] = v.sample()
        except:
            output[k] = v
    return output
