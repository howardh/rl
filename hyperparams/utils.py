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

def param_to_vec(param, space):
    # Type checking
    if type(space) is not dict:
        raise ValueError('Argument "space" must be a dictionary.')
    # Compute stuff
    if type(param) is not dict:
        param = dict(param)
    output = []
    for k in sorted(space.keys()):
        if not isinstance(space[k],hyperparams.distributions.Distribution):
            pass
        else:
            output.append(space[k].normalize(param[k]))
    return output

def vec_to_param(vec, space):
    assert type(vec) is list or type(vec) is tuple
    output = {}
    for k in sorted(space.keys()):
        if not isinstance(space[k],hyperparams.distributions.Distribution):
            output[k] = space[k]
        else:
            output[k] = space[k].unnormalize(vec[0])
            vec = vec[1:]
    return output
