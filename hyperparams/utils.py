import torch

import hyperparams.distributions
from hyperparams.distributions import Distribution, Uniform, LogUniform, CategoricalUniform, DiscreteUniform

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

def space_to_ranges(space):
    return [space[k].normalized_range() for k in sorted(space.keys()) if isinstance(space[k],Distribution)]

def param_to_vec(param, space):
    # Type checking
    if type(space) is not dict:
        raise ValueError('Argument "space" must be a dictionary.')
    # Compute stuff
    if type(param) is not dict:
        param = dict(param)
    output = []
    for k in sorted(space.keys()):
        if not isinstance(space[k],Distribution):
            pass
        else:
            output.append(space[k].normalize(param[k]))
    return output

def vec_to_param(vec, space):
    assert type(vec) is list or type(vec) is tuple
    output = {}
    for k in sorted(space.keys()):
        if not isinstance(space[k],Distribution):
            output[k] = space[k]
        else:
            output[k] = space[k].unnormalize(vec[0])
            vec = vec[1:]
    return output

def perturb_vec(vec, space, scale=0.01):
    assert type(vec) is list or type(vec) is tuple
    output = []
    for k in sorted(space.keys()):
        if not isinstance(space[k],Distribution):
            continue
        output.append(space[k].perturb(vec[0],scale))
        vec = vec[1:]
    return output
