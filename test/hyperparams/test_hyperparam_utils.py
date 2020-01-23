import pytest

import hyperparams
from hyperparams.utils import sample_hyperparam, param_to_vec, vec_to_param
from hyperparams.distributions import Uniform, LogUniform, CategoricalUniform, DiscreteUniform

def test_sample_empty():
    space = {}
    output = sample_hyperparam(space)
    assert output == {}

def test_sample():
    space = {
        'a': Uniform(-10,10)
    }
    output = sample_hyperparam(space)
    assert output['a'] > -10
    assert output['a'] < 10

def test_vec_conversion():
    space = {
        'a': Uniform(-10,10),
        'b': 'string',
        'c': 100,
        'd': CategoricalUniform(['foo','bar']),
        'e': DiscreteUniform(90,100),
        'f': LogUniform(0.0001,0.1)
    }
    output = sample_hyperparam(space)
    assert len(output) == len(space)

    vec = param_to_vec(output,space)
    unvec = vec_to_param(vec,space)

    assert unvec == output
