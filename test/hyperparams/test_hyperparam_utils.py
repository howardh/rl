import pytest

import rl.hyperparams
from rl.hyperparams.utils import sample_hyperparam, list_extremes, param_to_vec, vec_to_param
from rl.hyperparams.distributions import Uniform, LogUniform, CategoricalUniform, DiscreteUniform

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

def test_extremes():
    space = {
        'a': Uniform(-10,10)
    }
    output = list_extremes(space)
    assert len(output) == 2
    assert {'a':-10} in output
    assert {'a':10} in output

    space = {
        'a': Uniform(-10,10),
        'b': Uniform(-5,5)
    }
    output = list_extremes(space)
    assert len(output) == 4
    assert {'a':-10,'b':5} in output
    assert {'a':10,'b':5} in output
    assert {'a':-10,'b':-5} in output
    assert {'a':10,'b':-5} in output

    space = {
        'a': Uniform(-10,10),
        'b': Uniform(-5,5),
        'c': 'foo'
    }
    output = list_extremes(space)
    assert len(output) == 4
    assert {'c': 'foo', 'a':-10,'b':5} in output
    assert {'c': 'foo', 'a':10,'b':5} in output
    assert {'c': 'foo', 'a':-10,'b':-5} in output
    assert {'c': 'foo', 'a':10,'b':-5} in output

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

    assert unvec['a'] == pytest.approx(output['a'])
    assert unvec['b'] == output['b']
    assert unvec['c'] == pytest.approx(output['c'])
    assert unvec['d'] == output['d']
    assert unvec['e'] == pytest.approx(output['e'])
    assert unvec['f'] == pytest.approx(output['f'])
