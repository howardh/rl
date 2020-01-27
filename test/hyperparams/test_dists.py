import pytest

import hyperparams
import hyperparams.distributions

def test_uniform():
    dist = hyperparams.distributions.Uniform(5,10)
    x = dist.sample()
    assert x >= 5
    assert x <= 10
    x_norm = dist.normalize(x)
    x_unnorm = dist.unnormalize(x_norm)
    assert x == x_unnorm
    x2 = dist.unnormalize(dist.perturb(x_norm,0.1))
    assert x2 >= 5
    assert x2 <= 10

def test_loguniform():
    dist = hyperparams.distributions.LogUniform(5,10)
    x = dist.sample()
    assert x >= 5
    assert x <= 10
    x_norm = dist.normalize(x)
    x_unnorm = dist.unnormalize(x_norm)
    assert x == x_unnorm
    x2 = dist.unnormalize(dist.perturb(x_norm,0.1))
    assert x2 >= 5
    assert x2 <= 10

def test_categoricaluniform():
    vals = ['a','b','c']
    dist = hyperparams.distributions.CategoricalUniform(vals)
    x = dist.sample()
    assert x in vals
    x_norm = dist.normalize(x)
    x_unnorm = dist.unnormalize(x_norm)
    assert x == x_unnorm
    x2 = dist.unnormalize(dist.perturb(x_norm,0.1))
    assert x2 in vals

def test_discreteuniform():
    dist = hyperparams.distributions.DiscreteUniform(5,10)
    x = dist.sample()
    assert x >= 5
    assert x <= 10
    x_norm = dist.normalize(x)
    x_unnorm = dist.unnormalize(x_norm)
    assert x == x_unnorm
    x2 = dist.unnormalize(dist.perturb(x_norm,0.1))
    assert x2 >= 5
    assert x2 <= 10
