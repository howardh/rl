import pytest
import torch

from rl.experiments.pathways.models import *

#@pytest.mark.parametrize()
#def test_stuff():
#    inputs = [{
#    }]
#    model = ModularPolicy(
#    )

@pytest.mark.parametrize('shared_key',[(True,),(False,)])
def test_discrete_input_batch_1(shared_key):
    module = DiscreteInput(
            input_size=1,
            key_size=32,
            value_size=32,
            shared_key=shared_key
    )
    val = torch.tensor([0])
    output = module(val)
    assert torch.tensor(output['key'].shape).tolist() == [1,32]
    assert torch.tensor(output['value'].shape).tolist() == [1,32]

@pytest.mark.parametrize('shared_key',[(True,),(False,)])
def test_discrete_input_batch_2(shared_key):
    module = DiscreteInput(
            input_size=1,
            key_size=32,
            value_size=32,
            shared_key=shared_key
    )
    val = torch.tensor([0,0])
    output = module(val)
    assert torch.tensor(output['key'].shape).tolist() == [2,32]
    assert torch.tensor(output['value'].shape).tolist() == [2,32]

@pytest.mark.parametrize('shared_key',[(True,),(False,)])
def test_discrete_input_different_vals(shared_key):
    module = DiscreteInput(
            input_size=3,
            key_size=32,
            value_size=512, # Make this bigger so it's impossible to get the same values twice
            shared_key=shared_key
    )
    val = torch.tensor([0,1,2,0,1,2])
    output = module(val)
    assert torch.tensor(output['key'].shape).tolist() == [6,32]
    assert torch.tensor(output['value'].shape).tolist() == [6,512]
    assert (output['value'][0] == output['value'][3]).all()
    assert (output['value'][1] == output['value'][4]).all()
    assert (output['value'][2] == output['value'][5]).all()

@pytest.mark.parametrize('shape',[[1],[2,2]])
def test_discrete_2D_input(shape):
    module = DiscreteInput(
            input_size=1,
            key_size=32,
            value_size=32,
    )
    val = torch.zeros(shape)
    output = module(val)
    assert torch.tensor(output['key'].shape).tolist() == [*shape,32]
    assert torch.tensor(output['value'].shape).tolist() == [*shape,32]

def test_scalar_input_1():
    module = ScalarInput(
            key_size=32,
            value_size=32,
    )
    val = torch.empty([1])
    output = module(val)
    assert torch.tensor(output['key'].shape).tolist() == [32]
    assert torch.tensor(output['value'].shape).tolist() == [32]

def test_scalar_input_2():
    module = ScalarInput(
            key_size=32,
            value_size=32,
    )
    val = torch.empty([1,1])
    output = module(val)
    assert torch.tensor(output['key'].shape).tolist() == [1,32]
    assert torch.tensor(output['value'].shape).tolist() == [1,32]

def test_scalar_input_3():
    module = ScalarInput(
            key_size=32,
            value_size=32,
    )
    val = torch.empty([10,1])
    output = module(val)
    assert torch.tensor(output['key'].shape).tolist() == [10,32]
    assert torch.tensor(output['value'].shape).tolist() == [10,32]

def test_scalar_input_4():
    module = ScalarInput(
            key_size=32,
            value_size=32,
    )
    val = torch.empty([10])
    with pytest.raises(Exception):
        module(val)
