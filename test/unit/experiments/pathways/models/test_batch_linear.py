import pytest
import timeit

import torch

from rl.experiments.pathways.models import BatchLinear, NonBatchLinear

def test_batch_non_batch_match_one_input():
    """ BatchLinear and NonBatchLinear should output the same thing. """
    in_features = 4
    out_features = 5
    num_modules = 3
    batch_size = 5

    modules = [
            torch.nn.Linear(in_features=in_features, out_features=out_features)
            for _ in range(num_modules)
    ]

    x = torch.rand([batch_size, in_features])

    bl = BatchLinear(modules)
    nbl = NonBatchLinear(modules)

    output = bl(x)
    expected_output = nbl(x)

    assert output.shape[0] == num_modules
    assert output.shape == expected_output.shape
    assert torch.allclose(output, expected_output)

def test_batch_non_batch_match_multiple_inputs():
    """ BatchLinear and NonBatchLinear should output the same thing. """
    in_features = 4
    out_features = 5
    num_modules = 3
    batch_size = 5

    modules = [
            torch.nn.Linear(in_features=in_features, out_features=out_features)
            for _ in range(num_modules)
    ]

    x = torch.rand([num_modules, batch_size, in_features])

    bl = BatchLinear(modules)
    nbl = NonBatchLinear(modules)

    output = bl(x, batched=True)
    expected_output = nbl(x, batched=True)

    assert output.shape[0] == num_modules
    assert output.shape == expected_output.shape
    assert torch.allclose(output, expected_output)


@pytest.mark.skip(reason="Performance benchmark")
def test_batched_is_faster():
    """ BatchLinear should run faster than NonBatchLinear """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    batched = True
    num_iterations = 1_000
    in_features = 256
    out_features = 512
    num_modules = 8
    batch_size = 16

    modules = [
            torch.nn.Linear(in_features=in_features, out_features=out_features, device=device)
            for _ in range(num_modules)
    ]

    if batched:
        x = torch.rand([num_modules, batch_size, in_features], device=device)
    else:
        x = torch.rand([batch_size, in_features], device=device)

    bl = BatchLinear(modules)
    nbl = NonBatchLinear(modules)

    def run_batched():
        bl(x, batched=batched)
    def run_non_batched():
        nbl(x, batched=batched)

    total_time_batched = timeit.Timer(run_batched).timeit(number=num_iterations)
    total_time_nonbatched = timeit.Timer(run_non_batched).timeit(number=num_iterations)

    print(f"BatchLinear: {total_time_batched}")
    print(f"NonBatchLinear: {total_time_nonbatched}")

    assert total_time_batched < total_time_nonbatched
    assert False


@pytest.mark.skip(reason="Performance benchmark")
def test_batched_is_faster_backward():
    """ BatchLinear should run faster than NonBatchLinear """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    batched = True
    num_iterations = 1_000
    in_features = 256
    out_features = 512
    num_modules = 8
    batch_size = 16

    modules = [
            torch.nn.Linear(in_features=in_features, out_features=out_features, device=device)
            for _ in range(num_modules)
    ]

    if batched:
        x = torch.rand([num_modules, batch_size, in_features], device=device)
    else:
        x = torch.rand([batch_size, in_features], device=device)

    bl = BatchLinear(modules)
    nbl = NonBatchLinear(modules)

    def run_batched():
        output = bl(x, batched=batched)
        output.sum().backward()
    def run_non_batched():
        output = nbl(x, batched=batched)
        output.sum().backward()

    total_time_batched = timeit.Timer(run_batched).timeit(number=num_iterations)
    total_time_nonbatched = timeit.Timer(run_non_batched).timeit(number=num_iterations)

    print(f"BatchLinear: {total_time_batched}")
    print(f"NonBatchLinear: {total_time_nonbatched}")

    assert total_time_batched < total_time_nonbatched
    assert False
