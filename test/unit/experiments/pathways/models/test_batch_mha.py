import pytest
import torch
import timeit

from rl.experiments.pathways.models import BatchMultiHeadAttention, BatchMultiHeadAttentionEinsum, NonBatchMultiHeadAttention

@pytest.mark.parametrize("MHAModule", [BatchMultiHeadAttention, NonBatchMultiHeadAttention, BatchMultiHeadAttentionEinsum])
def test_match_module_unbatched(MHAModule):
    """ BatchMultiHeadAttention and NonBatchMultiHeadAttention should output the same thing. """
    key_size = 16
    num_heads = 4
    batch_size = 5
    num_inputs = 2
    num_modules = 3
    batch_size = 5

    modules = [
            torch.nn.MultiheadAttention(
                key_size, 
                num_heads=num_heads,
                batch_first=False
            )
            for _ in range(num_modules)
    ]

    input_query = torch.rand([batch_size, key_size])
    input_key = torch.rand([num_inputs, batch_size, key_size])
    input_value = torch.rand([num_inputs, batch_size, key_size])

    mha = MHAModule(modules, key_size=key_size, num_heads=num_heads)

    output = mha(input_query, input_key, input_value)
    expected_output = torch.stack([m(input_query.unsqueeze(0), input_key, input_value)[0] for m in modules])

    assert output.shape[0] == num_modules
    assert output.shape == expected_output.shape
    assert torch.allclose(output, expected_output, atol=1e-7)


@pytest.mark.parametrize("MHAModule", [BatchMultiHeadAttention, NonBatchMultiHeadAttention, BatchMultiHeadAttentionEinsum])
def test_match_module_batched(MHAModule):
    """ BatchMultiHeadAttention and NonBatchMultiHeadAttention should output the same thing. """
    key_size = 16
    num_heads = 4
    batch_size = 5
    num_inputs = 2
    num_modules = 3
    batch_size = 5

    modules = [
            torch.nn.MultiheadAttention(
                key_size, 
                num_heads=num_heads,
                batch_first=False
            )
            for _ in range(num_modules)
    ]

    input_query = torch.rand([num_modules, batch_size, key_size])
    input_key = torch.rand([num_modules, num_inputs, batch_size, key_size])
    input_value = torch.rand([num_modules, num_inputs, batch_size, key_size])

    mha = MHAModule(modules, key_size=key_size, num_heads=num_heads)

    output = mha(input_query, input_key, input_value, batched=True)
    expected_output = torch.stack([
        m(q.unsqueeze(0), k, v)[0]
        for m,q,k,v in zip(modules, input_query, input_key, input_value)
    ])

    assert output.shape[0] == num_modules
    assert output.shape == expected_output.shape
    assert torch.allclose(output, expected_output, atol=1e-7)


def test_batch_non_batch_match_one_input():
    """ BatchMultiHeadAttention and NonBatchMultiHeadAttention should output the same thing. """
    key_size = 16
    num_heads = 4
    batch_size = 5
    num_inputs = 2
    num_modules = 3
    batch_size = 5

    modules = [
            torch.nn.MultiheadAttention(
                key_size, 
                num_heads=num_heads,
                batch_first=False
            )
            for _ in range(num_modules)]

    input_query = torch.rand([batch_size, key_size])
    input_key = torch.rand([num_inputs, batch_size, key_size])
    input_value = torch.rand([num_inputs, batch_size, key_size])

    nbmha = NonBatchMultiHeadAttention(modules, key_size=key_size, num_heads=num_heads)
    bmha = BatchMultiHeadAttention(modules, key_size=key_size, num_heads=num_heads)

    output = bmha(input_query, input_key, input_value)
    expected_output = nbmha(input_query, input_key, input_value)

    assert output.shape[0] == num_modules
    assert output.shape == expected_output.shape
    assert torch.allclose(output, expected_output, atol=1e-7)


def test_batch_non_batch_match_multiple_inputs():
    """ BatchMultiHeadAttention and NonBatchMultiHeadAttention should output the same thing. """
    key_size = 16
    num_heads = 4
    batch_size = 5
    num_inputs = 2
    num_modules = 3
    batch_size = 5

    modules = [
            torch.nn.MultiheadAttention(
                key_size, 
                num_heads=num_heads,
                batch_first=False
            )
            for _ in range(num_modules)]

    input_query = torch.rand([num_modules, batch_size, key_size])
    input_key = torch.rand([num_modules, num_inputs, batch_size, key_size])
    input_value = torch.rand([num_modules, num_inputs, batch_size, key_size])

    nbmha = NonBatchMultiHeadAttention(modules, key_size=key_size, num_heads=num_heads)
    bmha = BatchMultiHeadAttention(modules, key_size=key_size, num_heads=num_heads)

    output = bmha(input_query, input_key, input_value, batched=True)
    expected_output = nbmha(input_query, input_key, input_value, batched=True)

    assert output.shape[0] == num_modules
    assert output.shape == expected_output.shape
    assert torch.allclose(output, expected_output, atol=1e-7)


@pytest.mark.skip(reason="Performance benchmark")
def test_batched_is_faster():
    """ BatchMultiHeadAttention should run faster than NonBatchMultiHeadAttention """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        num_iterations = 1_000
        key_size = 512
        num_heads = 8
        batch_size = 16
        num_inputs = 12
        num_modules = 12
    else:
        device = torch.device('cpu')
        num_iterations = 10
        key_size = 32
        num_heads = 4
        batch_size = 5
        num_inputs = 3
        num_modules = 2
    
    batched = False

    modules = [
            torch.nn.MultiheadAttention(
                key_size, 
                num_heads=num_heads,
                batch_first=False,
                device=device
            )
            for _ in range(num_modules)
    ]

    if batched:
        input_query = torch.rand([num_modules, batch_size, key_size], device=device)
        input_key = torch.rand([num_modules, num_inputs, batch_size, key_size], device=device)
        input_value = torch.rand([num_modules, num_inputs, batch_size, key_size], device=device)
    else:
        input_query = torch.rand([batch_size, key_size], device=device)
        input_key = torch.rand([num_inputs, batch_size, key_size], device=device)
        input_value = torch.rand([num_inputs, batch_size, key_size], device=device)

    bmha = BatchMultiHeadAttention(modules, key_size=key_size, num_heads=num_heads)
    bmhae = BatchMultiHeadAttentionEinsum(modules, key_size=key_size, num_heads=num_heads)
    nbmha = NonBatchMultiHeadAttention(modules, key_size=key_size, num_heads=num_heads)

    def run_batched():
        bmha(input_query, input_key, input_value, batched=batched)
    def run_batched_einsum():
        bmhae(input_query, input_key, input_value, batched=batched)
    def run_non_batched():
        nbmha(input_query, input_key, input_value, batched=batched)

    total_time_batched = timeit.Timer(run_batched).timeit(number=num_iterations)
    total_time_batched_einsum = timeit.Timer(run_batched_einsum).timeit(number=num_iterations)
    total_time_nonbatched = timeit.Timer(run_non_batched).timeit(number=num_iterations)

    print(f"BatchMultiHeadAttention: {total_time_batched}")
    print(f"BatchMultiHeadAttentionEinsum: {total_time_batched_einsum}")
    print(f"NonBatchMultiHeadAttention: {total_time_nonbatched}")

    #assert total_time_batched < total_time_nonbatched
    assert total_time_batched_einsum < total_time_nonbatched
    assert False


@pytest.mark.skip(reason="Performance benchmark")
def test_batched_is_faster_backward():
    """ BatchMultiHeadAttention should run faster than NonBatchMultiHeadAttention """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    batched = True
    num_iterations = 1_000
    key_size = 512
    num_heads = 8
    batch_size = 16
    num_inputs = 12
    num_modules = 12

    modules = [
            torch.nn.MultiheadAttention(
                key_size, 
                num_heads=num_heads,
                batch_first=False,
                device=device
            )
            for _ in range(num_modules)
    ]

    if batched:
        input_query = torch.rand([num_modules, batch_size, key_size], device=device)
        input_key = torch.rand([num_modules, num_inputs, batch_size, key_size], device=device)
        input_value = torch.rand([num_modules, num_inputs, batch_size, key_size], device=device)
    else:
        input_query = torch.rand([batch_size, key_size], device=device)
        input_key = torch.rand([num_inputs, batch_size, key_size], device=device)
        input_value = torch.rand([num_inputs, batch_size, key_size], device=device)

    bmha = BatchMultiHeadAttention(modules, key_size=key_size, num_heads=num_heads)
    bmhae = BatchMultiHeadAttentionEinsum(modules, key_size=key_size, num_heads=num_heads)
    nbmha = NonBatchMultiHeadAttention(modules, key_size=key_size, num_heads=num_heads)

    def run_batched():
        output = bmha(input_query, input_key, input_value, batched=batched)
        output.sum().backward()
    def run_batched_einsum():
        output = bmhae(input_query, input_key, input_value, batched=batched)
        output.sum().backward()
    def run_non_batched():
        output = nbmha(input_query, input_key, input_value, batched=batched)
        output.sum().backward()

    total_time_batched = timeit.Timer(run_batched).timeit(number=num_iterations)
    total_time_batched_einsum = timeit.Timer(run_batched_einsum).timeit(number=num_iterations)
    total_time_nonbatched = timeit.Timer(run_non_batched).timeit(number=num_iterations)

    print(f"BatchMultiHeadAttention: {total_time_batched}")
    print(f"BatchMultiHeadAttentionEinsum: {total_time_batched_einsum}")
    print(f"NonBatchMultiHeadAttention: {total_time_nonbatched}")

    assert total_time_batched < total_time_nonbatched
    assert False


@pytest.mark.skip(reason="Performance benchmark")
def test_thing():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    num_iterations = 1_000

    num_modules = 12
    num_inputs = 13
    batch_size = 16
    embed_dims = 512

    # Initialize inputs and weights/biases
    x = torch.rand([num_inputs, batch_size, embed_dims], device=device)
    weight = torch.rand([num_modules, embed_dims, embed_dims], device=device)
    bias = torch.rand([num_modules, embed_dims], device=device)

    x2 = x.unsqueeze(0)
    weight2 = weight.unsqueeze(1).transpose(-2,-1)
    bias2 = bias.unsqueeze(1).unsqueeze(2)

    # Functions to evaluate
    def run_batched():
        output = x.unsqueeze(0) @ weight.unsqueeze(1).transpose(-2,-1) + bias.unsqueeze(1).unsqueeze(2)
        torch.cuda.synchronize()
        return output

    def run_batched2():
        output = x2 @ weight2 + bias2
        torch.cuda.synchronize()
        return output

    def run_batched3():
        output = torch.einsum ('ijk,lmk -> lijm', x, weight) + bias2
        torch.cuda.synchronize()
        return output

    def run_non_batched():
        output = [
            x @ w.transpose(-2,-1) + b
            for w,b in zip(weight, bias)
        ]
        torch.cuda.synchronize()
        return torch.stack(output)

    # Ensure that they are all computing the same thing
    assert torch.allclose(run_batched(), run_non_batched())
    assert torch.allclose(run_batched2(), run_non_batched())
    assert torch.allclose(run_batched3(), run_non_batched())

    # Measure run time
    total_time_batched = timeit.Timer(run_batched).timeit(number=num_iterations)
    total_time_batched2 = timeit.Timer(run_batched2).timeit(number=num_iterations)
    total_time_batched3 = timeit.Timer(run_batched3).timeit(number=num_iterations)
    total_time_nonbatched = timeit.Timer(run_non_batched).timeit(number=num_iterations)

    print(f"Batch: {total_time_batched}")
    print(f"Batch2: {total_time_batched2}")
    print(f"Batch3: {total_time_batched3}")
    print(f"NonBatch: {total_time_nonbatched}")

    assert total_time_batched2 < total_time_nonbatched
    assert total_time_batched < total_time_nonbatched

    assert False


@pytest.mark.skip(reason="This only exists for debugging purposes. Not a real test.")
def test_thing2():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        num_iterations = 1_000
        embed_dims = 512
        batch_size = 16
        num_modules = 12
    else:
        device = torch.device('cpu')
        num_iterations = 10
        embed_dims = 32
        batch_size = 5
        num_modules = 2

    # Initialize inputs and weights/biases
    x = torch.rand([num_modules, batch_size, embed_dims], device=device)
    weight = torch.rand([num_modules, embed_dims, embed_dims], device=device)
    #bias = torch.rand([num_modules, embed_dims], device=device)

    x2 = x.view([1, num_modules, batch_size, 1, embed_dims])
    weight2 = weight.view([1, num_modules, 1, embed_dims, embed_dims]).transpose(-2,-1)
    #bias2 = bias

    # Functions to evaluate
    def run_batched():
        output = x.unsqueeze(2).unsqueeze(0) @ weight.unsqueeze(1).unsqueeze(0).transpose(-2,-1) #+ bias.unsqueeze(1).unsqueeze(2)
        #torch.cuda.synchronize()
        return output.squeeze()

    def run_batched2():
        output = x2 @ weight2 #+ bias2
        #torch.cuda.synchronize()
        return output.squeeze()

    def run_batched3():
        ##output = torch.einsum ('ijk,lmk -> lijm', x, weight) #+ bias2
        #output = torch.einsum ('nbi,nji-> nbj', x, weight) #+ bias2
        ##output = torch.einsum('abcde,bef -> abcde', x, weight)
        ##torch.cuda.synchronize()
        #return output

        output = torch.stack([
            torch.einsum ('bi,ji-> bj', x[n], weight[n]) #+ bias2
            for n in range(num_modules)
        ])
        #torch.cuda.synchronize()
        return output
    
    def run_batched4():
        ##output = torch.einsum ('ijk,lmk -> lijm', x, weight) #+ bias2
        #output = torch.einsum ('nbi,nji-> nbj', x, weight) #+ bias2
        ##output = torch.einsum('abcde,bef -> abcde', x, weight)
        ##torch.cuda.synchronize()
        #return output

        output = torch.einsum ('nbi,nji-> nbj', x, weight) #+ bias2
        #torch.cuda.synchronize()
        return output

    def run_non_batched():
        output = []
        for n in range(num_modules):
            output.append(
                x[n] @ weight[n].transpose(-2,-1) #+ bias[n]
            )
        #torch.cuda.synchronize()
        return torch.stack(output)

    # Ensure that they are all computing the same thing
    assert torch.allclose(run_batched(),  run_non_batched())
    assert torch.allclose(run_batched2(), run_non_batched())
    assert torch.allclose(run_batched3(), run_non_batched())
    assert torch.allclose(run_batched4(), run_non_batched())

    # Measure run time
    total_time_batched  = timeit.Timer(run_batched ).timeit(number=num_iterations)
    total_time_batched2 = timeit.Timer(run_batched2).timeit(number=num_iterations)
    total_time_batched3 = timeit.Timer(run_batched3).timeit(number=num_iterations)
    total_time_batched4 = timeit.Timer(run_batched4).timeit(number=num_iterations)
    total_time_nonbatched = timeit.Timer(run_non_batched).timeit(number=num_iterations)

    print(f"Batch: {total_time_batched}")
    print(f"Batch2: {total_time_batched2}")
    print(f"Batch3: {total_time_batched3}")
    print(f"Batch4: {total_time_batched4}")
    print(f"NonBatch: {total_time_nonbatched}")

    assert total_time_batched4 < total_time_nonbatched
    #assert total_time_batched < total_time_nonbatched

    assert False
