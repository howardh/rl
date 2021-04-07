import pytest
import numpy as np
import scipy.sparse
import itertools

import rl.utils as utils

def test_solve():
    a = scipy.sparse.lil_matrix(np.random.rand(5,5))
    b = scipy.sparse.lil_matrix(np.random.rand(5,1))
    x = utils.solve(a,b)
    diff = np.linalg.norm(a*x-b)
    assert diff == pytest.approx(0, abs=1e-6)

def test_gridsearch_single_process(tmpdir):
    def foo(**kwargs):
        utils.save_results((kwargs,None),str(tmpdir))
    params = {
            'a': [1,2,3],
            'b': [4,5,6]
    }
    funcs = utils.gridsearch(params, foo)
    utils.cc(funcs, proc=1)
    results = utils.get_all_results(str(tmpdir))
    results = list(results)

    assert len(results) == 3*3
    assert ({'a': 1, 'b': 4},None) in results
    assert ({'a': 1, 'b': 5},None) in results
    assert ({'a': 1, 'b': 6},None) in results
    assert ({'a': 2, 'b': 4},None) in results
    assert ({'a': 2, 'b': 5},None) in results
    assert ({'a': 2, 'b': 6},None) in results
    assert ({'a': 3, 'b': 4},None) in results
    assert ({'a': 3, 'b': 5},None) in results
    assert ({'a': 3, 'b': 6},None) in results

@pytest.mark.skip(reason="Unimportant for now. Fix later.")
def test_gridsearch_multi_process(tmpdir):
    def foo(**kwargs):
        utils.save_results((kwargs,None),str(tmpdir))
    params = {
            'a': [1,2,3],
            'b': [4,5,6]
    }
    funcs = utils.gridsearch(params, foo)
    utils.cc(funcs, proc=2)
    results = utils.get_all_results(str(tmpdir))
    results = list(results)

    assert len(results) == 3*3
    assert ({'a': 1, 'b': 4},None) in results
    assert ({'a': 1, 'b': 5},None) in results
    assert ({'a': 1, 'b': 6},None) in results
    assert ({'a': 2, 'b': 4},None) in results
    assert ({'a': 2, 'b': 5},None) in results
    assert ({'a': 2, 'b': 6},None) in results
    assert ({'a': 3, 'b': 4},None) in results
    assert ({'a': 3, 'b': 5},None) in results
    assert ({'a': 3, 'b': 6},None) in results

def test_gridsearch_strings():
    called_params = []
    def foo(**kwargs):
        called_params.append(kwargs)
    params = {
            'a': [1,2],
            'b': ['1','2']
    }
    funcs = utils.gridsearch(params, foo)
    utils.cc(funcs)

    assert len(called_params) == 2*2
    assert {'a': 1, 'b': '1'} in called_params
    assert {'a': 1, 'b': '2'} in called_params
    assert {'a': 2, 'b': '1'} in called_params
    assert {'a': 2, 'b': '2'} in called_params

def test_gridsearch_single_value():
    called_params = []
    def foo(**kwargs):
        called_params.append(kwargs)
    params = {
            'a': [1],
            'b': ['1','2']
    }
    funcs = utils.gridsearch(params, foo)
    utils.cc(funcs)

    assert len(called_params) == 2
    assert {'a': 1, 'b': '1'} in called_params
    assert {'a': 1, 'b': '2'} in called_params

def test_gridsearch_list():
    called_params = []
    def foo(**kwargs):
        called_params.append(kwargs)
    params = {
            'a': [1],
            'b': [[],[1,2,3]]
    }
    funcs = utils.gridsearch(params, foo)
    utils.cc(funcs)

    assert len(called_params) == 2
    assert {'a': 1, 'b': []} in called_params
    assert {'a': 1, 'b': [1,2,3]} in called_params

def test_save_results_creates_file(tmpdir):
    assert len(tmpdir.listdir()) == 0
    utils.save_results(({'a': 1}, [1,2,3]), directory=str(tmpdir))
    assert len(tmpdir.listdir()) == 1
    utils.save_results((None, None), directory=str(tmpdir))
    assert len(tmpdir.listdir()) == 2

def test_save_results_and_get_results(tmpdir):
    utils.save_results(({'a': 1}, [1,2,3]), directory=str(tmpdir))
    utils.save_results(({'b': 1}, [4,5,6]), directory=str(tmpdir))

    results = utils.get_results({}, directory=str(tmpdir))
    results = list(results)
    assert len(results) == 2
    assert [1,2,3] in results
    assert [4,5,6] in results

    results = utils.get_results({'a': 2}, directory=str(tmpdir))
    results = list(results)
    assert len(results) == 0

    results = utils.get_results({'b': 1}, directory=str(tmpdir))
    results = list(results)
    assert len(results) == 1
    assert [4,5,6] in results

def test_save_results_and_get_results_exact_match(tmpdir):
    utils.save_results(({'a': 1, 'b': 1}, [1,2,3]), directory=str(tmpdir))
    utils.save_results(({'b': 1}, [4,5,6]), directory=str(tmpdir))

    results = utils.get_results({}, directory=str(tmpdir), match_exactly=True)
    results = list(results)
    assert len(results) == 0

    results = utils.get_results({'a': 1}, directory=str(tmpdir), match_exactly=True)
    results = list(results)
    assert len(results) == 0

    results = utils.get_results({'a': 1, 'b': 1}, directory=str(tmpdir), match_exactly=True)
    results = list(results)
    assert len(results) == 1
    assert [1,2,3] in results

def test_save_results_and_get_results_reduce(tmpdir):
    utils.save_results(({'a': 1}, [1,2,3]), directory=str(tmpdir))
    utils.save_results(({'a': 1}, [2,3,4]), directory=str(tmpdir))
    utils.save_results(({'b': 1}, [4,5,6]), directory=str(tmpdir))

    def reduce(x,acc):
        print(acc)
        return acc+[sum(x)]
    results = utils.get_all_results_reduce(
             str(tmpdir), reduce, lambda: [])
    assert len(results) == 2
    assert 6 in results[frozenset({'a': 1}.items())]
    assert 9 in results[frozenset({'a': 1}.items())]
    assert 4+5+6 in results[frozenset({'b': 1}.items())]

def test_find_best_params(tmpdir):
    # Note: This is not a proper test.
    # The code in here are stuff that should be in separate functions, which are then tested.
    utils.save_results(({'a': 1, 'b': 1}, [[1,1],[2],[3]]), directory=str(tmpdir))
    utils.save_results(({'a': 1, 'b': 1}, [1,1,1]), directory=str(tmpdir))
    utils.save_results(({'a': 1, 'b': 2}, [4,5,6]), directory=str(tmpdir))
    utils.save_results(({'a': 2, 'b': 1}, [1,1,1]), directory=str(tmpdir))
    utils.save_results(({'a': 2, 'b': 2}, [4,3,2]), directory=str(tmpdir))
    def reduce(results,s):
        return s + [results]
    results = utils.get_all_results_reduce(str(tmpdir), reduce, lambda: [])
    performance = {}
    for k,v in results.items():
        """
        v =
        - Trial1
            - Epoch 1
                - Test iter 1
                - Test iter 2
                - ...
            - Epoch 2
                - ...
            - ...
        - Trial 2
            - etc.
        """
        # Average all iterations under an epoch
        # Do a cumulative mean over all epochs
        # Take a max over the cumulative means for each trial
        # Take a mean over all max cum means over all trials
        max_means = []
        for trial in v:
            mean_rewards = [np.mean(epoch) for epoch in trial]
            cum_mean = np.cumsum(mean_rewards)/np.arange(1,len(mean_rewards)+1)
            max_mean = np.max(cum_mean)
            max_means.append(max_mean)
        mean_max_mean = np.mean(max_means)
        performance[k] = mean_max_mean

    print(performance)
    assert len(performance) == 4
    assert performance[frozenset({'a': 1, 'b': 1}.items())] == ((1+2+3)/3+3/3)/2
    assert performance[frozenset({'a': 1, 'b': 2}.items())] == (4+5+6)/3
    assert performance[frozenset({'a': 2, 'b': 1}.items())] == 1
    assert performance[frozenset({'a': 2, 'b': 2}.items())] == 4
