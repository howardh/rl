import pytest
import numpy as np
import scipy.sparse
import itertools

import utils

def test_solve():
    a = scipy.sparse.lil_matrix(np.random.rand(5,5))
    b = scipy.sparse.lil_matrix(np.random.rand(5,1))
    x = utils.solve(a,b)
    diff = np.linalg.norm(a*x-b)
    assert diff == pytest.approx(0, abs=1e-6)

#def test_solve_approx(self):
#    a = scipy.sparse.lil_matrix(np.random.rand(5,5))
#    b = scipy.sparse.lil_matrix(np.random.rand(5,1))
#    x = utils.solve_approx(a,b)
#    diff = np.linalg.norm(a*x-b)
#    self.assertAlmostEqual(diff,0,delta=.3,msg="Diff too large: %s" % diff)

#class TestUtilsFile(unittest.TestCase):
#    def setUp(self):
#        self.tempdir = testfixtures.TempDirectory()
#        text = ['0,"[0,0,0,0,0]"',
#                '10,"[1,0,0,0,0]"',
#                '20,"[0,1,0,1,0]"',
#                '30,"[0,1,1,0,1]"'
#        ]
#        text = "\n".join(text)
#        self.path = self.tempdir.write("file.csv", text, "utf-8")
#
#    def tearDown(self):
#        self.tempdir.cleanup()
#
#    def test_parse_file(self):
#        output = utils.parse_file(self.path, 1/5)
#        assert output is not None
#        self.assertAlmostEqual(output[1], 6/(4*5))
#        self.assertAlmostEqual(output[2], 10)
#
#        output = utils.parse_file(self.path, 2/5)
#        self.assertAlmostEqual(output[2], 20)
#

def test_gridsearch():
    called_params = []
    def foo(**kwargs):
        called_params.append(kwargs)
    params = {
            'a': [1,2,3],
            'b': [4,5,6]
    }
    funcs = utils.gridsearch(params, foo)
    utils.cc(funcs)

    assert len(called_params) == 3*3
    assert {'a': 1, 'b': 4} in called_params
    assert {'a': 1, 'b': 5} in called_params
    assert {'a': 1, 'b': 6} in called_params
    assert {'a': 2, 'b': 4} in called_params
    assert {'a': 2, 'b': 5} in called_params
    assert {'a': 2, 'b': 6} in called_params
    assert {'a': 3, 'b': 4} in called_params
    assert {'a': 3, 'b': 5} in called_params
    assert {'a': 3, 'b': 6} in called_params

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
    utils.save_results({'a': 1}, [1,2,3], directory=str(tmpdir))
    assert len(tmpdir.listdir()) == 1
    utils.save_results(None, None, directory=str(tmpdir))
    assert len(tmpdir.listdir()) == 2

def test_save_results_and_get_results(tmpdir):
    utils.save_results({'a': 1}, [1,2,3], directory=str(tmpdir))
    utils.save_results({'b': 1}, [4,5,6], directory=str(tmpdir))

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
    utils.save_results({'a': 1, 'b': 1}, [1,2,3], directory=str(tmpdir))
    utils.save_results({'b': 1}, [4,5,6], directory=str(tmpdir))

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
