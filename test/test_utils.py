import unittest
import numpy as np
import scipy.sparse

import utils

class TestUtils(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_parse_file_name(self):
        file_name = "g1.000-u200-eb0.100-et0.010-l0.010-621.csv"
        output = utils.parse_file_name(file_name)
        expectedOutput = {
                "g": "1.000",
                "u": "200",
                "eb": "0.100",
                "et": "0.010",
                "l": "0.010"}
        self.assertDictEqual(output, expectedOutput)

    def test_solve(self):
        a = scipy.sparse.lil_matrix(np.random.rand(5,5))
        b = scipy.sparse.lil_matrix(np.random.rand(5,1))
        x = utils.solve(a,b)
        diff = np.linalg.norm(a*x-b)
        self.assertAlmostEqual(diff,0)

    #def test_solve_approx(self):
    #    a = scipy.sparse.lil_matrix(np.random.rand(5,5))
    #    b = scipy.sparse.lil_matrix(np.random.rand(5,1))
    #    x = utils.solve_approx(a,b)
    #    diff = np.linalg.norm(a*x-b)
    #    self.assertAlmostEqual(diff,0,delta=.3,msg="Diff too large: %s" % diff)
