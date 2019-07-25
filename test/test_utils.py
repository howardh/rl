import unittest
import numpy as np
import scipy.sparse
import itertools

import utils

class TestUtils(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_solve(self):
        a = scipy.sparse.lil_matrix(np.random.rand(5,5))
        b = scipy.sparse.lil_matrix(np.random.rand(5,1))
        x = utils.solve(a,b)
        diff = np.linalg.norm(a*x-b)
        self.assertAlmostEqual(diff,0,6)

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
