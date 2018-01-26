import unittest
import numpy as np
import scipy.sparse
import itertools

import testfixtures

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

    def test_collect_file_params(self):
        a = [1,2,3]
        b = [0.1,0.2,0.3]
        c = [15,2000,101,0]
        file_names = ["a%d-b%f-c%d-0.csv" % (x,y,z) for x,y,z in itertools.product(a,b,c)]
        params = utils.collect_file_params(file_names)
        self.assertEqual(set(a),set([eval(x) for x in params['a']]))
        self.assertEqual(set(b),set([eval(x) for x in params['b']]))
        self.assertEqual(set(c),set([eval(x) for x in params['c']]))

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

class TestUtilsFile(unittest.TestCase):
    def setUp(self):
        self.tempdir = testfixtures.TempDirectory()
        text = ['0,"[0,0,0,0,0]"',
                '10,"[1,0,0,0,0]"',
                '20,"[0,1,0,1,0]"',
                '30,"[0,1,1,0,1]"'
        ]
        text = "\n".join(text)
        self.path = self.tempdir.write("file.txt", text, "utf-8")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_parse_file(self):
        output = utils.parse_file(self.path, 1/5)
        self.assertAlmostEqual(output[0], 6/(4*5))
        self.assertAlmostEqual(output[1], 10)

        output = utils.parse_file(self.path, 2/5)
        self.assertAlmostEqual(output[1], 20)

