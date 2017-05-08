import unittest
import numpy as np

from frozenlake8x8 import features

class TestFrozenLakeFeatures(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_generate_tiles(self):
        expected_output = [
                [0,1,4,5],
                [1,2,5,6],
                [2,3,6,7],
                [4,5,8,9],
                [5,6,9,10],
                [6,7,10,11],
                [8,9,12,13],
                [9,10,13,14],
                [10,11,14,15]
        ]
        x = features.generate_tiles((4,4),(2,2))
        self.assertEqual(len(x), len(expected_output), "Wrong number of entries. %s" % x)
        self.assertListEqual(x,expected_output)
