import unittest
import numpy as np

from frozenlake import features

class TestFrozenLakeFeatures(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_one_hot(self):
        output = features.one_hot(0)
        expectedOutput = np.array(
                [[1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
        close = np.allclose(output, expectedOutput)
        self.assertTrue(close, "Expected: %s, Received: %s" %(expectedOutput, output))
        
        output = features.one_hot(7)
        expectedOutput = np.array(
                [[0], [0], [0], [0], [0], [0], [0], [1], [0], [0], [0], [0], [0], [0], [0], [0]])
        close = np.allclose(output, expectedOutput)
        self.assertTrue(close, "Expected: %s, Received: %s" %(expectedOutput, output))

        output = features.one_hot(15)
        expectedOutput = np.array(
                [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1]])
        close = np.allclose(output, expectedOutput)
        self.assertTrue(close, "Expected: %s, Received: %s" %(expectedOutput, output))

    def test_tile_coding(self):
        output = features.tile_coding(0)
        expectedOutput = np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [1]])
        close = np.allclose(output, expectedOutput)
        self.assertTrue(close, "Expected: %s, Received: %s" %(expectedOutput, output))

        output = features.tile_coding(1)
        expectedOutput = np.array([[0.5], [0.5], [0], [0], [0], [0], [0], [0], [0], [1]])
        close = np.allclose(output, expectedOutput)
        self.assertTrue(close, "Expected: %s, Received: %s" %(expectedOutput, output))

        output = features.tile_coding(5)
        expectedOutput = np.array([[.25], [.25], [0], [.25], [.25], [0], [0], [0], [0], [1]])
        close = np.allclose(output, expectedOutput)
        self.assertTrue(close, "Expected: %s, Received: %s" % (expectedOutput, output))

        output = features.tile_coding(15)
        expectedOutput = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [1], [1]])
        close = np.allclose(output, expectedOutput)
        self.assertTrue(close, "Expected: %s, Received: %s" %(expectedOutput, output))
