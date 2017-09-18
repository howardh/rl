import unittest
import numpy as np

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

