''' Unit tests for trajectory_visualization.py '''

import numpy as np
import unittest
from trajectory_visualization import define_xydata


class TestSplitSuccessful(unittest.TestCase):

    # Test whether the data was successfully split into desired # of datasets.
    def testSplitSuccessfulTrue(self):
        # First test dataset
        a = np.arange(220).reshape(10, 22)
        time, Runs = define_xydata(a, 1)
        result = type(time) == type(a)
        self.assertTrue(result)

        # Second test dataset
        time1, Runs1 = define_xydata(a, 6)
        result2 = type(time) == type(a)
        self.assertTrue(result2)

    # Test error message for the case where set is not a whole number
    def testWholeNumberFalse(self):
        a = np.arange(220).reshape(10, 22)
        result = define_xydata(a, 1.5)
        self.assertFalse(result[1])

    # Test error message for the case where set is too large for dataset
    def testJustRightSetsFalse(self):
        a = np.arange(220).reshape(10, 22)
        result = define_xydata(a, 7)
        self.assertFalse(result[2])

    # Test error message for the case where dataset is not correct format
    def testCorrectSplitsFalse(self):
        a = np.arange(210).reshape(10, 21)
        result = define_xydata(a, 1)
        self.assertFalse(result[3])


if __name__ == '__main__':
    unittest.main()
