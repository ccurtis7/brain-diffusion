''' Unit tests for trajectory_visualization.py '''

import numpy as np
import unittest
import os
from trajectory_visualization import shift_trajectory


class TestShiftTrajectory(unittest.TestCase):

    # Tests if the trajectory was successfully shifted.
    def testShiftTrajectoryTrue(self):
        a = np.arange(9).reshape(3, 3)
        result = shift_trajectory(a.astype(np.float64))
        self.assertTrue(result[0])
        self.assertTrue(result[1])

    # Tests the error message for non-numeric data.
    def testNumericFalse(self):
        charar = np.chararray((3, 3))
        charar[:] = 'a'
        result = shift_trajectory(charar)
        self.assertFalse(result[0])

    # Tests the error message when the number of columns in dataset is not 3.
    def testJustRightFalse(self):
        a = np.arange(1).reshape(1, 1)
        result = shift_trajectory(a.astype(np.float64))
        self.assertFalse(result[1])

        b = np.arange(49).reshape(7, 7)
        result2 = shift_trajectory(b.astype(np.float64))
        self.assertFalse(result[1])

if __name__ == '__main__':
    unittest.main()
