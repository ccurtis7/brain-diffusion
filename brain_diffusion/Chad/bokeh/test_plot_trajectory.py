''' Unit tests for trajectory_visualization.py '''

import numpy as np
import unittest
import os
from trajectory_visualization import plot_trajectory


class TestPlotTrajectory(unittest.TestCase):

    # Tests if both datasets were successfully plotted.
    def testPlotTrue(self):
        a = np.arange(9).reshape(3, 3)
        result = plot_trajectory(a.astype(np.float64), 'Plot 1')
        self.assertTrue(result)

    # Tests the error message for datasets that don't have correct number of
    # columns.
    def testJustRightFalse(self):
        a = np.arange(9).reshape(3, 3)
        b = np.arange(1).reshape(1, 1)
        c = np.arange(49).reshape(7, 7)

        result = plot_trajectory(b.astype(np.float64), 'Plot 1')
        self.assertFalse(result)

        result2 = plot_trajectory(c.astype(np.float64), 'Plot 1')
        self.assertFalse(result2)


if __name__ == '__main__':
    unittest.main()
