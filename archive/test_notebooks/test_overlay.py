''' Unit tests for trajectory_visualization.py '''

import numpy as np
import unittest
import os
from trajectory_visualization import overlay


class TestOverlay(unittest.TestCase):

    # Tests if both datasets were successfully plotted.
    def testPlotTrue(self):
        a = np.arange(9).reshape(3, 3)
        b = np.arange(12).reshape(4, 3)
        result = overlay(a.astype(np.float64), b.astype(np.float64), 'Plot 1')
        self.assertTrue(result)

    # Tests the error message for datasets that don't have correct number of
    # columns.
    def testJustRightFalse(self):
        a = np.arange(9).reshape(3, 3)
        b = np.arange(1).reshape(1, 1)
        c = np.arange(49).reshape(7, 7)

        result = overlay(a.astype(np.float64), b.astype(np.float64), 'Plot 1')
        self.assertFalse(result)

        result2 = overlay(b.astype(np.float64), a.astype(np.float64), 'Plot 1')
        self.assertFalse(result2)

        result3 = overlay(b.astype(np.float64), b.astype(np.float64), 'Plot 1')
        self.assertFalse(result3)

        result4 = overlay(a.astype(np.float64), c.astype(np.float64), 'Plot 1')
        self.assertFalse(result4)

        result5 = overlay(c.astype(np.float64), a.astype(np.float64), 'Plot 1')
        self.assertFalse(result5)

        result6 = overlay(c.astype(np.float64), c.astype(np.float64), 'Plot 1')
        self.assertFalse(result6)

if __name__ == '__main__':
    unittest.main()
