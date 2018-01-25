''' Unit tests for trajectory_visualization.py '''

import numpy as np
import unittest
import os
from trajectory_visualization import download_trajectory_data, csv_writer


class TestDownloadTrajectory(unittest.TestCase):

    # Test whether csv file was downloaded successfully. Requires a dataset
    # called unit_test_data.csv within the same folder.
    def testDownloadSuccessfulTrue(self):
        # Test csv File
        a = np.arange(220).reshape(10, 22)
        csv_writer()
        name = download_trajectory_data('output.csv')
        result = type(name) == type(a)
        self.assertTrue(result)

    # Tests the error message for a nonexistent file.
    def testFileExistsFalse(self):
        # Test csv File
        result = download_trajectory_data('nOnSeNsE.csv')
        self.assertFalse(result[1])

    # Tests the error message for a non-csv file
    def testcsvFileFalse(self):
        # Test csv File
        result = download_trajectory_data('data_file.py')
        self.assertFalse(result[0])


if __name__ == '__main__':
    unittest.main()
