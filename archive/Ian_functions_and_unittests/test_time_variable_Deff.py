"""Unit tests for time_variable_Deff.py"""

import numpy as np
import os
import pandas as pd
import sys
import unittest

from StringIO import StringIO

import time_variable_Deff


class TestDataCleaning(unittest.TestCase):


    def test_duplicate_removal(self):
        # Part of cleaning these data (and any data, since the
        # subsequent functions in this file require no duplicate
        # columns) is to create a list of particle chemistries with the
        # duplicates (originally there for biological replicates)
        # removed, yielding one column title for each chemistry.  This
        # test verifies that there are no duplicates in this list.
        vc = pd.Series(time_variable_Deff.columns2).value_counts()
        self.assertEqual(vc[vc > 1].index.tolist(), [])

    def test_msd_index_float(self):
        # Part of the data cleaning involves changing the timepoint
        # values, which will become the index of the msd dataframe,
        # from strings into floats.  This enables the downstream
        # timepoint cutoffs.  This test verifies that this change did
        # indeed occur.
        self.assertEqual(type(time_variable_Deff.msd.index[np.random.randint(
            len(time_variable_Deff.msd))]), np.float64
        )

    def test_geo_means_append(self):
        # A key part of the data cleaning is to compute and append
        # geometric means for each timepoint within each chemistry.
        # This test will make sure this append happened by simply
        # checking whether the final column in msd contains the string
        # 'geo', as it should if it was appended.
        columntitle = time_variable_Deff.msd.columns[len(
            time_variable_Deff.msd.columns)-1]
        self.assertTrue('geo' in columntitle)


class TestComputeHistDeff(unittest.TestCase):


    def test_hist_html_file_output(self):
        # This test will check if compute_hist_Deff() actually got to
        # the point of saving the figure to the directory, assuming
        # that everything must have functioned correctly up to that
        # point.
        # First, have to make sure the image file is freshly created.
        if os.path.exists('Deffs_hist.html'):
            os.unlink('Deffs_hist.html')
        time_variable_Deff.compute_hist_Deff('PLGA15k 0.5CHA', 2, 7)
        self.assertTrue(os.path.exists('Deffs_hist.html'))


class TestComputePlotAllDeff(unittest.TestCase):


    def test_hist_html_file_output(self):
        # This test will check if compute_plot_all_Deff() actually got
        # to the point of saving the figure to the directory, assuming
        # that everything must have functioned correctly up to that
        # point.
        # First, have to make sure the image file is freshly created.
        if os.path.exists('Deffs_hist_and_line_plot.html'):
            os.unlink('Deffs_hist_and_line_plot.html')
        time_variable_Deff.compute_hist_Deff(2, 7, 'PLGA15k 0.5CHA')
        self.assertTrue(os.path.exists('Deffs_hist_and_line_plot.html'))


if __name__ == '__main__':
    unittest.main()
