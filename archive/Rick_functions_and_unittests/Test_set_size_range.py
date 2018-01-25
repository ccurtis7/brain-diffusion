''' Unit tests for set_size_range '''

import unittest
import os
import pandas as pd
import zipfile
from functions_rick import obtain_data
from functions_rick import set_size_range


class Test_set_size_range(unittest.TestCase):

    def testPropHasColumnSize(self):
        deff, prop = obtain_data()
        self.assertTrue('Size' in prop.columns)

    def testsizesCreated(self):
        sizes = set_size_range(100, 10)
        self.assertTrue(isinstance(sizes, pd.DataFrame))

if __name__ == '__main__':
    unittest.main()
