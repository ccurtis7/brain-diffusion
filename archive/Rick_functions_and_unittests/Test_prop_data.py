''' Unit tests for prop_data '''

import unittest
import os
import pandas as pd
import zipfile
from functions_rick import obtain_data
from functions_rick import set_size_range
from functions_rick import set_zp_range
from functions_rick import prop_data


class Test_prop_data(unittest.TestCase):

    def testSizeAndZPColumnsEntered(self):
        prop = prop_data(100, 10, 5)
        self.assertTrue(prop['Size_Range'][0] != 0 and
                        prop['ZP_Range'][0] != 0)

if __name__ == '__main__':
    unittest.main()
