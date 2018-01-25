''' Unit tests for set_zp_range '''

import unittest
import os
import pandas as pd
import zipfile
from functions_rick import obtain_data
from functions_rick import set_zp_range


class Test_set_zp_range(unittest.TestCase):

    def testPropHasColumnZeta_Potential(self):
        deff, prop = obtain_data()
        self.assertTrue('Zeta_Potential' in prop.columns)

    def testzpCreated(self):
        zp = set_zp_range(5)
        self.assertTrue(isinstance(zp, pd.DataFrame))

if __name__ == '__main__':
    unittest.main()
