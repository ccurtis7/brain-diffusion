''' Unit tests for obtain_data '''

import unittest
import os
import pandas as pd
import zipfile
from functions_rick import obtain_data


class Test_obtain_data(unittest.TestCase):

    def testDeffAndPropCreated(self):
        deff, prop = obtain_data()
        self.assertTrue(isinstance(deff, pd.DataFrame) and
                        isinstance(prop, pd.DataFrame))

if __name__ == '__main__':
    unittest.main()
