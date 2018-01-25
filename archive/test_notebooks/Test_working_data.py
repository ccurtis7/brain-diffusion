''' Unit tests for working_data '''

import unittest
import os
import pandas as pd
import zipfile
from functions_rick import obtain_data
from functions_rick import set_size_range
from functions_rick import set_zp_range
from functions_rick import prop_data
from functions_rick import working_data


class Test_working_data(unittest.TestCase):

    def testAllColumnsAdded(self):
        data = working_data(100, 10, 5)
        columns = ['Particle', 'Deff', 'Particle_Type', 'Surfactant', 'PEG',
                   'Size_Range', 'ZP_Range']
        match = True
        for x in range(0, len(columns)):
            if data.columns[x] != columns[x]:
                match = False
        self.assertTrue(match is True)

if __name__ == '__main__':
    unittest.main()
