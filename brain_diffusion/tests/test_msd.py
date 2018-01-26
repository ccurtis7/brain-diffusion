import os
import csv
import sys
import scipy.optimize as opt
import scipy.stats as stat
from operator import itemgetter
import random
import numpy as np
import numpy.ma as ma
import numpy.linalg as la
import numpy.testing as npt

from brain_diffusion.msd import fillin2, MSD_iteration, vectorized_MMSD_calcs


def test_fillin2():
    n = 6
    df = np.zeros((n, 5))
    df[:, 0] = np.ones(n)
    df[:, 1] = np.linspace(0, 10, n)
    df[:, 2] = np.linspace(0, 10, n)
    df[:, 3] = np.linspace(0, 10, n)
    df[:, 4] = np.zeros(n)
    fillin2(df)

    test = np.array([[1.,   0.,   0.,   0.,   0.],
                     [1.,   1.,   0.,   0.,   0.],
                     [1.,   2.,   2.,   2.,   0.],
                     [1.,   3.,   2.,   2.,   0.],
                     [1.,   4.,   4.,   4.,   0.],
                     [1.,   5.,   4.,   4.,   0.],
                     [1.,   6.,   6.,   6.,   0.],
                     [1.,   7.,   6.,   6.,   0.],
                     [1.,   8.,   8.,   8.,   0.],
                     [1.,   9.,   8.,   8.,   0.],
                     [1.,  10.,  10.,  10.,   0.]])

    npt.assert_equal(test, fillin2(df))
