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

from msd import fillin2, MSD_iteration, vectorized_MMSD_calcs


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


def test_MSD_iteration():
    n = 6
    p = 2
    df = np.zeros((p*n, 12))
    for i in range(1, p+1):
        df[(i-1)*n:i*n, 0] = np.ones(n) + i - 1
        df[(i-1)*n:i*n, 1] = np.ones(n) + i - 1
        df[(i-1)*n:i*n, 2] = np.linspace(0, 10, n) + 2 + i
        df[(i-1)*n:i*n, 3] = np.linspace(0, 10, n) + i
        df[(i-1)*n:i*n, 4] = np.linspace(0, 10, n) + 3 + i
        df[(i-1)*n:i*n, 5] = np.zeros(n)
        df[(i-1)*n:i*n, 6:12] = np.zeros((n, 6))
    np.savetxt("../Traj_test_data_1.tif.csv", df, delimiter=",")
    folder = '../'
    name = 'test_data'
    total1, frames, xs_m, ys_m, x_m, y_m = MSD_iteration(folder, name)

    test1 = np.array([[1.,   2.],
                      [1.,   2.],
                      [3.,   4.],
                      [3.,   4.],
                      [5.,   6.],
                      [5.,   6.],
                      [7.,   8.],
                      [7.,   8.],
                      [9.,  10.],
                      [9.,  10.],
                      [11.,  12.],
                      [0.,   0.],
                      [0.,   0.],
                      [0.,   0.],
                      [0.,   0.]])

    test2 = np.array([[4.,   5.],
                      [4.,   5.],
                      [6.,   7.],
                      [6.,   7.],
                      [8.,   9.],
                      [8.,   9.],
                      [10.,  11.],
                      [10.,  11.],
                      [12.,  13.],
                      [12.,  13.],
                      [14.,  15.],
                      [0.,   0.],
                      [0.,   0.],
                      [0.,   0.],
                      [0.,   0.]])

    test3 = np.array([[0.,   0.],
                      [0.,   0.],
                      [0.,   0.],
                      [1.,   0.],
                      [1.,   2.],
                      [3.,   2.],
                      [3.,   4.],
                      [5.,   4.],
                      [5.,   6.],
                      [7.,   6.],
                      [7.,   8.],
                      [9.,   8.],
                      [9.,  10.],
                      [11.,  10.],
                      [0.,  12.]])

    test4 = np.array([[0.,   0.],
                      [0.,   0.],
                      [0.,   0.],
                      [4.,   0.],
                      [4.,   5.],
                      [6.,   5.],
                      [6.,   7.],
                      [8.,   7.],
                      [8.,   9.],
                      [10.,   9.],
                      [10.,  11.],
                      [12.,  11.],
                      [12.,  13.],
                      [14.,  13.],
                      [0.,  15.]])

    assert total1 == p
    assert frames == 14
    npt.assert_equal(test1, xs_m)
    npt.assert_equal(test2, ys_m)
    npt.assert_equal(test3, x_m)
    npt.assert_equal(test4, y_m)


def test_vectorized_MMSD_calcs():
    n = 6
    p = 2
    df = np.zeros((p*n, 12))
    for i in range(1, p+1):
        df[(i-1)*n:i*n, 0] = np.ones(n) + i - 1
        df[(i-1)*n:i*n, 1] = np.ones(n) + i - 1
        df[(i-1)*n:i*n, 2] = np.linspace(0, 10, n) + 2 + i
        df[(i-1)*n:i*n, 3] = np.linspace(0, 10, n) + i
        df[(i-1)*n:i*n, 4] = np.linspace(0, 10, n) + 3 + i
        df[(i-1)*n:i*n, 5] = np.zeros(n)
        df[(i-1)*n:i*n, 6:12] = np.zeros((n, 6))
    np.savetxt("../Traj_test_data_1.tif.csv", df, delimiter=",")
    folder = '../'
    name = 'test_data'
    total1, frames, xs_m, ys_m, x_m, y_m = MSD_iteration(folder, name)
    geoM2xy, gSEM, SM1x, SM1y, SM2xy = vectorized_MMSD_calcs(frames, total1, xs_m, ys_m)

    test1 = np.array([0.,  1.38629436, 2.07944154, 2.99573227, 3.4657359,
                      3.95124372, 4.27666612, 4.60517019, 4.85203026, 5.09986643,
                      5.29831737,  0.,  0.,  0.])

    test2 = np.zeros(14)

    test3 = np.array([[0.,    0.],
                      [2.,    2.],
                      [4.,    4.],
                      [10.,   10.],
                      [16.,   16.],
                      [26.,   26.],
                      [36.,   36.],
                      [50.,   50.],
                      [64.,   64.],
                      [82.,   82.],
                      [100.,  100.],
                      [0.,    0.],
                      [0.,    0.],
                      [0.,    0.]])

    npt.assert_almost_equal(test1, geoM2xy)
    npt.assert_equal(test2, gSEM)
    npt.assert_equal(test3, SM1x)
    npt.assert_equal(test3, SM1y)
    npt.assert_equal(2*test3, SM2xy)
