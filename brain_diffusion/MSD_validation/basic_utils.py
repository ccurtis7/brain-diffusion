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


def fillin2(data):
    """
    Fills in blanks of arrays without shifting frames by the starting frame.  Compare to fillin.

    Input: trajectory dataset from MOSAIC tracking software read into a numpy array
    Output: modified numpy array with missing frames filled in.
    """

    shap = int(max(data[:, 1])) + 1
    shape1 = int(min(data[:, 1]))
    newshap = shap - shape1
    filledin = np.zeros((newshap, 5))
    filledin[0, :] = data[0, :]

    count = 0
    new = 0
    other = 0
    tot = 0

    for num in range(1, newshap):
        if data[num-new, 1]-data[num-new-1, 1] == 1:
            count = count + 1
            filledin[num, :] = data[num-new, :]
        elif data[num - new, 1]-data[num - new - 1, 1] > 1:
            new = new + 1
            iba = int(data[num - new+1, 1]-data[num - new, 1])
            togoin = data[num - new]
            togoin[1] = togoin[1] + 1
            filledin[num, :] = togoin
            # dataset[2] = np.insert(dataset[2], [num + new - 2], togoin, axis=0)

        else:
            other = other + 1
        tot = count + new + other

    return filledin


def MSD_iteration(folder, name, cut, totvids, conversion, frames):
    """
    Cleans up data for MSD analysis from csv files.  Outputs in form of
    dictionaries.
    """

    trajectory = dict()
    tots = dict()  # Total particles in each video
    newtots = dict()  # Cumulative total particles.
    newtots[0] = 0
    tlen = dict()
    tlength = dict()
    tlength[0] = 0

    for num in range(1, totvids + 1):
        trajectory[num] = np.genfromtxt(folder+'Traj_{}_{}.tif.csv'.format(name, num), delimiter=",")
        trajectory[num] = np.delete(trajectory[num], 0, 1)

        tots[num] = trajectory[num][-1, 0].astype(np.int64)
        newtots[num] = newtots[num-1] + tots[num]

        tlen[num] = trajectory[num].shape[0]
        tlength[num] = tlength[num-1] + tlen[num]

    placeholder = np.zeros((tlength[totvids], 11))

    for num in range(1, totvids + 1):
        placeholder[tlength[num-1]:tlength[num], :] = trajectory[num]
        placeholder[tlength[num-1]:tlength[num], 0] = placeholder[tlength[num-1]:tlength[num], 0] + newtots[num-1]

    dataset = dict()
    rawdataset = np.zeros(placeholder.shape)
    particles = placeholder[:, 0]
    total = int(max(particles))
    total1 = total + 1
    rawdataset = placeholder[:, :]

    fixed = np.zeros(placeholder.shape)
    fixed[:, 0:2] = rawdataset[:, 0:2]
    fixed[:, 2:4] = conversion[0] * rawdataset[:, 2:4]
    fixed[:, 4] = conversion[2] * rawdataset[:, 4]

    x = np.zeros((frames, total1 - 1))
    y = np.zeros((frames, total1 - 1))
    xs = np.zeros((frames, total1 - 1))
    ys = np.zeros((frames, total1 - 1))

    nones = 0
    cutoff = cut
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)

        if max1 - min1 < cutoff:
            nones = nones + 1
        else:
            holdplease = fillin2(fixed[min1:max1+1, 0:5])
            x[int(holdplease[0, 1]):int(holdplease[-1, 1])+1, num - nones - 1] = holdplease[:, 2]
            y[int(holdplease[0, 1]):int(holdplease[-1, 1])+1, num - nones - 1] = holdplease[:, 3]

            xs[0:int(holdplease[-1, 1])+1-int(holdplease[0, 1]), num - nones - 1] = holdplease[:, 2]
            ys[0:int(holdplease[-1, 1])+1-int(holdplease[0, 1]), num - nones - 1] = holdplease[:, 3]

    total1 = total1 - nones - 1
    x_m = x[:, :total1-1]
    y_m = y[:, :total1-1]
    xs_m = xs[:, :total1-1]
    ys_m = ys[:, :total1-1]

    return total1, xs_m, ys_m, x_m, y_m


def vectorized_MMSD_calcs(frames, total1, xs_m, ys_m, x_m, y_m, frame_m):

    SM1x = np.zeros((frames, total1-1))
    SM1y = np.zeros((frames, total1-1))
    SM2xy = np.zeros((frames, total1-1))

    xs_m = ma.masked_equal(xs_m, 0)
    ys_m = ma.masked_equal(ys_m, 0)

    x_m = ma.masked_equal(x_m, 0)
    y_m = ma.masked_equal(y_m, 0)

    geoM1x = np.zeros(frame_m)
    geoM1y = np.zeros(frame_m)

    for frame in range(1, frame_m):
        bx = xs_m[frame, :]
        cx = xs_m[:-frame, :]
        Mx = (bx - cx)**2

        Mxa = np.mean(Mx, axis=0)
        # Mxab = np.mean(np.log(Mxa), axis=0)

        # geoM1x[frame] = Mxab

        by = ys_m[frame, :]
        cy = ys_m[:-frame, :]
        My = (by - cy)**2

        Mya = np.mean(My, axis=0)
        # Myab = np.mean(np.log(Mya), axis=0)

        # geoM1y[frame] = Myab
        SM1x[frame, :] = Mxa
        SM1y[frame, :] = Mya

    geoM2xy = np.mean(np.log(Mya+Mxa), axix=0)
    gSEM = stat.sem(np.log(Mya+Mxa), axis=0)
    SM2xy = SM1x + SM1y

    return geoM2xy, gSEM, SM1x, SM1y, SM2xy
