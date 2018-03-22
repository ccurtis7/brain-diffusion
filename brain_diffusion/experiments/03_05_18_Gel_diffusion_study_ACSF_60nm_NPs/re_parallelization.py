import os
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import csv
import sys
import scipy.optimize as opt
import scipy.stats as stat
from operator import itemgetter
import random
import numpy as np
import numpy.ma as ma
import numpy.linalg as la

pi = np.pi
sin = np.sin
cos = np.cos

################################################################################


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


def MSD_iteration(folder, name, cut=1, totvids=1, conversion=(1, 1, 1)):

    assert type(folder) is str, 'folder must be a string'
    assert folder[-1] == '/', 'folder must end with a /'
    assert type(name) is str, 'name must be a string'
    assert 'Traj_{}_1.tif.csv'.format(name) in os.listdir(folder), 'folder must contain Traj_{}_1_.tif.csv'.format(name)
    assert type(cut) is int, 'cut must be an integer'
    assert type(totvids) is int, "totvids must be an integer"
    for i in range(1, totvids+1):
        assert 'Traj_{}_{}.tif.csv'.format(name, i) in os.listdir(folder), "folder must contain 'Traj_{}_{}_.tif.csv".format(name, i)
    assert type(conversion) is tuple, "conversion must be a tuple"
    assert len(conversion) == 3, "conversion must contain 3 elements"

    frames = 0
    trajectory = dict()
    tots = dict()  # Total particles in each video
    newtots = dict()  # Cumulative total particles.
    newtots[0] = 0
    tlen = dict()
    tlength = dict()
    tlength[0] = 0

    for num in range(1, totvids + 1):
        trajectory[num] = np.genfromtxt(folder+'Traj_{}_{}.tif.csv'.format(name, num), delimiter=",")
        trajectory[num] = np.delete(trajectory[num], 0, 0)
        trajectory[num] = np.delete(trajectory[num], 0, 1)

        tots[num] = trajectory[num][-1, 0].astype(np.int64)
        newtots[num] = newtots[num-1] + tots[num]

        tlen[num] = trajectory[num].shape[0]
        tlength[num] = tlength[num-1] + tlen[num]

        if np.max(trajectory[num][:, 1]) > frames:
            frames = int(np.max(trajectory[num][:, 1]))

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

    x = np.zeros((frames+1, total1))
    y = np.zeros((frames+1, total1))
    xs = np.zeros((frames+1, total1))
    ys = np.zeros((frames+1, total1))

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

    x_m = x[:, :total1]
    y_m = y[:, :total1]
    xs_m = xs[:, :total1]
    ys_m = ys[:, :total1]

    return total1, frames, xs_m, ys_m, x_m, y_m


def vectorized_MMSD_calcs(frames, total1, xs_m, ys_m):

    assert type(frames) is int, 'frames must be an integer'
    assert type(total1) is int, 'total1 must be an integer'
    assert type(xs_m) is np.ndarray, 'xs_m must be a numpy array'
    assert type(ys_m) is np.ndarray, 'ys_m must an a numpy array'
    assert xs_m.shape == ys_m.shape, 'xs_m and ys_m must be the same size'

    SM1x = np.zeros((frames, total1))
    SM1y = np.zeros((frames, total1))
    SM2xy = np.zeros((frames, total1))

    xs_m = ma.masked_equal(xs_m, 0)
    ys_m = ma.masked_equal(ys_m, 0)

    geoM1x = np.zeros(frames)
    geoM1y = np.zeros(frames)

    for frame in range(1, frames):
        bx = xs_m[frame:, :]
        cx = xs_m[:-frame, :]
        Mx = (bx - cx)**2
        Mxa = np.mean(Mx, axis=0)

        by = ys_m[frame:, :]
        cy = ys_m[:-frame, :]
        My = (by - cy)**2
        Mya = np.mean(My, axis=0)

        SM1x[frame, :] = Mxa
        SM1y[frame, :] = Mya

    SM2xy = SM1x + SM1y
    dist = ma.log(ma.masked_equal(SM2xy, 0))

    geoM2xy = ma.mean(dist, axis=1)
    gSEM = stat.sem(dist, axis=1)
    geoM2xy = geoM2xy.data

    return geoM2xy, gSEM, SM1x, SM1y, SM2xy

################################################################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

folder = "./"
path = "./geoM2xy_{sample_name}.csv"
frames = 70
SD_frames = [10, 20, 50, 80]
conversion = (0.10, 11.11, 1)  # (0.3, 3.95, 1)
to_frame = 60
dimension = "2D"
time_to_calculate = 1

base = "well"
base_name = "RED"
test_bins = np.linspace(0, 75, 76)

# name = 'RED_KO_PEG_P1_S1_cortex'
cut = 1
totvids = 9
frame_m = 70  # atm I can't go lower than the actual value.

parameters = {}
parameters["channels"] = ["RED"]
parameters["surface functionalities"] = ["nPEG"]
parameters["slices"] = [1, 2, 3, 4]
parameters["videos"] = [1, 2, 3, 4, 5]
parameters["replicates"] = [1, 2, 3, 4, 5, 6, 7, 8, 9]


channels = parameters["channels"]
surface_functionalities = parameters["surface functionalities"]
slices = parameters["slices"]
videos = parameters["videos"]
replicates = parameters["replicates"]

geoM2xy = {}
gSEM = {}
SM1x = {}
SM1y = {}
SM2xy = {}

################################################################################

check_rank = 1

for channel in channels:
    for surface_functionality in surface_functionalities:
        slice_counter = 0
        for slic in slices:
            for video in videos:
                if rank == check_rank:
                    suffix = suffixes[slice_counter]
                    sample_name = "well{}_XY{}".format(slic, video)
                    DIR = folder

                    total1, frames, xs_m, ys_m, x_m, ys_m = MSD_iteration(DIR, sample_name, cut, totvids, conversion)

                    geoM2xy[sample_name], gSEM[sample_name], SM1x[sample_name], SM1y[sample_name],\
                        SM2xy[sample_name] = vectorized_MMSD_calcs(frames, total1, xs_m, ys_m)
                    np.savetxt(DIR+'geoM2xy_{}.csv'.format(sample_name), geoM2xy[sample_name], delimiter=',')
                    np.savetxt(DIR+'gSEM_{}.csv'.format(sample_name), gSEM[sample_name], delimiter=',')
                    np.savetxt(DIR+'SM2xy_{}.csv'.format(sample_name), SM2xy[sample_name], delimiter=',')

        #                 geoM2xy[sample_name] = np.genfromtxt(DIR + 'geoM2xy_{}.csv'.format(sample_name), delimiter=",");
        #                 SM2xy[sample_name] = np.genfromtxt(DIR + "SM2xy_{}.csv".format(sample_name), delimiter=",");

                    slice_counter = slice_counter + 1
                check_rank = check_rank + 1
