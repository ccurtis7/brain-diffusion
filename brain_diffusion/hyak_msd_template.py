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
from msd import fillin2, MSD_iteration, vectorized_MMSD_calcs


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

folder = "./"
path = "./geoM2xy_{sample_name}.csv"
conversion = (0.16, 20.08, 1)  # (0.3, 3.95, 1)
cut = 1
base = '37C_72pH'

parameters = {}
parameters["channels"] = ["RED"]
parameters["surface functionalities"] = ["nPEG"]
parameters["slices"] = ["S1", "S2"]
parameters["videos"] = [1, 2]
parameters["replicates"] = [1, 2, 3]

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
totvids = len(replicates)

################################################################################
check_rank = 1

for channel in channels:
    for surface_functionality in surface_functionalities:
        slice_counter = 0
        for slic in slices:
            for video in videos:
                if rank == check_rank:
                    sample_name = "{}_{}_{}_{}_{}".format(channel, surface_functionality, base, slic, video)
                    DIR = folder

                    total1, frames, xs, ys, x, y = MSD_iteration(DIR, sample_name, cut=cut, totvids=totvids, conversion=conversion)

                    geoM2xy[sample_name], gSEM[sample_name], SM1x[sample_name], SM1y[sample_name],\
                        SM2xy[sample_name] = vectorized_MMSD_calcs(frames, total1, xs, ys)
                    np.savetxt(DIR+'geoM2xy_{}.csv'.format(sample_name), geoM2xy[sample_name], delimiter=',')
                    np.savetxt(DIR+'gSEM_{}.csv'.format(sample_name), gSEM[sample_name], delimiter=',')
                    np.savetxt(DIR+'SM2xy_{}.csv'.format(sample_name), SM2xy[sample_name], delimiter=',')

                    slice_counter = slice_counter + 1
                check_rank = check_rank + 1
