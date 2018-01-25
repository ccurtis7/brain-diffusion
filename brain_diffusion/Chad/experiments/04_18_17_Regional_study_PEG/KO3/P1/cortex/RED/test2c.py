import os
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy.optimize as opt
import scipy.stats as stat
from mpl_toolkits.mplot3d import Axes3D
from operator import itemgetter
import random
import numpy as np
import numpy.ma as ma
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D

pi = np.pi
sin = np.sin
cos = np.cos

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

name = 'RED_KO_PEG_P1_S3_cortex'

logplot = '{}_logplot'.format(name)
Mplot = '{}_Mplot'.format(name)
Dplot = '{}_Dplot'.format(name)
Hplot = '{}_Hplot'.format(name)
Hlogplot = '{}_Hlogplot'.format(name)
Cplot = '{}_Cplot'.format(name)
Tplot = '{}_Tplot'.format(name)
T2plot = '{}_T2plot'.format(name)

conversion = dict()
trajectory = dict()

cut = 4
great = 10000
filtered = False
new_method = True
totvids = 5
tofilt = np.array([])

tau_m = 89
frame_m = 90 #atm I can't go lower than the actual value.

for num in range(1, totvids + 1):
    conversion[num] = (0.3, 3.95, 1)
    trajectory[num] = np.genfromtxt('./Traj_{}_{}.tif.csv'.format(name, num), delimiter =",")
    trajectory[num]=np.delete(trajectory[num],0,1)

parts = dict()
tots = dict()
newtots = dict()
newtots[0] = 0
tlen = dict()
tlength = dict()
tlength[0] = 0

for num in range(1, totvids + 1):
    tots[num] = trajectory[num][-1, 0].astype(np.int64)
    parts[num] = tots[num]
    counter = 1
    newtots[num] = newtots[num-1] + tots[num]

    tlen[num] = trajectory[num].shape[0]
    tlength[num] = tlength[num-1] + tlen[num]

placeholder = np.zeros((tlength[totvids], 11))

for num in range(1, totvids + 1):
    placeholder[tlength[num-1]:tlength[num], :] = trajectory[num]
    placeholder[tlength[num-1]:tlength[num], 0] = placeholder[tlength[num-1]:tlength[num], 0] + newtots[num-1]

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

rawframes = placeholder[:, 1]
frames = np.linspace(min(rawframes),max(rawframes), max(rawframes)+1).astype(np.int64)
time = frames / conversion[1][1]

x = dict()
y = dict()

xs = dict()
ys = dict()

# M1x = dict() # MSD dictionaries (without shifting)
# M1y = dict()
# M2xy = dict()

SM1x = dict() # Shifted MSD dictionaries.
SM1y = dict()
SM2xy = dict()

SD1x = dict() # Shifted diffusion coefficient dictionaries.
SD1y = dict()
SD2xy = dict()

dataset = dict()
rawdataset = np.zeros(placeholder.shape)
particles = placeholder[:, 0]
total = int(max(particles))
total1 = total + 1
rawdataset = placeholder[:, :]

fixed = np.zeros(placeholder.shape)
fixed[:, 0:2] = rawdataset[:, 0:2]
fixed[:, 2:4] = conversion[1][0] * rawdataset[:, 2:4]
fixed[:, 4] = conversion[1][2] * rawdataset[:, 4]

for num in range(1, total1):

    hold = np.where(particles == num)
    itindex = hold[0]
    min1 = min(itindex)
    max1 = max(itindex)
    dataset[num] = (fixed[min1:max1+1, 0:5])

I = dict()

for num in range(1, total1):
    #Construct x, y, z
    dataset[num] = fillin2(dataset[num])
    x[num] = np.zeros(frames.shape[0])
    x[num][int(dataset[num][0,1]):int(dataset[num][-1,1])+1] = dataset[num][:, 2]
    y[num] = np.zeros(frames.shape[0])
    y[num][int(dataset[num][0,1]):int(dataset[num][-1,1])+1] = dataset[num][:, 3]

    xs[num] = np.zeros(frames.shape[0])
    xs[num][0:int(dataset[num][-1,1])+1-int(dataset[num][0,1])] = dataset[num][:, 2]
    ys[num] = np.zeros(frames.shape[0])
    ys[num][0:int(dataset[num][-1,1])+1-int(dataset[num][0,1])] = dataset[num][:, 3]

cutoff = cut

x1 = dict()
y1 = dict()

xs1 = dict()
ys1 = dict()

fifties = 0
nones = 0

for num in range(1, total1):
    if np.count_nonzero(x[num]) < cutoff:
        nones = nones + 1
    else:
        fifties = fifties + 1
        x1[num - nones] = x[num]
        y1[num - nones] = y[num]

        xs1[num - nones] = xs[num]
        ys1[num - nones] = ys[num]

x = x1
y = y1

xs = xs1
ys = ys1

for num in range(1, fifties):
    xs[num] = ma.masked_equal(xs[num], 0)
    ys[num] = ma.masked_equal(ys[num], 0)

    x[num] = ma.masked_equal(x[num], 0)
    y[num] = ma.masked_equal(y[num], 0)

total1 = fifties + 1

print('Total particles after merging datasets and filtering short trajectories:', fifties)

xymask = dict()

# Intermediates for new method
m1xa = dict()
m1ya = dict()
m2xya = dict()

a1x = np.zeros((total1, frame_m))
a1y = np.zeros((total1, frame_m))
a2xy = np.zeros((total1, frame_m))

# Portion of code where preemption and parallelization are important
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# Parallelized MSD dictionaries and parameters
pSM1x = dict() # MSD dictionaries (without shifting)
pSM1y = dict()
pSM2xy = dict()

quotient = divmod(total1, size)
local_n = quotient[0]
particles_lost = quotient[1]

exists = os.path.isfile('pM1xc_{}.csv'.format(rank))

if exists == False:
    last_size = 1
    with open('pM1xc_{}.csv'.format(rank), "wb") as f_handle:
        pass
    with open('pM1yc_{}.csv'.format(rank), "wb") as f_handle:
        pass
    with open('pM2xyc_{}.csv'.format(rank), "wb") as f_handle:
        pass
else:
    with open('pM1xc_{}.csv'.format(rank), "rb") as f_handle:
        reader = np.genfromtxt(f_handle, delimiter = ",")
        data = list(reader)
        last_size = data[0].shape[0]

for num in range(last_size, local_n+1):

    current = local_n*rank + num

    # M1x[num] = np.zeros(frame_m)
    # M1x[num][0:frame_m] = 0
    # M1y[num] = np.zeros(frame_m)
    # M1y[num][0:frame_m] = 0
    # M2xy[num] = np.zeros(frame_m)
    # M2xy[num][0:frame_m] = 0

    pSM1x[num] = np.zeros(frame_m)
    pSM1y[num] = np.zeros(frame_m)
    pSM2xy[num] = np.zeros(frame_m)

    SD1x[num] = np.zeros(frame_m)
    SD1y[num] = np.zeros(frame_m)
    SD2xy[num] = np.zeros(frame_m)

    # I[num] = np.nonzero(x[num])[0]
    # first = I[num][0]
    # last = I[num][-1] + 1
    # startx = x[num][first]
    # starty = y[num][first]

    if new_method == True:

        m1xa[num] = dict()
        m1ya[num] = dict()
        m2xya[num] = dict()

        for num1 in range(1, tau_m):

            tau = num1

            m1xa[num][num1] = np.zeros(frame_m - tau)
            m1ya[num][num1] = np.zeros(frame_m - tau)
            m2xya[num][num1] = np.zeros(frame_m - tau)

            for num2 in range(0, frame_m - tau):
                m1xa[num][num1][num2] = (xs[current][num2 + tau] - xs[current][num2])**2
                m1ya[num][num1][num2] = (ys[current][num2 + tau] - ys[current][num2])**2
                m2xya[num][num1][num2] = m1xa[num][num1][num2] + m1ya[num][num1][num2]

            m1xa[num][num1] = ma.masked_invalid(m1xa[num][num1])
            m1ya[num][num1] = ma.masked_invalid(m1ya[num][num1])
            m2xya[num][num1] = ma.masked_invalid(m2xya[num][num1])

            pSM1x[num][num1] = np.mean(m1xa[num][num1])
            pSM1y[num][num1] = np.mean(m1ya[num][num1])
            pSM2xy[num][num1] = np.mean(m2xya[num][num1])

#         SM1x[num] = ma.masked_invalid(SM1x[num])
#         SM1y[num] = ma.masked_invalid(SM1y[num])
#         SM2xy[num] = ma.masked_invalid(SM2xy[num])

    #     M1x[num][first:] = pSM1x[num][:frame_m-first]
    #     M1y[num][first:] = pSM1y[num][:frame_m-first]
    #     M2xy[num][first:] = pSM2xy[num][:frame_m-first]
    #
    #     M1x[num] = ma.masked_invalid(M1x[num])
    #     M1y[num] = ma.masked_invalid(M1y[num])
    #     M2xy[num] = ma.masked_invalid(M2xy[num])
    #
    # M1x[num] = ma.masked_equal(M1x[num], 0)
    # M1y[num] = ma.masked_equal(M1y[num], 0)
    # M2xy[num] = ma.masked_equal(M2xy[num], 0)

    # a1x[num-1, :] = pSM1x[num]
    # a1y[num-1, :] = pSM1y[num]
    # a2xy[num-1, :] = pSM2xy[num]
    #
    # a1x[num] = ma.masked_equal(a1x[num], 0)
    # a1y[num] = ma.masked_equal(a1y[num], 0)
    # a2xy[num] = ma.masked_equal(a2xy[num], 0)
    #
    # a1x[num] = ma.masked_invalid(a1x[num])
    # a1y[num] = ma.masked_invalid(a1y[num])
    # a2xy[num] = ma.masked_invalid(a2xy[num])

    if num == 1:
        with open('pM1xc_{}.csv'.format(rank), "wb") as f_handle:
            np.savetxt(f_handle, pSM1x[num], delimiter=",")

        with open('pM1yc_{}.csv'.format(rank), "wb") as f_handle:
            np.savetxt(f_handle, pSM1y[num], delimiter=",")

        with open('pM2xyc_{}.csv'.format(rank), "wb") as f_handle:
            np.savetxt(f_handle, pSM2xy[num], delimiter=",")

    else:
        with open('pM1xc_{}.csv'.format(rank), "rb") as f_handle:
            old_one = np.genfromtxt(f_handle, delimiter=",")
        new = np.column_stack((old_one, pSM1x[num]))
        with open('pM1xc_{}.csv'.format(rank), "wb") as f_handle:
            np.savetxt(f_handle, new, delimiter=",")

        with open('pM1yc_{}.csv'.format(rank), "rb") as f_handle:
            old_one1 = np.genfromtxt(f_handle, delimiter=",")
        new1 = np.column_stack((old_one1, pSM1y[num]))
        with open('pM1yc_{}.csv'.format(rank), "wb") as f_handle:
            np.savetxt(f_handle, new1, delimiter=",")

        with open('pM2xyc_{}.csv'.format(rank), "rb") as f_handle:
            old_one2 = np.genfromtxt(f_handle, delimiter=",")
        new2 = np.column_stack((old_one2, pSM2xy[num]))
        with open('pM2xyc_{}.csv'.format(rank), "wb") as f_handle:
            np.savetxt(f_handle, new2, delimiter=",")


SM1x = dict()
SM1y = dict()
SM2xy = dict()

if rank == 0:
    for num in range(0, size):
        with open('pM1xc_{}.csv'.format(num), "rb") as f_handle:
            interim = np.genfromtxt(f_handle, delimiter=",")
        for i in range(1, local_n+1):
            current = local_n*num + i
            SM1x[current] = interim[:, i-1]
        with open('pM1yc_{}.csv'.format(num), "rb") as f_handle:
            interim1 = np.genfromtxt(f_handle, delimiter=",")
        for i in range(1, local_n+1):
            current = local_n*num + i
            SM1y[current] = interim1[:, i-1]
        with open('pM2xyc_{}.csv'.format(num), "rb") as f_handle:
            interim2 = np.genfromtxt(f_handle, delimiter=",")
        for i in range(1, local_n+1):
            current = local_n*num + i
            SM2xy[current] = interim2[:, i-1]

# with open('pM1x_{}.csv'.format(rank), "rb") as f_handle:
#     test = np.genfromtxt(f_handle, delimiter=",")
# with open('pM1y_{}.csv'.format(rank), "rb") as f_handle:
#     test1 = np.genfromtxt(f_handle, delimiter=",")
# with open('pM2xy_{}.csv'.format(rank), "rb") as f_handle:
#     test2 = np.genfromtxt(f_handle, delimiter=",")
#
# pSM1x = dict()
# pSM1y = dict()
# pSM2xy = dict()
#
# for num in range(1, local_n):
#     pSM1x[num] = test[:, num - 1]
#     pSM1y[num] = test1[:, num - 1]
#     pSM2xy[num] = test2[:, num - 1]
#
#     pSM1x[num] = ma.masked_equal(pSM1x[num], 0)
#     pSM1y[num] = ma.masked_equal(pSM1y[num], 0)
#     pSM2xy[num] = ma.masked_equal(pSM2xy[num], 0)
#
#     pSM1x[num] = ma.masked_invalid(pSM1x[num])
#     pSM1y[num] = ma.masked_invalid(pSM1y[num])
#     pSM2xy[num] = ma.masked_invalid(pSM2xy[num])
#
#     pSM1x[num] = pSM1x[num].filled(0)
#     pSM1y[num] = pSM1y[num].filled(0)
#     pSM2xy[num] = pSM2xy[num].filled(0)
#
#
# SM1x = dict()
# SM1y = dict()
# SM2xy = dict()
#
# package = dict()
# package[1] = pSM1x
# package[2] = pSM1y
# package[3] = pSM2xy
#
# comm.Barrier()
#
# buffer_1 = dict()
# buffer_1[1] = dict()
# buffer_1[2] = dict()
# buffer_1[3] = dict()
#
# for num in range(1, local_n):
#     buffer_1[1][num] = np.zeros(frame_m)
#     buffer_1[2][num] = np.zeros(frame_m)
#     buffer_1[3][num] = np.zeros(frame_m)
#
# # Recombining parallel code into serial form.  Only operate on rank=0
# # ------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------
#
# if rank == 0:
#     for num in range(1, local_n):
#         SM1x[num] = pSM1x[num]
#         SM1y[num] = pSM1y[num]
#         SM2xy[num] = pSM2xy[num]
#     for i in range(1, size):
#         comm.Recv(buffer_1, ANY_SOURCE)
#         bottom = (i*local_n) + 1
#         top = (i+1)*local_n
#         counter = 1
#         for num in range(bottom, top):
#             SM1x[num] = buffer_1[1][counter]
#             SM1y[num] = buffer_1[2][counter]
#             SM2xy[num] = buffer_1[3][counter]
#             counter = counter + 1
# else:
#     comm.Send(package, dest=0)

total_check = local_n*size

if rank == 0:
        # xymask[num] = SM2xy[num].recordmask
        # xs[num] = ma.array(xs[num][:frame_m], mask = xymask[num])
        # ys[num] = ma.array(ys[num][:frame_m], mask = xymask[num])
        #
        # xs[num] = ma.masked_equal(xs[num], 0)
        # ys[num] = ma.masked_equal(ys[num], 0)

    arM1x = np.zeros(SM1x[1].shape[0])
    arM1y = np.zeros(SM1x[1].shape[0])
    arM2xy = np.zeros(SM1x[1].shape[0])

    arD1x = np.zeros(SM1x[1].shape[0])
    arD1y = np.zeros(SM1x[1].shape[0])
    arD2xy = np.zeros(SM1x[1].shape[0])

    st_arM1x = np.zeros(SM1x[1].shape[0])
    st_arM1y = np.zeros(SM1x[1].shape[0])
    st_arM2xy = np.zeros(SM1x[1].shape[0])

    st_arD1x = np.zeros(SM1x[1].shape[0])
    st_arD1y = np.zeros(SM1x[1].shape[0])
    st_arD2xy = np.zeros(SM1x[1].shape[0])

    tots = total_check
    time2 = time

    gM1x = dict()
    gM1y = dict()
    gM2xy = dict()

    log_gM1x = dict()
    log_gM1y = dict()
    log_gM2xy = dict()

    geoM1x = np.zeros(SM1x[1].shape[0])
    geoM1y = np.zeros(SM1x[1].shape[0])
    geoM2xy = np.zeros(SM1x[1].shape[0])

    st_geoM1x = np.zeros(SM1x[1].shape[0])
    st_geoM1y = np.zeros(SM1x[1].shape[0])
    st_geoM2xy = np.zeros(SM1x[1].shape[0])

    geoD1x = np.zeros(SM1x[1].shape[0])
    geoD1y = np.zeros(SM1x[1].shape[0])
    geoD2xy = np.zeros(SM1x[1].shape[0])

    st_geoD1x = np.zeros(SM1x[1].shape[0])
    st_geoD1y = np.zeros(SM1x[1].shape[0])
    st_geoD2xy = np.zeros(SM1x[1].shape[0])

    for num2 in range(0, SM1x[1].shape[0]):
        gM1x[num2+1] = np.zeros(tots)
        gM1y[num2+1] = np.zeros(tots)
        gM2xy[num2+1] = np.zeros(tots)

        for num in range(1, tots+1):
            gM1x[num2+1][num-1] = SM1x[num][num2]
            gM1y[num2+1][num-1] = SM1y[num][num2]
            gM2xy[num2+1][num-1] = SM2xy[num][num2]

        gM1x[num2+1] = ma.masked_invalid(gM1x[num2+1])
        gM1y[num2+1] = ma.masked_invalid(gM1y[num2+1])
        gM2xy[num2+1] = ma.masked_invalid(gM2xy[num2+1])

        gM1x[num2+1] = ma.masked_equal(gM1x[num2+1], 0)
        gM1y[num2+1] = ma.masked_equal(gM1y[num2+1], 0)
        gM2xy[num2+1] = ma.masked_equal(gM2xy[num2+1], 0)

        log_gM1x[num2+1] = np.log(gM1x[num2+1])
        log_gM1y[num2+1] = np.log(gM1y[num2+1])
        log_gM2xy[num2+1] = np.log(gM2xy[num2+1])

        geoM1x[num2] = stat.gmean(gM1x[num2+1])
        geoM1y[num2] = stat.gmean(gM1y[num2+1])
        geoM2xy[num2] = stat.gmean(gM2xy[num2+1])

        st_geoM1x[num2] = np.abs(geoM1x[num2]-np.exp(np.mean(np.log(gM1x[num2+1]))-np.std(np.log(gM1x[num2+1]))/np.sqrt(gM1x[num2+1].shape[0])))
        st_geoM1y[num2] = np.abs(geoM1y[num2]-np.exp(np.mean(np.log(gM1y[num2+1]))-np.std(np.log(gM1y[num2+1]))/np.sqrt(gM1y[num2+1].shape[0])))
        st_geoM2xy[num2] = np.abs(geoM2xy[num2]-np.exp(np.mean(np.log(gM2xy[num2+1]))-np.std(np.log(gM2xy[num2+1]))/np.sqrt(gM2xy[num2+1].shape[0])))

        geoD1x[num2] = geoM1x[num2]/time2[num2]
        geoD1y[num2] = geoM1y[num2]/time2[num2]
        geoD2xy[num2] = geoM2xy[num2]/time2[num2]

        arM1x[num2] = np.mean(gM1x[num2+1])
        arM1y[num2] = np.mean(gM1y[num2+1])
        arM2xy[num2] = np.mean(gM2xy[num2+1])

        st_arM1x[num2] = np.std(gM1x[num2+1])
        st_arM1y[num2] = np.std(gM1y[num2+1])
        st_arM2xy[num2] = np.std(gM2xy[num2+1])

        arD1x[num2] = arM1x[num2]/time2[num2]
        arD1y[num2] = arM1y[num2]/time2[num2]
        arD2xy[num2] = arM2xy[num2]/time2[num2]

        st_arD1x[num2] = st_arM1x[num2]/time2[num2]
        st_arD1y[num2] = st_arM1y[num2]/time2[num2]
        st_arD2xy[num2] = st_arM2xy[num2]/time2[num2]

    np.savetxt('geoD2xy_{}.csv'.format(name), geoD2xy, delimiter=',')
    np.savetxt('arD2xy_{}.csv'.format(name), arD2xy, delimiter=',')
    #np.savetxt('FD2xy_{}.csv'.format(name), FD2xy, delimiter=',')

    np.savetxt('geoM2xy_{}.csv'.format(name), geoM2xy, delimiter=',')
    np.savetxt('arM2xy_{}.csv'.format(name), arM2xy, delimiter=',')
    #np.savetxt('FM2xy_{}.csv'.format(name), FM2xy, delimiter=',')

    np.savetxt('geoM1x_{}.csv'.format(name), geoM1x, delimiter=',')
    np.savetxt('arM1x_{}.csv'.format(name), arM1x, delimiter=',')
    np.savetxt('geoM1y_{}.csv'.format(name), geoM1y, delimiter=',')
    np.savetxt('arM1y_{}.csv'.format(name), arM1y, delimiter=',')
