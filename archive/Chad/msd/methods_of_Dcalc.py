from bokeh.io import output_notebook
from bokeh.plotting import figure, show, gridplot, hplot, vplot, curdoc
import numpy as np
import os
import csv
from bokeh.client import push_session
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt


def Dpointder(traj, n1, n2, n3, tn, nD):
    """
    Outputs the average diffusion coefficient at a desired timepoint.  User puts in a time.  The code finds the point
    closest to that time and gives the diffusion coefficient at that time.

    Note that this code is only for 2D datasets.  I need to modify it slightly for 3D datasets.

    n1: particle numbers (typically 0)
    n2: time (15 when using prettify)
    n3: MSDs or Deffs (9 for MSDs 17 for Deffs)
    tn: desired time
    nD: Either '2D', '1Dx', or '1Dy'

    Returns the values:

    MMSD: The diffusion coefficient at the desired time
    SD: The standard deviation
    total1: The number of particles taken into account in the calculation
    t: The timepoint at which the diffusion coefficient was calculated
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    total = int(max(particles))
    total1 = total + 1
    rawtime = traj[:, n2]
    bow = traj.shape[0]
    raw2DMSDs = np.zeros((bow, 4))
    raw2DMSDs[:, 0] = traj[:, n3]
    raw2DMSDs[:, 1:4] = traj[:, n3 + 3:n3 + 6]
    MSD = dict()
    time = dict()

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        MSD[num] = (raw2DMSDs[min1+2:max1, :])
        time[num] = (rawtime[min1+2:max1])

    MMSD = MSD[1]
    for num in range(2, total1):
        MMSD = MMSD + MSD[num]
    MMSD = MMSD/total1
    SD = np.zeros(np.shape(MMSD))
    t = time[1][:]

    # Now to calculate the standard dev at each point:
    for num in range(1, total1):
        SDunit = (MSD[num] - MMSD)**2
        SD = SD + SDunit
    SD = np.sqrt(SD/total1)
    SE = SD/np.sqrt(total1)

    def find_nearest(array, value):
        idx = (np.abs(array-value)).argmin()
        return array[idx], idx

    td, idx = find_nearest(t, tn)

    if nD == '2D':
        D = MMSD[idx, 0]
        SDd = SD[idx, 0]
    elif nD == '1Dx':
        D = MMSD[idx, 1]
        SDd = SD[idx, 1]
    else:
        D = MMSD[idx, 2]
        SDd = SD[idx, 2]

    return D, SDd, total1, td


def Dlinalg(traj, n1, n2, n3, nD):
    """
    Outputs the average diffusion coefficient at a desired timepoint.  User puts in a time.  The code finds the point
    closest to that time and gives the diffusion coefficient at that time.

    Note that this code is only for 2D datasets.  I need to modify it slightly for 3D datasets.

    n1: particle numbers (typically 0)
    n2: time (15 when using prettify)
    n3: MSDs or Deffs (9 for MSDs 17 for Deffs) **MUST be in MSDs in this case, not Deffs**
    nD: Either '2D', '1Dx', or '1Dy'

    Returns the values:

    MMSD: The diffusion coefficient at the desired time
    SD: The standard deviation
    total1: The number of particles taken into account in the calculation
    t: The timepoint at which the diffusion coefficient was calculated
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    total = int(max(particles))
    total1 = total + 1
    rawtime = traj[:, n2]
    bow = traj.shape[0]
    raw2DMSDs = np.zeros((bow, 4))
    raw2DMSDs[:, 0] = traj[:, n3]
    raw2DMSDs[:, 1:4] = traj[:, n3 + 3:n3 + 6]
    MSD = dict()
    time = dict()

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        MSD[num] = (raw2DMSDs[min1+2:max1, :])
        time[num] = (rawtime[min1+2:max1])

    MMSD = MSD[1]
    for num in range(2, total1):
        MMSD = MMSD + MSD[num]
    MMSD = MMSD/total1
    SD = np.zeros(np.shape(MMSD))
    t = time[1][:]

    # Now to calculate the standard dev at each point:
    for num in range(1, total1):
        SDunit = (MSD[num] - MMSD)**2
        SD = SD + SDunit
    SD = np.sqrt(SD/total1)
    SE = SD/np.sqrt(total1)

    w = dict()
    cov = dict()
    sd = dict()
    line = dict()
    # A = np.ones((np.shape(t)[0], 2))
    # A[:, 0] = t
    # w[0] = np.linalg.lstsq(A, MMSD[:, 0])[0]
    # w[1] = np.linalg.lstsq(A, MMSD[:, 1])[0]
    # w[2] = np.linalg.lstsq(A, MMSD[:, 2])[0]

    w[0], cov[0] = np.polyfit(t, MMSD[:, 0], 1, cov=True)
    w[1], cov[1] = np.polyfit(t, MMSD[:, 1], 1, cov=True)
    w[2], cov[2] = np.polyfit(t, MMSD[:, 2], 1, cov=True)

    sd[0] = np.sqrt(np.diag(cov[0]))
    sd[1] = np.sqrt(np.diag(cov[1]))
    sd[2] = np.sqrt(np.diag(cov[2]))

    if nD == '2D':
        D = w[0][0]/4
        SDd = sd[0][0]/4
    elif nD == '1Dx':
        D = w[1][0]/2
        SDd = sd[1][0]/2
    else:
        D = w[2][0]/2
        SDd = sd[2][0]/2

    return D, SDd, total1


def Dlinalgw(traj, n1, n2, n3, nD):
    """
    Outputs the average diffusion coefficient at a desired timepoint.  User puts in a time.  The code finds the point
    closest to that time and gives the diffusion coefficient at that time.

    Note that this code is only for 2D datasets.  I need to modify it slightly for 3D datasets.

    n1: particle numbers (typically 0)
    n2: time (15 when using prettify)
    n3: MSDs or Deffs (9 for MSDs 17 for Deffs) **MUST be in MSDs in this case, not Deffs**
    nD: Either '2D', '1Dx', or '1Dy'

    Returns the values:

    MMSD: The diffusion coefficient at the desired time
    SD: The standard deviation
    total1: The number of particles taken into account in the calculation
    t: The timepoint at which the diffusion coefficient was calculated
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    total = int(max(particles))
    total1 = total + 1
    rawtime = traj[:, n2]
    bow = traj.shape[0]
    raw2DMSDs = np.zeros((bow, 4))
    raw2DMSDs[:, 0] = traj[:, n3]
    raw2DMSDs[:, 1:4] = traj[:, n3 + 3:n3 + 6]
    MSD = dict()
    time = dict()

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        MSD[num] = (raw2DMSDs[min1+2:max1, :])
        time[num] = (rawtime[min1+2:max1])

    MMSD = MSD[1]
    for num in range(2, total1):
        MMSD = MMSD + MSD[num]
    MMSD = MMSD/total1
    SD = np.zeros(np.shape(MMSD))
    t = time[1][:]

    # Now to calculate the standard dev at each point:
    for num in range(1, total1):
        SDunit = (MSD[num] - MMSD)**2
        SD = SD + SDunit
    SD = np.sqrt(SD/total1)
    SE = SD/np.sqrt(total1)

    w = dict()
    cov = dict()
    sd = dict()
    line = dict()
    # A = np.ones((np.shape(t)[0], 2))
    # A[:, 0] = t
    # w[0] = np.linalg.lstsq(A, MMSD[:, 0])[0]
    # w[1] = np.linalg.lstsq(A, MMSD[:, 1])[0]
    # w[2] = np.linalg.lstsq(A, MMSD[:, 2])[0]

    weight = dict()
    weight[0] = [1.0 / ty for ty in SD[:, 0]]
    weight[1] = [1.0 / ty for ty in SD[:, 1]]
    weight[2] = [1.0 / ty for ty in SD[:, 2]]

    w[0], cov[0] = np.polyfit(t, MMSD[:, 0], 1, w=weight[0], full=False, cov=True)
    w[1], cov[1] = np.polyfit(t, MMSD[:, 1], 1, w=weight[1], full=False, cov=True)
    w[2], cov[2] = np.polyfit(t, MMSD[:, 2], 1, w=weight[2], full=False, cov=True)

    sd[0] = np.sqrt(np.diag(cov[0]))
    sd[1] = np.sqrt(np.diag(cov[1]))
    sd[2] = np.sqrt(np.diag(cov[2]))

    if nD == '2D':
        D = w[0][0]/4
        SDd = sd[0][0]/4
    elif nD == '1Dx':
        D = w[1][0]/2
        SDd = sd[1][0]/2
    else:
        D = w[2][0]/2
        SDd = sd[2][0]/2

    return D, SDd, total1


def Dpointder3D(traj, n1, n2, n3, tn, nD):
    """
    Outputs the average diffusion coefficient at a desired timepoint.  User puts in a time.  The code finds the point
    closest to that time and gives the diffusion coefficient at that time.

    Note that this code is only for 2D datasets.  I need to modify it slightly for 3D datasets.

    n1: particle numbers (typically 0)
    n2: time (15 when using prettify)
    n3: MSDs or Deffs (8 for 3D MSDs 37 for 3D Deffs)
    tn: desired time
    nD: Either '2D', '1Dx', or '1Dy'

    Returns the values:

    MMSD: The diffusion coefficient at the desired time
    SD: The standard deviation
    total1: The number of particles taken into account in the calculation
    t: The timepoint at which the diffusion coefficient was calculated
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    total = int(max(particles))
    total1 = total + 1
    rawtime = traj[:, n2]
    bow = traj.shape[0]
    raw3DMSDs = np.zeros((bow, 5))
    raw3DMSDs[:, 0:4] = traj[:, n3: n3 + 4]
    MSD = dict()
    time = dict()

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        MSD[num] = (raw3DMSDs[min1+1:max1, :])
        time[num] = (rawtime[min1+1:max1])

    MMSD = MSD[1]
    for num in range(2, total1):
        MMSD = MMSD + MSD[num]
    MMSD = MMSD/total1
    SD = np.zeros(np.shape(MMSD))
    t = time[1][:]

    # Now to calculate the standard dev at each point:
    for num in range(1, total1):
        SDunit = (MSD[num] - MMSD)**2
        SD = SD + SDunit
    SD = np.sqrt(SD/total1)
    SE = SD/np.sqrt(total1)

    def find_nearest(array, value):
        idx = (np.abs(array-value)).argmin()
        return array[idx], idx

    td, idx = find_nearest(t, tn)

    if nD == '3D':
        D = MMSD[idx, 0]
        SDd = SD[idx, 0]
    elif nD == '2Dxy':
        D = MMSD[idx, 1]
        SDd = SD[idx, 1]
    elif nD == '2Dxz':
        D = MMSD[idx, 2]
        SDd = SD[idx, 2]
    else:
        D = MMSD[idx, 3]
        SDd = SD[idx, 3]

    return D, SDd, total1, td


def Dlinalg3D(traj, n1, n2, n3, nD):
    """
    Outputs the average diffusion coefficient at a desired timepoint.  User puts in a time.  The code finds the point
    closest to that time and gives the diffusion coefficient at that time.

    Note that this code is only for 2D datasets.  I need to modify it slightly for 3D datasets.

    n1: particle numbers (typically 0)
    n2: time (15 when using prettify)
    n3: MSDs or Deffs (9 for MSDs 17 for Deffs) **MUST be in MSDs in this case, not Deffs**
    nD: Either '2D', '1Dx', or '1Dy'

    Returns the values:

    MMSD: The diffusion coefficient at the desired time
    SD: The standard deviation
    total1: The number of particles taken into account in the calculation
    t: The timepoint at which the diffusion coefficient was calculated
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    total = int(max(particles))
    total1 = total + 1
    rawtime = traj[:, n2]
    bow = traj.shape[0]
    raw3DMSDs = np.zeros((bow, 5))
    raw3DMSDs[:, 0:4] = traj[:, n3: n3 + 4]
    MSD = dict()
    time = dict()

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        MSD[num] = (raw3DMSDs[min1+2:max1, :])
        time[num] = (rawtime[min1+2:max1])

    MMSD = MSD[1]
    for num in range(2, total1):
        MMSD = MMSD + MSD[num]
    MMSD = MMSD/total1
    SD = np.zeros(np.shape(MMSD))
    t = time[1][:]

    # Now to calculate the standard dev at each point:
    for num in range(1, total1):
        SDunit = (MSD[num] - MMSD)**2
        SD = SD + SDunit
    SD = np.sqrt(SD/total1)
    SE = SD/np.sqrt(total1)

    w = dict()
    cov = dict()
    sd = dict()
    line = dict()
    # A = np.ones((np.shape(t)[0], 2))
    # A[:, 0] = t
    # w[0] = np.linalg.lstsq(A, MMSD[:, 0])[0]
    # w[1] = np.linalg.lstsq(A, MMSD[:, 1])[0]
    # w[2] = np.linalg.lstsq(A, MMSD[:, 2])[0]

    w[0], cov[0] = np.polyfit(t, MMSD[:, 0], 1, cov=True)
    w[1], cov[1] = np.polyfit(t, MMSD[:, 1], 1, cov=True)
    w[2], cov[2] = np.polyfit(t, MMSD[:, 2], 1, cov=True)
    w[3], cov[3] = np.polyfit(t, MMSD[:, 3], 1, cov=True)

    sd[0] = np.sqrt(np.diag(cov[0]))
    sd[1] = np.sqrt(np.diag(cov[1]))
    sd[2] = np.sqrt(np.diag(cov[2]))
    sd[3] = np.sqrt(np.diag(cov[3]))

    if nD == '3D':
        D = w[0][0]/6
        SDd = sd[0][0]/6
    elif nD == '2Dxy':
        D = w[1][0]/4
        SDd = sd[1][0]/4
    elif nD == '2Dxz':
        D = w[2][0]/4
        SDd = sd[2][0]/4
    else:
        D = w[3][0]/4
        SDd = sd[3][0]/4

    return D, SDd, total1


def Dlinalgw3D(traj, n1, n2, n3, nD):
    """
    Outputs the average diffusion coefficient at a desired timepoint.  User puts in a time.  The code finds the point
    closest to that time and gives the diffusion coefficient at that time.

    Note that this code is only for 2D datasets.  I need to modify it slightly for 3D datasets.

    n1: particle numbers (typically 0)
    n2: time (15 when using prettify)
    n3: MSDs or Deffs (9 for MSDs 17 for Deffs) **MUST be in MSDs in this case, not Deffs**
    nD: Either '2D', '1Dx', or '1Dy'

    Returns the values:

    MMSD: The diffusion coefficient at the desired time
    SD: The standard deviation
    total1: The number of particles taken into account in the calculation
    t: The timepoint at which the diffusion coefficient was calculated
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    total = int(max(particles))
    total1 = total + 1
    rawtime = traj[:, n2]
    bow = traj.shape[0]
    raw3DMSDs = np.zeros((bow, 5))
    raw3DMSDs[:, 0:4] = traj[:, n3: n3 + 4]
    MSD = dict()
    time = dict()

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        MSD[num] = (raw3DMSDs[min1+2:max1, :])
        time[num] = (rawtime[min1+2:max1])

    MMSD = MSD[1]
    for num in range(2, total1):
        MMSD = MMSD + MSD[num]
    MMSD = MMSD/total1
    SD = np.zeros(np.shape(MMSD))
    t = time[1][:]

    # Now to calculate the standard dev at each point:
    for num in range(1, total1):
        SDunit = (MSD[num] - MMSD)**2
        SD = SD + SDunit
    SD = np.sqrt(SD/total1)
    SE = SD/np.sqrt(total1)

    w = dict()
    cov = dict()
    sd = dict()
    line = dict()
    # A = np.ones((np.shape(t)[0], 2))
    # A[:, 0] = t
    # w[0] = np.linalg.lstsq(A, MMSD[:, 0])[0]
    # w[1] = np.linalg.lstsq(A, MMSD[:, 1])[0]
    # w[2] = np.linalg.lstsq(A, MMSD[:, 2])[0]

    weight = dict()
    weight[0] = [1.0 / ty for ty in SD[:, 0]]
    weight[1] = [1.0 / ty for ty in SD[:, 1]]
    weight[2] = [1.0 / ty for ty in SD[:, 2]]
    weight[3] = [1.0 / ty for ty in SD[:, 3]]

    w[0], cov[0] = np.polyfit(t, MMSD[:, 0], 1, w=weight[0], full=False, cov=True)
    w[1], cov[1] = np.polyfit(t, MMSD[:, 1], 1, w=weight[1], full=False, cov=True)
    w[2], cov[2] = np.polyfit(t, MMSD[:, 2], 1, w=weight[2], full=False, cov=True)
    w[3], cov[3] = np.polyfit(t, MMSD[:, 3], 1, w=weight[3], full=False, cov=True)

    sd[0] = np.sqrt(np.diag(cov[0]))
    sd[1] = np.sqrt(np.diag(cov[1]))
    sd[2] = np.sqrt(np.diag(cov[2]))
    sd[3] = np.sqrt(np.diag(cov[3]))

    if nD == '3D':
        D = w[0][0]/6
        SDd = sd[0][0]/6
    elif nD == '2Dxy':
        D = w[1][0]/4
        SDd = sd[1][0]/4
    elif nD == '2Dxz':
        D = w[2][0]/4
        SDd = sd[2][0]/4
    else:
        D = w[3][0]/4
        SDd = sd[3][0]/4

    return D, SDd, total1
