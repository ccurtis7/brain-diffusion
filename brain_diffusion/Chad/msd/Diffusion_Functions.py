import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def frame_adjust(data):

    # This function (start_shift) will shift all frames down to a point where the first frame of
    # every particle is zero.

    def start_shift(x):
        startframe = x[0, 1]
        x[:,1] = x[:,1] - startframe
        return x

    data = start_shift(data)

    # Determine the largest frame in the data set (data), which will be used to size our output dataset:
    max_frame = int(max(data[:,1]))

    # Create an empty dataset that can fit the required amount of frames:
    final = np.zeros((max_frame + 1, 6))

    # Let the first row in the output data set equal the first row of the input data set:
    final[0, :] = data[0, :]

    # set up a counter to keep track of how many new rows have been created:
    new = 0

    p = 1
    while p <= max_frame:
        # If the difference between the current frame # and the previous tracked frame # = 1,
        # then the row will be filled in with the corresponding row of the old data set frame
        if data[p - new, 1] - data[p - new - 1, 1] == 1:
            final[p, :] = data[p - new, :]
        #if data[p, 1] - data[p-1, 1] == 1:
        #    final[p + new, :] = data[p, :]

        #elif data[p, 1] - data[p-1, 1] > 1:
        #    diff = data[p, 1] - data[p-1, 1]
        #    new = new + (diff - 1)
        #    b = 1
        #    while b < diff:
        #        final[(data[p,1]-)]
        #        b += 1
        # Else If the difference between the current frame # and the previous tracked frame # > 1,
        # then the data set corresponding to the previously tracked frame will be duplicated, and
        # inserted just after the previous row, the only difference being it's frame now increased by 1
        elif data[p - new, 1] - data[p - new - 1, 1] > 1:
            new = new + 1
            duplicate = data[p - new]
            duplicate[1] = duplicate[1] + 1
            final[p, :] = duplicate
        p += 1
    return final


def MSD_Deff_Calcs(data, cutoff, limit, umppx, fps, umps):
    """
    # data: the data imported from the csv file output by Image J (MOSAIC Particle Tracking plug-in)
    # cutoff: if a given trajectory does not include at least this # of frames, it will be discarded
    # limit: this is the maximum amount of frames that will be kept from each trajectory (any frames after 40 will be discarded)
    # umppx: conversion factor for x,y coordinate data (used to convert x,y location from pixels to microns, units are microns       # per pixel)
    # fps: conversion factor for frame # (used to convert frame # to time, units are frames per sec)
    # umps: conversion factor for z coordinate data (units are microns per z stack)
    """
    dataset = dict()
    # Since we're now doing frame by frame, it's going to be important to retain the original
    # frame for each data point.

    # This extracts the whole column corresponding to the particle #
    particles = data[:,0]
    # This determines the total # of particles tracked in the given video
    total_part = int(max(particles))
    # Convert x,y,z coordinates from pixels to microns ****** SHOULD Z be INCLUDED HERE?
    data[:, 2:5] = umppx * data[:, 2:5]
    # Does something for the 3D data.. not really sure, ask Chad
    data[:, 4] = umps * data[:, 4]

    # Since we're now doing frame by frame, it's going to be important to retain the original
    # frame for each data point.
    # This is going to make each particle its own key in a library
    i = 1
    while i <= total_part:
        current_particle_data = np.where(particles == i)[0]
        current_min = min(current_particle_data)
        current_max = max(current_particle_data)
        dataset[i] = data[current_min:(current_max + 1), 0:5]
        frame_data = np.zeros((len(dataset[i]), 1))
        n = 0
        while n < len(dataset[i]):
            frame_data[n, 0] = dataset[i][n, 1]
            n += 1
        dataset[i] = np.append(dataset[i], frame_data, axis = 1)
        i += 1

    calcs_included = dict()
    q = 1
    while q <= total_part:
        calcs_included[q] = frame_adjust(dataset[q])

        # Calculations of the max and min values of each coordinate
        xmax = max(calcs_included[q][:, 2])
        xmin = min(calcs_included[q][:, 2])
        ymax = max(calcs_included[q][:, 3])
        ymin = min(calcs_included[q][:, 3])
        zmax = max(calcs_included[q][:, 4])
        zmin = min(calcs_included[q][:, 4])

        # Use min and max values to calculate the shift required to center each x, y, and z data point (uses mid-range):
        xc = np.array([calcs_included[q][:, 2] - ((xmax+xmin)/2)])
        yc = np.array([calcs_included[q][:, 3] - ((ymax+ymin)/2)])
        zc = np.array([calcs_included[q][:, 4] - ((zmax+zmin)/2)])

        # define the starting point of each particle (will be used to calc MSD):
        xstart = xc[0, 0]
        ystart = yc[0, 0]
        zstart = zc[0, 0]

        # append the shifted x,y,z data sets to the final data set after they have been shifted
        # in a way that they will encompass the midrange of the data set
        calcs_included[q] = np.append(calcs_included[q], xc.T, axis=1)
        calcs_included[q] = np.append(calcs_included[q], yc.T, axis=1)
        calcs_included[q] = np.append(calcs_included[q], zc.T, axis=1)

        # Create set of blank arrays that will be filled with various calculated values:
        N_rows = len(calcs_included[q])
        MSD3 = np.zeros((N_rows, 1))
        MSD2xy = np.zeros((N_rows, 1))
        MSD2xz = np.zeros((N_rows, 1))
        MSD2yz = np.zeros((N_rows, 1))
        MSD1x = np.zeros((N_rows, 1))
        MSD1y = np.zeros((N_rows, 1))
        MSD1z = np.zeros((N_rows, 1))
        D3 = np.zeros((N_rows, 1))
        D2xy = np.zeros((N_rows, 1))
        D2xz = np.zeros((N_rows, 1))
        D2yz = np.zeros((N_rows, 1))
        D1x = np.zeros((N_rows, 1))
        D1y = np.zeros((N_rows, 1))
        D1z = np.zeros((N_rows, 1))

        # Calculate the time at which each xyz data set was obtained, using fps conversion.. then append
        time = calcs_included[q][:,1] * (1/fps)
        time[0] = 0.0000000001

        # Calculations of MSD's and diffusion coefficients in 3D and all possible 2D and 1D directions:
        n = 0
        while n < N_rows:
            MSD3[n, 0] = (xc[0, n] - xstart)**2 + (yc[0, n] - ystart)**2 + (zc[0, n] - zstart)**2
            MSD2xy[n, 0] = (xc[0,n]-xstart)**2 + (yc[0, n] - ystart)**2
            MSD2xz[n, 0] = (xc[0,n]-xstart)**2 + (zc[0, n] - zstart)**2
            MSD2yz[n, 0] = (zc[0,n]-zstart)**2 + (yc[0, n] - ystart)**2
            MSD1x[n, 0] = (xc[0,n]-xstart)**2
            MSD1y[n, 0] = (yc[0,n]-ystart)**2
            MSD1z[n, 0] = (zc[0,n]-zstart)**2

            D3[n, 0] = MSD3[n, 0]/6/time[n]
            D2xy[n, 0] = MSD2xy[n, 0]/4/time[n]
            D2xz[n, 0] = MSD2xz[n, 0]/4/time[n]
            D2yz[n, 0] = MSD2yz[n, 0]/4/time[n]
            D1x[n, 0] = MSD1x[n, 0]/2/time[n]
            D1y[n, 0] = MSD1y[n, 0]/2/time[n]
            D1z[n, 0] = MSD1z[n, 0]/2/time[n]
            n += 1

        # Now append these calculated values to dictionary "calcs_included":

        calcs_included[q] = np.append(calcs_included[q], MSD3, axis=1)
        calcs_included[q] = np.append(calcs_included[q], MSD2xy, axis=1)
        calcs_included[q] = np.append(calcs_included[q], MSD2xz, axis=1)
        calcs_included[q] = np.append(calcs_included[q], MSD2yz, axis=1)
        calcs_included[q] = np.append(calcs_included[q], MSD1x, axis=1)
        calcs_included[q] = np.append(calcs_included[q], MSD1y, axis=1)
        calcs_included[q] = np.append(calcs_included[q], MSD1z, axis=1)
        calcs_included[q] = np.append(calcs_included[q], np.array([time]).T, axis=1)
        calcs_included[q] = np.append(calcs_included[q], D3, axis=1)
        calcs_included[q] = np.append(calcs_included[q], D2xy, axis=1)
        calcs_included[q] = np.append(calcs_included[q], D2xz, axis=1)
        calcs_included[q] = np.append(calcs_included[q], D2yz, axis=1)
        calcs_included[q] = np.append(calcs_included[q], D1x, axis=1)
        calcs_included[q] = np.append(calcs_included[q], D1y, axis=1)
        calcs_included[q] = np.append(calcs_included[q], D1z, axis=1)

        q += 1

    # The following code will take into account the cutoff value provided (min # of frames required for data to be kept)
    # Any trajectory that doesn't meet the cutoff will be dropped.
    # In addition, this will set the limit on the total # of frames a given trajectory can have,
    # set by user as imput "limit". Any trajectory data past that frame limit is dropped.

    exclude_short_traj = dict()
    exclude_count = 0
    keep_count = 0

    w = 1
    while w <= total_part:
        if len(calcs_included[w]) < cutoff:
            exclude_count = exclude_count + 1
        else:
            keep_count = keep_count + 1
            exclude_short_traj[w - exclude_count] = calcs_included[w]
            exclude_short_traj[w - exclude_count][:,0] = keep_count
        w += 1

    final = dict()

    w = 1
    # Feel like this should maybe be less than or equal to?
    while w < keep_count:
        final[w] = exclude_short_traj[w][0:limit,:]
        w += 1

    return(final, (keep_count - 1))


def prettify2(traj, cut, lim, umppx, fps, umps):
    """
    This function takes a trajectory dataset that has been extracted from a csv file from the MOSAIC code and augments
    it by calculating MSDs and Deffs and putting those in new columns.  The final output looks like this:

    Output:
    0 particle number
    1 frames
    2 x coordinate
    3 y coordinate
    4 z coordinate
    5 centered x coordinate
    6 centered y coordinate
    7 centered z coordinate
    8 3D MSD
    9 2D xy MSD
    10 2D xz MSD
    11 2D yz MSD
    12 1D x MSD
    13 1D y MSD
    14 1D z MSD
    15 time
    16 3D Deff
    17 2D xy Deff
    18 2D xz Deff
    19 2D yz Deff
    20 1D x Deff
    21 1D y Deff
    22 1D z Deff

    New functionality to this code includes user inputs to define um/px defined by the microscope settings to
    convert from pixels to ums.

    traj: a dataset from the MOSAIC code with the top row and first column removed.
    cut: the minimum number of frames required to be included in final dataset
    lim: the specified number of frames to be included in final dataset (often the same as cut)
    fps: frames per second
    umppx: microns per pixel
    umps: microns per slice (for 3D datasets, set to 1 otherwise)
    """

    dataset = dict()

    particles = traj[:, 0]
    total = int(max(particles))
    total1 = total + 1
    rawdataset = traj[:, :]

    # I defined the new variable fixed to fix the self-updating prettify
    # function.
    fixed = np.zeros(traj.shape)
    fixed[:, 0:2] = rawdataset[:, 0:2]
    fixed[:, 2:4] = umppx * rawdataset[:, 2:4]
    fixed[:, 4] = umps * rawdataset[:, 4]

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        dataset[num] = (fixed[min1:max1, 0:5])

    flee = dict()
    for num in range(1, total1):
        flee[num] = fillin2(dataset[num])

        xmax = max(flee[num][:, 2])
        xmin = min(flee[num][:, 2])
        ymax = max(flee[num][:, 3])
        ymin = min(flee[num][:, 3])
        zmax = max(flee[num][:, 4])
        zmin = min(flee[num][:, 4])

        xc = np.array([flee[num][:, 2] - ((xmax+xmin)/2)])
        yc = np.array([flee[num][:, 3] - ((ymax+ymin)/2)])
        zc = np.array([flee[num][:, 4] - ((zmax+zmin)/2)])

        xstart = xc[0, 0]
        ystart = yc[0, 0]
        zstart = zc[0, 0]

        flee[num] = np.append(flee[num], xc.T, axis=1)
        flee[num] = np.append(flee[num], yc.T, axis=1)
        flee[num] = np.append(flee[num], zc.T, axis=1)

        the = flee[num].shape[0]
        MSD3 = np.zeros((the, 1))
        M2xy = np.zeros((the, 1))
        M2xz = np.zeros((the, 1))
        M2yz = np.zeros((the, 1))
        M1x = np.zeros((the, 1))
        M1y = np.zeros((the, 1))
        M1z = np.zeros((the, 1))

        # This defines the units of time.  This is more approrpriately an input to the function.  Will fix.
        time = (1/fps) * flee[num][:, 1]
        time[0] = 0.0000000001
        time1 = np.array([time]).T

        D3 = np.zeros((the, 1))
        D2xy = np.zeros((the, 1))
        D2xz = np.zeros((the, 1))
        D2yz = np.zeros((the, 1))
        D1x = np.zeros((the, 1))
        D1y = np.zeros((the, 1))
        D1z = np.zeros((the, 1))

        for bum in range(0, the):
            MSD3[bum, 0] = (xc[0, bum] - xstart)**2 + (yc[0, bum] - ystart)**2 + (zc[0, bum] - zstart)**2
            M2xy[bum, 0] = (xc[0, bum] - xstart)**2 + (yc[0, bum] - ystart)**2
            M2xz[bum, 0] = (xc[0, bum] - xstart)**2 + (zc[0, bum] - zstart)**2
            M2yz[bum, 0] = (yc[0, bum] - ystart)**2 + (zc[0, bum] - zstart)**2
            M1x[bum, 0] = (xc[0, bum] - xstart)**2
            M1y[bum, 0] = (yc[0, bum] - ystart)**2
            M1z[bum, 0] = (zc[0, bum] - zstart)**2

            D3[bum, 0] = MSD3[bum, 0]/(6*time[bum])
            D2xy[bum, 0] = M2xy[bum, 0]/(4*time[bum])
            D2xz[bum, 0] = M2xz[bum, 0]/(4*time[bum])
            D2yz[bum, 0] = M2yz[bum, 0]/(4*time[bum])
            D1x[bum, 0] = M1x[bum, 0]/(2*time[bum])
            D1y[bum, 0] = M1y[bum, 0]/(2*time[bum])
            D1z[bum, 0] = M1z[bum, 0]/(2*time[bum])

        flee[num] = np.append(flee[num], MSD3, axis=1)
        flee[num] = np.append(flee[num], M2xy, axis=1)
        flee[num] = np.append(flee[num], M2xz, axis=1)
        flee[num] = np.append(flee[num], M2yz, axis=1)
        flee[num] = np.append(flee[num], M1x, axis=1)
        flee[num] = np.append(flee[num], M1y, axis=1)
        flee[num] = np.append(flee[num], M1z, axis=1)
        flee[num] = np.append(flee[num], time1, axis=1)
        flee[num] = np.append(flee[num], D3, axis=1)
        flee[num] = np.append(flee[num], D2xy, axis=1)
        flee[num] = np.append(flee[num], D2xz, axis=1)
        flee[num] = np.append(flee[num], D2yz, axis=1)
        flee[num] = np.append(flee[num], D1x, axis=1)
        flee[num] = np.append(flee[num], D1y, axis=1)
        flee[num] = np.append(flee[num], D1z, axis=1)

        teancum = dict()
        fifties = 0
        nones = 0
    cutoff = cut

    for num in range(1, total1):
        if flee[num].shape[0] < cutoff:
            nones = nones + 1
        else:
            fifties = fifties + 1
            teancum[num - nones] = flee[num]
            # I must also redefine the particle numbers to reflect the new set.
            teancum[num - nones][:, 0] = fifties

    moroni = dict()
    limit = lim

    for num in range(1, fifties):
        moroni[num] = teancum[num][0:limit, :]

    fifties = fifties - 1

    return (moroni, fifties)


def fillin2(data):
    """
    This function is perfect.  It shifts the frames by the startframe and fills in any blank frames.
    """
    def startshift(data1):
        startframe = data1[0, 1]
        data1[:, 1] = data1[:, 1] - startframe
        return data1

    data1 = startshift(data)

    shap = int(max(data1[:, 1])) + 1
    filledin = np.zeros((shap, 5))
    filledin[0, :] = data1[0, :]

    count = 0
    new = 0
    other = 0
    tot = 0

    for num in range(1, shap):
        if data1[num-new, 1]-data1[num-new-1, 1] == 1:
            count = count + 1
            filledin[num, :] = data1[num-new, :]
        elif data1[num - new, 1]-data1[num - new - 1, 1] > 1:
            new = new + 1
            iba = int(data1[num - new+1, 1]-data1[num - new, 1])
            togoin = data1[num - new]
            togoin[1] = togoin[1] + 1
            filledin[num, :] = togoin
            # dataset[2] = np.insert(dataset[2], [num + new - 2], togoin, axis=0)

        else:
            other = other + 1
        tot = count + new + other

    return filledin
