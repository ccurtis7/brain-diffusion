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
    fillin2(data)

    Fills in blanks in an input trajectory dataset.

    Parameters
    ----------
    data : numpy array
        Must have 5 columns containing in order Track ID, frames, x coordinates,
        y coordinates, and z coordinates.  Must contain a single Track ID, no
        more.  Frames must be in ascending order.

    Returns
    ----------
    filledin : numpy array
        Numpy array of size frames x 5 containing Track ID, frames, x
        coordinates, y coordinates, and z coordinates.  Frames are filled in
        using a carryover method (no regression performed).

    Examples
    ----------
    >>> n = 6
    >>> df = np.zeros((6, 5))
    >>> df[:, 0] = np.ones(6)
    >>> df[:, 1] = np.linspace(0, 100, 6)
    >>> df[:, 2] = np.linspace(0, 100, 6)
    >>> df[:, 3] = np.linspace(0, 100, 6)
    >>> df[:, 4] = np.zeros(6)
    >>> fillin2(df)
    array([[  1.,   0.,   0.,   0.,   0.],
           [  1.,   1.,   0.,   0.,   0.],
           [  1.,   2.,   2.,   2.,   0.],
           [  1.,   3.,   2.,   2.,   0.],
           [  1.,   4.,   4.,   4.,   0.],
           [  1.,   5.,   4.,   4.,   0.],
           [  1.,   6.,   6.,   6.,   0.],
           [  1.,   7.,   6.,   6.,   0.],
           [  1.,   8.,   8.,   8.,   0.],
           [  1.,   9.,   8.,   8.,   0.],
           [  1.,  10.,  10.,  10.,   0.]])

    """

    assert data.shape[1] == 5, "Input array must have five columns."
    assert type(data) == np.ndarray, "Input must be a numpy array"
    assert np.all(np.diff(data[:, 1]) == abs(np.diff(data[:, 1]))), "Frames must all increase."
    assert data.shape[0] > 0, "Array must not be empty."

    shap = int(max(data[:, 1])) + 1
    shape1 = int(min(data[:, 1]))
    newshap = shap - shape1
    filledin = np.zeros((newshap, 5))
    filledin[0, :] = data[0, :]
    frames = data[:, 1]
    filledin[:, 1] = np.linspace(shape1, shap-1, newshap)

    count = 0
    new = 0
    other = 0
    tot = 0

    for num in range(1, newshap):
        # if a frame isn't skipped
        if filledin[num, 1]-frames[num-new-1] == 0:
            count = count + 1
        elif filledin[num, 1]-frames[num-new-1] < 0:
            new = new + 1
        else:
            other = other + 1

        filledin[num, 0] = data[num-new-1, 0]
        filledin[num, 2:5] = data[num-new-1, 2:5]
        tot = count + new + other

    return filledin


def MSD_iteration(folder, name, cut=1, totvids=1, conversion=(1, 1, 1)):
    """
    Arranges trajectory xy data into numpy arrays of dimensions frames x particles

    Parameters
    ----------
    folder : string
        Directory containing datasets to be analyzed.
    name : string
        Base name of files to be analzed.  The code has a very specific naming
        convenction, and requires files to be of the structure:

        Traj_{}_{}.csv.format(name, video_number)

        where name is the base name of the files and video number is the number
        associated with the video.  Numbers must begin at 1 and increase in
        units of 1.
    cut : integer
        Minimum number of frames for a trajectory to be included in the final
        dataset.  Trajectories in the csv file with less datasets will be cut
        out.
    totvids : integer
        Total number of csv files to be compiled in the dataset.
    conversion: list of floats or integers
        Contains the frames per second associated with the video, the xy pixel
        resolution, and the z-stack depth respectively.  Currently the code
        discards z information.

    Returns
    ----------
    total1 : integer
        Total number of particles contained in all csv files being analyzed.
    frames : integer
        Total number of frames in the video.
    x_m : numpy array of dimensions frames x particles
        Contains x coordinates of all trajectories in all csv files being
        analyzed.  If a particle isn't present in a frame, then it is filled in
        with a 0.
    y_m : numpy array of dimensions frames x particles
        Similar to x_m with y coordinates.
    xs_m : numpy array of dimensions frames x particles
        Contains x coordinates of all trajectories in all csv files being
        analyzed.  Trajectories have been shifted such that all trajectories
        begin at frame 0.
    ys_m : numpy array of dimensions frames x particles
        Similar to xs_m with y coordinates.

    Examples
    ----------
    >>> n = 6
    >>> p = 2
    >>> df = np.zeros((p*n, 12))
    >>> for i in range(1, p+1):
            df[(i-1)*n:i*n, 0] = np.ones(n) + i - 1
            df[(i-1)*n:i*n, 1] = np.ones(n) + i - 1
            df[(i-1)*n:i*n, 2] = np.linspace(0, 10, n) + 2 + i
            df[(i-1)*n:i*n, 3] = np.linspace(0, 10, n) + i
            df[(i-1)*n:i*n, 4] = np.linspace(0, 10, n) + 3 + i
            df[(i-1)*n:i*n, 5] = np.zeros(n)
            df[(i-1)*n:i*n, 6:12] = np.zeros((n, 6))
    >>> np.savetxt("../Traj_test_data_1.tif.csv", df, delimiter=",")
    >>> folder = '../'
    >>> name = 'test_data'
    >>> MSD_iteration(folder, name)

    """

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
    """
    Calculates the geometrically averaged mean squared displacement of the input trajectories.

    Parameters
    ----------
    frames : integer
        Total number of frames in the video. Output from MSD_iteration
    total1 : integer
        Total number of particles contained in all csv files being analyzed.
        Output from MSD_iteration.
    xs_m : numpy array of dimensions frames x particles
        Contains x coordinates of all trajectories in all csv files being
        analyzed.  Trajectories have been shifted such that all trajectories
        begin at frame 0. Output from MSD_iteration.
    ys_m : numpy array of dimensions frames x particles
        Similar to xs_m with y coordinates. Output from MSD_iteration.

    Returns
    ----------
    geoM2xy : frames x 1 numpy.ndarray of float64s
        Average of the log 2D MSDs of input xy data.
    gSEM : frames x 1 numpy.ndarray of float64s
        Standard error of the log 2D MSDs of input xy data.
    SM1x : frames x total1 numpy.ndarray of float64s
        x component of the 2D MSD of input xy data for each trajectory.
    SM1y : frames x total1 numpy.ndarray of float64s
        y component of the 2D MSD of input xy data for each trajectory.
    SM2xy : frames x total1 numpy.ndarray of float64s
        2D MSDs of input xy data for each trajectory.

    Examples:
    ----------
    >>> n = 6
    >>> p = 2
    >>> df = np.zeros((p*n, 12))
    >>> for i in range(1, p+1):
            df[(i-1)*n:i*n, 0] = np.ones(n) + i - 1
            df[(i-1)*n:i*n, 1] = np.ones(n) + i - 1
            df[(i-1)*n:i*n, 2] = np.linspace(0, 10, n) + 2 + i
            df[(i-1)*n:i*n, 3] = np.linspace(0, 10, n) + i
            df[(i-1)*n:i*n, 4] = np.linspace(0, 10, n) + 3 + i
            df[(i-1)*n:i*n, 5] = np.zeros(n)
            df[(i-1)*n:i*n, 6:12] = np.zeros((n, 6))
    >>> np.savetxt("../Traj_test_data_1.tif.csv", df, delimiter=",")
    >>> folder = '../'
    >>> name = 'test_data'
    >>> cut = 1
    >>> totvids = 1
    >>> conversion = (1, 1, 1)
    >>> total1, frames, xs_m, ys_m, x_m, y_m = MSD_iteration(folder, name)
    >>> vectorized_MMSD_calcs(frames, total1, xs_m, ys_m)
    (array([ 0.        ,  1.38629436,  2.07944154,  2.99573227,  3.4657359 ,
             3.95124372,  4.27666612,  4.60517019,  4.85203026,  5.09986643,
             5.29831737,  0.        ,  0.        ,  0.        ]),
     array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]),
     array([[   0.,    0.],
            [   2.,    2.],
            [   4.,    4.],
            [  10.,   10.],
            [  16.,   16.],
            [  26.,   26.],
            [  36.,   36.],
            [  50.,   50.],
            [  64.,   64.],
            [  82.,   82.],
            [ 100.,  100.],
            [   0.,    0.],
            [   0.,    0.],
            [   0.,    0.]]),
     array([[   0.,    0.],
            [   2.,    2.],
            [   4.,    4.],
            [  10.,   10.],
            [  16.,   16.],
            [  26.,   26.],
            [  36.,   36.],
            [  50.,   50.],
            [  64.,   64.],
            [  82.,   82.],
            [ 100.,  100.],
            [   0.,    0.],
            [   0.,    0.],
            [   0.,    0.]]),
     array([[   0.,    0.],
            [   4.,    4.],
            [   8.,    8.],
            [  20.,   20.],
            [  32.,   32.],
            [  52.,   52.],
            [  72.,   72.],
            [ 100.,  100.],
            [ 128.,  128.],
            [ 164.,  164.],
            [ 200.,  200.],
            [   0.,    0.],
            [   0.,    0.],
            [   0.,    0.]]))

    """

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
