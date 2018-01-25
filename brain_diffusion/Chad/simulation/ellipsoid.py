from __future__ import division
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib as mpl


def mvee(points, tol=0.001):
    """
    Defines a minimum volume enclosing ellipsoid (MVEE) surrounding all the
    points contained in "points."  Points is a n x 3 array containing xyz data.
    If the dataset is too large, mvee may not work properly.  Hence the purpose
    of a maxtraj function to select the largest trajectories from a dataset.
    Use maxtraj first before sticking a large dataset into mvee.

    Outputs two arrays: A, which is the rotation matrix of the of the ellipsoid
    after centering, and c, which is the centroid of the ellipsoid.
    """
    N, d = points.shape

    Q = np.zeros([N, d+1])
    Q[:, 0:d] = points[0:N, 0:d]
    Q[:, d] = np.ones([1, N])

    Q = np.transpose(Q)
    points = np.transpose(points)
    count = 1
    err = 1
    u = (1/N) * np.ones(shape=(N,))

    while err > tol:

        X = np.dot(np.dot(Q, np.diag(u)), np.transpose(Q))
        M = np.diag(np.dot(np.dot(np.transpose(Q), la.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx] - d - 1)/((d+1)*(M[jdx] - 1))
        new_u = (1 - step_size)*u
        new_u[jdx] = new_u[jdx] + step_size
        count = count + 1
        err = la.norm(new_u - u)
        u = new_u

    U = np.diag(u)
    c = np.dot(points, u)
    A = (1/d) * la.inv(np.dot(np.dot(points, U), np.transpose(points)) - np.dot(c, np.transpose(c)))
    return A, np.transpose(c)


def ellipse(u, v):
    x = rx*cos(u)*cos(v)
    y = ry*sin(u)*cos(v)
    z = rz*sin(v)
    return x, y, z


def extrema(traj, n1, n2, frames):
    """
    n1: column containing particle data
    n2: start column of trajectory data
    frames: number of frames per trajectory

    extrema: array containing xyz data of start and end points of trajectories
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    position = traj[:, n2:n2+3]
    total = int(max(particles))
    total1 = total + 1
    path = dict()

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        path[num] = (position[min1:max1, :])

    pathmax = np.zeros((total-1, 3))
    pathmin = np.zeros((total-1, 3))
    # maxi = dict()

    for num in range(1, total):

        # maxi[num] = path[num].argmax(axis=0)
        pathmax[num-1, :] = path[num][0, :]
        pathmin[num-1, :] = path[num][frames-2, :]

    extrema = np.append(pathmax, pathmin, axis=0)
    return extrema


def enclosed_MSD(traj, n1, n2, n3, frames):
    """
    Creates a set of 6 points that will be used to form a diffusion ellipse
    based on MSD data. DEFUNCT

    n1: particle numbers
    n2: time
    n3: MSDs or Deffs
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    total = int(max(particles))
    total1 = total + 1
    rawtime = traj[:, n2]
    raw2DMSDs = traj[:, n3:n3+4]
    MSD = dict()
    time = dict()

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        MSD[num] = (raw2DMSDs[min1:max1, :])
        time[num] = (rawtime[min1:max1])

    MMSD = MSD[1]
    for num in range(2, total1):
        MMSD = MMSD + MSD[num]
    MMSD = MMSD/total1
    MMSD = MMSD[frames - 2, :]
    disp = np.sqrt(MMSD)

    pts = np.array([[disp[1],  disp[1],  0],
                    [-disp[1], -disp[1], 0],
                    [disp[1],  -disp[1], 0],
                    [-disp[1], disp[1],  0],
                    [disp[2],  0,        disp[2]],
                    [-disp[2], 0,        -disp[2]],
                    [-disp[2], 0,        disp[2]],
                    [disp[2],  0,        -disp[2]],
                    [0,       disp[3],   disp[3]],
                    [0,       -disp[3],  -disp[3]],
                    [0,       -disp[3],  disp[3]],
                    [0,       disp[3],   -disp[3]]])

    return pts


def maxtraj(traj, n1, n2, p, q):
    """
    Creates a 3-column matrix of xyz data of xyzmaxes and xyzmins from traj
    ectory dataset.

    traj: original trajectory dataset
    n1: particle numbers (0)
    n2: xyz data (8)
    p: percentile to include in dataset
    q: upper percentil to exclude
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    position = traj[:, n2:n2+3]
    total = int(max(particles))
    total1 = total + 1
    path = dict()
    noob = int(round(p*total))
    noob1 = int(round(q*total))

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        path[num] = (position[min1:max1, :])

        pathmax = np.zeros((total-1, 3))
        pathmin = np.zeros((total-1, 3))

        maxi = dict()
        mini = dict()
        maxes = np.zeros((6*(total-1), 3))

    for num in range(1, total):

        maxi[num] = path[num].argmax(axis=0)
        mini[num] = path[num].argmax(axis=0)
        maxes[num-1, 0:3] = path[num][maxi[num][0], :]
        maxes[(total-1) + num-1, 0:3] = path[num][mini[num][0], :]
        maxes[2*(total-1) + num-1, 0:3] = path[num][maxi[num][1], :]
        maxes[3*(total-1) + num-1, 0:3] = path[num][mini[num][1], :]
        maxes[4*(total-1) + num-1, 0:3] = path[num][maxi[num][2], :]
        maxes[5*(total-1) + num-1, 0:3] = path[num][mini[num][2], :]

    blank = np.zeros((maxes.shape[0], 1))

    for num in range(0, maxes.shape[0]):
        blank[num, 0] = maxes[num, 0]**2 + maxes[num, 1]**2 + maxes[num, 2]**2

    maxes = np.append(maxes, blank, 1)
    maxes = maxes[np.argsort(-maxes[:, 3])]
    maxes = maxes[noob1:noob, 0:3]

    return maxes


def plot_mvee(mtraj, dec, limit, filename):
    """
    Plots the MVEE of a dataset as determined by the function mvee.
    """
    pi = np.pi
    sin = np.sin
    cos = np.cos

    A, centroid = mvee(mtraj)
    U, D, V = la.svd(A)
    rx, ry, rz = 1./np.sqrt(D)
    u, v = np.mgrid[0:2*pi:20j, -pi/2:pi/2:10j]

    def ellipse(u, v):
        x = rx*cos(u)*cos(v)
        y = ry*sin(u)*cos(v)
        z = rz*sin(v)
        return x, y, z

    E = np.dstack(ellipse(u, v))
    E = np.dot(E, V) + centroid
    x, y, z = np.rollaxis(E, axis=-1)

    fig = plt.figure(figsize=(24, 18), dpi=80)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z, cstride=1, rstride=1, alpha=0.05)
    ax.scatter(mtraj[:, 0], mtraj[:, 1], mtraj[:, 2])

    axbox = ax.get_position()
    # ax.legend(loc=(0.86, 0.90), prop={'size': 20})
    ax.locator_params(nbins=6)
    ax.view_init(elev=38, azim=72)

    plt.gca().set_xlim([-limit, limit])
    plt.gca().set_ylim([-limit, limit])
    plt.gca().set_zlim([-limit, limit])

    for item in ([ax.xaxis.label, ax.yaxis.label, ax.zaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
        item.set_fontsize(13)

    ax.title.set_fontsize(35)
    ax.tick_params(direction='out', pad=16)
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f um'.format(dec)))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f um'.format(dec)))
    plt.gca().zaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f um'.format(dec)))

    plt.show()
    # Save your figure
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')


def plot_mveeoverl(traj, n1, n2, p, q, scale, dec, filename, limit):
    """
    This function creates a single 3D plot from trajectory data.  This dataset
    must include a column of particle numbers as well as the x, y, and z
    coordinates of of each particle at each frame. Output will be saved as a
    .png file of the desired name.
    Inputs:
    traj: array of trajectory data e.g. particle #, frames, x, y, z, Deff, MSD
    n1: particle# column
    n2: xyz data start (so x data column, 29 for a normal dataset)
    a range)
    dec: how many decimals you would like to be displayed in the graph.
    filename: what you want to name the file.  Must be in ''.
    limit defines the range of z, y, and z.
    Can also use plt.show() afterwards to preview your data, even if it skews the title and legend a bit.
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    position = traj[:, n2:n2+4]
    total = int(max(particles))
    total1 = total + 1
    path = dict()

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        path[num] = (position[min1:max1, :])

    but = maxtraj(traj, n1, n2, p, q)

    pi = np.pi
    sin = np.sin
    cos = np.cos

    A, centroid = mvee(but)
    A = scale * A
    U, D, V = la.svd(A)
    rx, ry, rz = 1./np.sqrt(D)
    u, v = np.mgrid[0:2*pi:20j, -pi/2:pi/2:10j]

    def ellipse(u, v):
        x = rx*cos(u)*cos(v)
        y = ry*sin(u)*cos(v)
        z = rz*sin(v)
        return x, y, z

    E = np.dstack(ellipse(u, v))
    E = np.dot(E, V) + centroid
    x, y, z = np.rollaxis(E, axis=-1)

    # Creates figure
    fig = plt.figure(figsize=(24, 18), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Particle Trajectories', x=0.5, y=1.15)

    # Plots individual trajectories
    for num in range(1, total1):

        ax.plot(path[num][:, 0], path[num][:, 1], path[num][:, 2], label='Particle {}'.format(num))

    ax.plot_surface(x, y, z, cstride=1, rstride=1, alpha=0.05)

    axbox = ax.get_position()
    # ax.legend(loc=(0.86, 0.90), prop={'size': 20})
    ax.locator_params(nbins=6)
    ax.view_init(elev=38, azim=72)

    plt.gca().set_xlim([-limit, limit])
    plt.gca().set_ylim([-limit, limit])
    plt.gca().set_zlim([-limit, limit])

    for item in ([ax.xaxis.label, ax.yaxis.label, ax.zaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
        item.set_fontsize(13)

    ax.title.set_fontsize(35)
    ax.tick_params(direction='out', pad=16)
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f um'.format(dec)))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f um'.format(dec)))
    plt.gca().zaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f um'.format(dec)))

    # Save your figure
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')


def plot_mveeoverl2(traj, n1, n2, p, q, scale, dec, filename, limit):
    """
    This function creates a single 3D plot from trajectory data.  This dataset
    must include a column of particle numbers as well as the x, y, and z
    coordinates of of each particle at each frame. Output will be saved as a
    .png file of the desired name.
    Inputs:
    traj: array of trajectory data e.g. particle #, frames, x, y, z, Deff, MSD
    n1: particle# column
    n2: xyz data start (so x data column, 29 for a normal dataset)
    a range)
    dec: how many decimals you would like to be displayed in the graph.
    filename: what you want to name the file.  Must be in ''.
    limit defines the range of z, y, and z.
    Can also use plt.show() afterwards to preview your data, even if it skews the title and legend a bit.

    This differs from plot_mveeoverl in that it excludes the maxtraj function.
    The ellipse is based off ALL points in the dataset.  This is good for small
    er datasets.
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    position = traj[:, n2:n2+3]
    total = int(max(particles))
    total1 = total + 1
    path = dict()

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        path[num] = (position[min1:max1, :])

    # but = maxtraj(traj, n1, n2, p, q)

    pi = np.pi
    sin = np.sin
    cos = np.cos

    A, centroid = mvee(position)
    A = scale * A
    U, D, V = la.svd(A)
    rx, ry, rz = 1./np.sqrt(D)
    u, v = np.mgrid[0:2*pi:20j, -pi/2:pi/2:10j]

    def ellipse(u, v):
        x = rx*cos(u)*cos(v)
        y = ry*sin(u)*cos(v)
        z = rz*sin(v)
        return x, y, z

    E = np.dstack(ellipse(u, v))
    E = np.dot(E, V) + centroid
    x, y, z = np.rollaxis(E, axis=-1)

    # Creates figure
    fig = plt.figure(figsize=(24, 18), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Particle Trajectories', x=0.5, y=1.15)

    # Plots individual trajectories
    for num in range(1, total1):

        ax.plot(path[num][:, 0], path[num][:, 1], path[num][:, 2], label='Particle {}'.format(num))

    ax.plot_surface(x, y, z, cstride=1, rstride=1, alpha=0.05)

    axbox = ax.get_position()
    # ax.legend(loc=(0.86, 0.90), prop={'size': 20})
    ax.locator_params(nbins=6)
    ax.view_init(elev=38, azim=72)

    plt.gca().set_xlim([-limit, limit])
    plt.gca().set_ylim([-limit, limit])
    plt.gca().set_zlim([-limit, limit])

    for item in ([ax.xaxis.label, ax.yaxis.label, ax.zaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
        item.set_fontsize(13)

    ax.title.set_fontsize(35)
    ax.tick_params(direction='out', pad=16)
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f um'.format(dec)))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f um'.format(dec)))
    plt.gca().zaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f um'.format(dec)))

    plt.show()
    # Save your figure
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')


def rotmat(traj, n1, n2, A):
    """
    A similar function to roti, this function will rotate any given three-coor
    dinate dataset of trajectories (traj) according to the given rotation
    matrix A.  This matrix can be constructed manually or can be determined
    from the funtion mvee.

    Inputs:
    traj: trajectory dataset
    n1: particle number column (normally 0)
    n2: first column in xyz dataset (others are assumed to follow)
    A: rotation matrix (3 x 3 numpy array)

    Outputs:
    A modified form of traj with the columns defined by n2 being replaced with
    new rotated coordinates.
    """

    rotate = np.zeros((traj.shape[0], 3))

    rotate[:, 0] = traj[:, n2]
    rotate[:, 1] = traj[:, n2 + 1]
    rotate[:, 2] = traj[:, n2 + 2]
    rotpaths = np.transpose(np.dot(A, np.transpose(rotate)))

    traj[:, n2] = rotpaths[:, 0]
    traj[:, n2 + 1] = rotpaths[:, 1]
    traj[:, n2 + 2] = rotpaths[:, 2]

    return traj


def rotmat2(traj, n1, n2, p, q):
    """
    A similar function to rotmat, this function will rotate any given three-coor
    dinate dataset of trajectories (traj) to align it with the xyz axes.  This
    matrix can be constructed manually or can be determined from the funtion
    mvee.

    Inputs:
    traj: trajectory dataset
    n1: particle number column (normally 0)
    n2: first column in xyz dataset (others are assumed to follow)
    A: rotation matrix (3 x 3 numpy array)

    Outputs:
    A modified form of traj with the columns defined by n2 being replaced with
    new rotated coordinates.
    """

    rotate = np.zeros((traj.shape[0], 3))
    but = maxtraj(traj, n1, n2, p, q)
    A, centroid = mvee(but)
    U, D, V = la.svd(A)

    rotate[:, 0] = traj[:, n2]
    rotate[:, 1] = traj[:, n2 + 1]
    rotate[:, 2] = traj[:, n2 + 2]
    rotpaths = np.transpose(np.dot(V, np.transpose(rotate)))

    traj[:, n2] = rotpaths[:, 0]
    traj[:, n2 + 1] = rotpaths[:, 1]
    traj[:, n2 + 2] = rotpaths[:, 2]

    return traj


def ellipsax(traj, n1, n2, n3, t):
    """
    Calculates the in all three directions, xyz, for a diffusion dataset that
    has been pre-rotated using rotmat2.  The MSDs are calculated at the speci
    fied frame t.  Due to some confusion in my random dataset generator, there
    is no distinction between the time column and the frame column.  This makes
    this code less robust atm.  I need certain aspects of the code to be int
    egers and certain aspects to reflect time accurately.  This is fine for a
    random dataset, but not a real dataset.  Also, this code will NOT work if
    there are missing frames in the dataset.

    This code is used as an input function to "MSDS" to calculate for all times
    in the dataset.

    Outputs data in an 1 X 3 array (MSD) representing xyz msds as well as a
    time/frame column for a single particle (which should be identical for all
    particles for the code to work).

    n1: particle number column
    n2: position column (x)
    n3: time/frame column
    t: specified frame at which to calculate MSDs
    """
    particles = traj[:, n1]
    position = traj[:, n2:n2+3]
    total = int(max(particles))
    total1 = total + 1
    path = dict()
    time = dict()
    spreadxyz = np.zeros((total, 3))

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)

        time[num] = traj[min1:max1, n3]
        hold1 = np.where(time[num] > t)
        itindex1 = hold1[0][0]

        path[num] = (position[min1:max1, :])[0:itindex1]
        # spreadxyz[num-1] = np.square(np.ptp(path[num], axis=0))
        spreadxyz[num-1] = np.square(path[num][t, :]-path[num][0, :])

    mx = np.average(spreadxyz[:, 0])
    my = np.average(spreadxyz[:, 1])
    mz = np.average(spreadxyz[:, 2])
    mt = mx + my + mz
    MSD = np.array([mx, my, mz, mt])

    return MSD, time[1]


def MSDS(traj, n1, n2, n3):
    """
    This code uses ellipsax to create an array of MSDs at every timepoint in a
    trajectory dataset.

    Output is in the form of a time X 3 array (msds), as well as a time/frame
    array (frames).
    """

    msd, frames = ellipsax(traj, n1, n2, n3, 0)
    msds = np.zeros((int(max(frames)), 4))
    for num in range(1, int(max(frames))):
        msds[num, :], frames = ellipsax(traj, n1, n2, n3, num)
    return msds, frames


def andiff(MMSD, frames):
    """
    Calculates the 3 principle diffusion coefficients of an anisotropic diff
    usion dataset that has been properly rotated using rotmat2.  Uses the funct
    ion "MSDS" to perform calculations.
    """

    frames1 = frames[0:frames.shape[0]-1]
    diff = np.zeros(MMSD.shape)
    diff[:, 0] = MMSD[:, 0]/(2*frames1)
    diff[:, 1] = MMSD[:, 1]/(2*frames1)
    diff[:, 2] = MMSD[:, 2]/(2*frames1)
    diff[:, 3] = MMSD[:, 3]/(6*frames1)

    return diff, frames


def plot_anisoMSDS(MMSD, frames, dec, filename, limit1, limit2):
    """
    Calculates the 3 principles MSDs (xyz) of a pre-rotated dataset using "MSDS"
    and plots them.
    """

    frames1 = frames[0:frames.shape[0]-1]

    # Creates figure
    fig = plt.figure(figsize=(24, 18), dpi=80)
    ax = fig.add_subplot(111)
    # ax.set_title('Particle Trajectories', x=0.5, y=1.15)

    ax.plot(frames1, MMSD[:, 0], linewidth=2.5, label='x MSD')
    ax.plot(frames1, MMSD[:, 1], linewidth=2.5, label='y MSD')
    ax.plot(frames1, MMSD[:, 2], linewidth=2.5, label='z MSD')

    # A few adjustments to prettify the graph
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(36)

    ax.title.set_fontsize(45)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MSD (um2)')
    ax.tick_params(direction='out', pad=16)
    ax.legend(loc=(0.76, 0.76), prop={'size': 35})
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))

    # plt.gca().set_xlim([-limit, limit])
    plt.gca().set_ylim([limit1, limit2])

    # Save your figure
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')


def plot_anisodiff(diff, frames, dec, filename, limit1, limit2):
    """
    Calculates the 3 principle diffusion coefficients of a pre-rotated trajector
    y dataset and plots them.

    n1: particle numbers
    n2: first column in xyz dataset
    n3: time or frame column
    n4: MSD column (11 in random dataset)
    """

    frames1 = frames[0:frames.shape[0]-1]

    # Creates figure
    fig = plt.figure(figsize=(24, 18), dpi=80)
    ax = fig.add_subplot(111)
    # ax.set_title('Particle Trajectories', x=0.5, y=1.15)

    ax.plot(frames1, diff[:, 3], linewidth=2.5, label='3D D')
    ax.plot(frames1, diff[:, 0], linewidth=2.5, label='Dx')
    ax.plot(frames1, diff[:, 1], linewidth=2.5, label='Dy')
    ax.plot(frames1, diff[:, 2], linewidth=2.5, label='Dz')

    # A few adjustments to prettify the graph
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(36)

    ax.title.set_fontsize(45)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Diffusion Coefficient (um2/s)')
    ax.tick_params(direction='out', pad=16)
    ax.legend(loc=(0.76, 0.76), prop={'size': 35})
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))

    # plt.gca().set_xlim([-limit, limit])
    plt.gca().set_ylim([limit1, limit2])

    # Save your figure
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')

    return diff
