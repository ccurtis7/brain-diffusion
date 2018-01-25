import numpy as np
import os
import csv
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt


def randaniso(b, s, f, p, xs, ys, zs):
    """
    Builds a single random anisotropic trajectory without using spherical
    coordinates, as randtraj does. Anisotropic behavior is determined by the
    "stretching" coefficients xs ys zs. Default is 1.

    b: base magnitude of single step
    s: variation in step size
    f: number of frames or steps to Takes
    p: particle number (should be 1 for now)
    xs: x stretching coefficient
    ys: y stretching coefficient
    zs: z stretching coefficient

    Output:
    0 particle number
    1 time or frames
    2 x movement
    3 angle 1 (not used)
    4 angle 2 (note used)
    5 x coordinate
    6 y coordinate
    7 z coordinate
    8 centered x coordinate
    9 centered y coordinate
    10 centered z coordinate
    11 MSD
    12 2D xy MSD
    13 2D xz MSD
    14 2D yz MSD
    15 Diffusion Coefficient (Deff)
    16 2D xy Deff
    17 2D xz Deff
    18 2D yz Deff
    19 y movement
    20 z movement
    """

    base = b
    step = s
    pi = 3.14159265359
    frames = f

    ttraject = np.zeros((frames, 22))

    for num in range(1, frames):

        # Create particle number
        ttraject[num, 0] = p
        ttraject[num-1, 0] = p
        # Create frame
        ttraject[num, 1] = 1 + ttraject[num-1, 1]
        # Create magnitude vectors
        ttraject[num, 2] = xs*base*(random.random()-0.5)
        ttraject[num, 19] = ys*base*(random.random()-0.5)
        ttraject[num, 20] = zs*base*(random.random()-0.5)
        # Create Angle Vectors
        # ttraject[num, 3] = 2 * pi * random.random()
        # ttraject[num, 4] = pi * random.random()
        # Build trajectories
        ttraject[num, 5] = (ttraject[num-1, 5] + ttraject[num, 2])
        ttraject[num, 6] = (ttraject[num-1, 6] + ttraject[num, 19])
        ttraject[num, 7] = (ttraject[num-1, 7] + ttraject[num, 20])

    particle = ttraject[:, 0]
    time = ttraject[:, 1]
    x = ttraject[:, 5]
    y = ttraject[:, 6]
    z = ttraject[:, 7]

    ttraject[:, 8] = (x - ((max(x)+min(x))/2))
    cx = ttraject[:, 8]
    ttraject[:, 9] = (y - ((max(y)+min(y))/2))
    cy = ttraject[:, 9]
    ttraject[:, 10] = (z - ((max(z)+min(z))/2))
    cz = ttraject[:, 10]

    # Calculate MSDs and Deffs
    for num in range(1, frames):

        ttraject[num, 11] = ((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 9]-ttraject[0, 9])**2 +
                             (ttraject[num, 10]-ttraject[0, 10])**2)
        ttraject[num, 12] = ((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 9]-ttraject[0, 9])**2)
        ttraject[num, 13] = ((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 10]-ttraject[0, 10])**2)
        ttraject[num, 14] = ((ttraject[num, 10]-ttraject[0, 10])**2 + (ttraject[num, 9]-ttraject[0, 9])**2)

        ttraject[num, 15] = ttraject[num, 11]/(6*ttraject[num, 1])
        ttraject[num, 16] = ttraject[num, 12]/(4*ttraject[num, 1])
        ttraject[num, 17] = ttraject[num, 13]/(4*ttraject[num, 1])
        ttraject[num, 18] = ttraject[num, 14]/(4*ttraject[num, 1])

    MSD = ttraject[:, 11]
    MSDxy = ttraject[:, 12]
    MSDxz = ttraject[:, 13]
    MSDyz = ttraject[:, 14]

    Deff = ttraject[:, 15]
    Deffxy = ttraject[:, 16]
    Deffxz = ttraject[:, 17]
    Deffyz = ttraject[:, 18]

    return ttraject


def randanisorot(b, s, f, p, xs, ys, zs, th1, th2, th3):
    """
    Builds a single random anisotropic trajectory without using spherical
    coordinates, as randtraj does. Anisotropic behavior is determined by the
    "stretching" coefficients xs ys zs. Default is 1.  Also rotates the
    trajectories by the given input angles

    b: base magnitude of single step
    s: variation in step size
    f: number of frames or steps to Takes
    p: particle number (should be 1 for now)
    xs: x stretching coefficient
    ys: y stretching coefficient
    zs: z stretching coefficient
    th1: x angle of rotation
    th2: y angle of rotation
    th3: z angle of rotation

    Output:
    0 particle number
    1 time or frames
    2 x movement
    3 angle 1 (not used)
    4 angle 2 (note used)
    5 x coordinate
    6 y coordinate
    7 z coordinate
    8 centered x coordinate
    9 centered y coordinate
    10 centered z coordinate
    11 MSD
    12 2D xy MSD
    13 2D xz MSD
    14 2D yz MSD
    15 Diffusion Coefficient (Deff)
    16 2D xy Deff
    17 2D xz Deff
    18 2D yz Deff
    19 y movement
    20 z movement
    """

    base = b
    step = s
    pi = 3.14159265359
    frames = f

    ttraject = np.zeros((frames, 22))
    rotate = np.zeros((frames,3))

    for num in range(1, frames):

        # Create particle number
        ttraject[num, 0] = p
        ttraject[num-1, 0] = p
        # Create frame
        ttraject[num, 1] = 1 + ttraject[num-1, 1]
        # Create magnitude vectors
        ttraject[num, 2] = xs*base*(random.random()-0.5)
        ttraject[num, 19] = ys*base*(random.random()-0.5)
        ttraject[num, 20] = zs*base*(random.random()-0.5)
        # Create Angle Vectors
        # ttraject[num, 3] = 2 * pi * random.random()
        # ttraject[num, 4] = pi * random.random()
        # Build trajectories
        ttraject[num, 5] = (ttraject[num-1, 5] + ttraject[num, 2])
        ttraject[num, 6] = (ttraject[num-1, 6] + ttraject[num, 19])
        ttraject[num, 7] = (ttraject[num-1, 7] + ttraject[num, 20])

    # rotation matrix for x
    R1 = np.array([[1, 0, 0],
                   [0, np.cos(th1), -np.sin(th1)],
                   [0, np.sin(th1), np.cos(th1)]])
    # rotation matrix for y
    R2 = np.array([[np.cos(th2), 0, np.sin(th2)],
                   [0, 1, 0],
                   [-np.sin(th2), 0, np.cos(th2)]])
    # rotation matrix for z
    R3 = np.array([[np.cos(th3), -np.sin(th3), 0],
                   [np.sin(th3), np.cos(th3), 0],
                   [0, 0, 1]])
    # total rotation matrix
    R = np.dot(np.dot(R1, R2), R3)

    particle = ttraject[:, 0]
    time = ttraject[:, 1]
    x = ttraject[:, 5]
    y = ttraject[:, 6]
    z = ttraject[:, 7]

    rotate[:, 0] = (x - ((max(x)+min(x))/2))
    rotate[:, 1] = (y - ((max(y)+min(y))/2))
    rotate[:, 2] = (z - ((max(z)+min(z))/2))
    rotpaths = np.transpose(np.dot(R, np.transpose(rotate)))

    ttraject[:, 8] = rotpaths[:, 0]
    cx = ttraject[:, 8]
    ttraject[:, 9] = rotpaths[:, 1]
    cy = ttraject[:, 9]
    ttraject[:, 10] = rotpaths[:, 2]
    cz = ttraject[:, 10]

    # Calculate MSDs and Deffs
    for num in range(1, frames):

        ttraject[num, 11] = ((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 9]-ttraject[0, 9])**2 +
                             (ttraject[num, 10]-ttraject[0, 10])**2)
        ttraject[num, 12] = ((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 9]-ttraject[0, 9])**2)
        ttraject[num, 13] = ((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 10]-ttraject[0, 10])**2)
        ttraject[num, 14] = ((ttraject[num, 10]-ttraject[0, 10])**2 + (ttraject[num, 9]-ttraject[0, 9])**2)

        ttraject[num, 15] = ttraject[num, 11]/(6*ttraject[num, 1])
        ttraject[num, 16] = ttraject[num, 12]/(4*ttraject[num, 1])
        ttraject[num, 17] = ttraject[num, 13]/(4*ttraject[num, 1])
        ttraject[num, 18] = ttraject[num, 14]/(4*ttraject[num, 1])

    MSD = ttraject[:, 11]
    MSDxy = ttraject[:, 12]
    MSDxz = ttraject[:, 13]
    MSDyz = ttraject[:, 14]

    Deff = ttraject[:, 15]
    Deffxy = ttraject[:, 16]
    Deffxz = ttraject[:, 17]
    Deffyz = ttraject[:, 18]

    return ttraject


def randconv(b, s, f, p, xc, yc, zc):
    """
    Builds a single random trajectory with a convection term. Magnitude of the
    convection term is determined by xc, yc, and zc. Default is 0.

    b: base magnitude of single step
    s: variation in step size
    f: number of frames or steps to Takes
    p: particle number (should be 1 for now)
    xc: x convection
    yc: y convection
    zc: z convection

    Output:
    0 particle number
    1 time or frames
    2 x movement
    3 angle 1 (not used)
    4 angle 2 (note used)
    5 x coordinate
    6 y coordinate
    7 z coordinate
    8 centered x coordinate
    9 centered y coordinate
    10 centered z coordinate
    11 MSD
    12 2D xy MSD
    13 2D xz MSD
    14 2D yz MSD
    15 Diffusion Coefficient (Deff)
    16 2D xy Deff
    17 2D xz Deff
    18 2D yz Deff
    19 y movement
    20 z movement
    """

    base = b
    step = s
    pi = 3.14159265359
    frames = f

    ttraject = np.zeros((frames, 22))

    for num in range(1, frames):

        # Create particle number
        ttraject[num, 0] = p
        ttraject[num-1, 0] = p
        # Create frame
        ttraject[num, 1] = 1 + ttraject[num-1, 1]
        # Create magnitude vectors
        ttraject[num, 2] = base*(random.random()-0.5)
        ttraject[num, 19] = base*(random.random()-0.5)
        ttraject[num, 20] = base*(random.random()-0.5)
        # Create Angle Vectors
        # ttraject[num, 3] = 2 * pi * random.random()
        # ttraject[num, 4] = pi * random.random()
        # Build trajectories
        ttraject[num, 5] = ttraject[num-1, 5] + ttraject[num, 2] + xc
        ttraject[num, 6] = ttraject[num-1, 6] + ttraject[num, 19] + yc
        ttraject[num, 7] = ttraject[num-1, 7] + ttraject[num, 20] + zc

    particle = ttraject[:, 0]
    time = ttraject[:, 1]
    x = ttraject[:, 5]
    y = ttraject[:, 6]
    z = ttraject[:, 7]

    ttraject[:, 8] = x - ((max(x)+min(x))/2)
    cx = ttraject[:, 8]
    ttraject[:, 9] = y - ((max(y)+min(y))/2)
    cy = ttraject[:, 9]
    ttraject[:, 10] = z - ((max(z)+min(z))/2)
    cz = ttraject[:, 10]

    # Calculate MSDs and Deffs
    for num in range(1, frames):

        ttraject[num, 11] = ((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 9]-ttraject[0, 9])**2 +
                             (ttraject[num, 10]-ttraject[0, 10])**2)
        ttraject[num, 12] = ((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 9]-ttraject[0, 9])**2)
        ttraject[num, 13] = ((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 10]-ttraject[0, 10])**2)
        ttraject[num, 14] = ((ttraject[num, 10]-ttraject[0, 10])**2 + (ttraject[num, 9]-ttraject[0, 9])**2)

        ttraject[num, 15] = ttraject[num, 11]/(6*ttraject[num, 1])
        ttraject[num, 16] = ttraject[num, 12]/(4*ttraject[num, 1])
        ttraject[num, 17] = ttraject[num, 13]/(4*ttraject[num, 1])
        ttraject[num, 18] = ttraject[num, 14]/(4*ttraject[num, 1])

    MSD = ttraject[:, 11]
    MSDxy = ttraject[:, 12]
    MSDxz = ttraject[:, 13]
    MSDyz = ttraject[:, 14]

    Deff = ttraject[:, 15]
    Deffxy = ttraject[:, 16]
    Deffxz = ttraject[:, 17]
    Deffyz = ttraject[:, 18]

    return ttraject


def multrandaniso(b, s, f, p, xs, ys, zs):
    """
    Builds an array of multiple trajectories appended to each other. Number of
    trajectories is determined by p.
    """

    parts = p
    one = randaniso(b, s, f, 1, xs, ys, zs)
    counter = 1

    while counter < p + 1:
        counter = counter + 1
        one = np.append(one, randaniso(b, s, f, counter, xs, ys, zs), axis=0)

    return one


def multrandanisorot(b, s, f, p, xs, ys, zs, th1, th2, th3):
    """
    Builds an array of multiple trajectories appended to each other. Number of
    trajectories is determined by p.  Also rotated.
    """

    parts = p
    one = randanisorot(b, s, f, 1, xs, ys, zs, th1, th2, th3)
    counter = 1

    while counter < p + 1:
        counter = counter + 1
        one = np.append(one, randanisorot(b, s, f, counter, xs, ys, zs, th1, th2, th3), axis=0)

    return one


def multrandconv(b, s, f, p, xc, yc, zc):
    """
    Builds an array of multiple trajectories appended to each other. Number of
    trajectories is determined by p.
    """

    parts = p
    one = randconv(b, s, f, 1, xc, yc, zc)
    counter = 1

    while counter < p + 1:
        counter = counter + 1
        one = np.append(one, randconv(b, s, f, counter, xc, yc, zc), axis=0)

    return one


def multrandsamp(b, s, f, p, samp):
    """
    Builds an array of multiple trajectories appended to each other. Number of
    trajectories is determined by p.
    """

    parts = p
    one = randsamp(b, s, f, 1, samp)
    counter = 1

    while counter < p + 1:
        counter = counter + 1
        one = np.append(one, randsamp(b, s, f, counter, samp), axis=0)

    return one


def multrandall(b, s, f, p, xc, yc, zc, xs, ys, zs, samp):
    """
    Builds an array of multiple trajectories appended to each other. Number of
    trajectories is determined by p.
    """

    parts = p
    one = randall(b, s, f, 1, xc, yc, zc, xs, ys, zs, samp)
    counter = 1

    while counter < p + 1:
        counter = counter + 1
        one = np.append(one, randall(b, s, f, counter, xc, yc, zc, xs, ys, zs, samp), axis=0)

    return one


def randsamp(b, s, f, p, samp):
    """
    Builds a single random trajectory with poor sampling time. Samp term
    determines how many frames are deleted from overall trajectory.

    b: base magnitude of single step
    s: variation in step size
    f: number of frames or steps to Takes
    p: particle number (should be 1 for now)
    sampl: sampling factor

    Output:
    0 particle number
    1 time or frames
    2 x movement
    3 angle 1 (not used)
    4 angle 2 (note used)
    5 x coordinate
    6 y coordinate
    7 z coordinate
    8 centered x coordinate
    9 centered y coordinate
    10 centered z coordinate
    11 MSD
    12 2D xy MSD
    13 2D xz MSD
    14 2D yz MSD
    15 Diffusion Coefficient (Deff)
    16 2D xy Deff
    17 2D xz Deff
    18 2D yz Deff
    19 y movement
    20 z movement
    """

    base = b
    step = s
    pi = 3.14159265359
    frames = f

    ttraject = np.zeros((frames, 22))

    for num in range(1, frames):

        # Create particle number
        ttraject[num, 0] = p
        ttraject[num-1, 0] = p
        # Create frame
        ttraject[num, 1] = 1 + ttraject[num-1, 1]
        # Create magnitude vectors
        ttraject[num, 2] = base*(random.random()-0.5)
        ttraject[num, 19] = base*(random.random()-0.5)
        ttraject[num, 20] = base*(random.random()-0.5)
        # Create Angle Vectors
        # ttraject[num, 3] = 2 * pi * random.random()
        # ttraject[num, 4] = pi * random.random()
        # Build trajectories
        ttraject[num, 5] = ttraject[num-1, 5] + ttraject[num, 2]
        ttraject[num, 6] = ttraject[num-1, 6] + ttraject[num, 19]
        ttraject[num, 7] = ttraject[num-1, 7] + ttraject[num, 20]

    particle = ttraject[:, 0]
    time = ttraject[:, 1]
    x = ttraject[:, 5]
    y = ttraject[:, 6]
    z = ttraject[:, 7]

    ttraject[:, 8] = x - ((max(x)+min(x))/2)
    cx = ttraject[:, 8]
    ttraject[:, 9] = y - ((max(y)+min(y))/2)
    cy = ttraject[:, 9]
    ttraject[:, 10] = z - ((max(z)+min(z))/2)
    cz = ttraject[:, 10]

    # Calculate MSDs and Deffs
    for num in range(1, frames):

        ttraject[num, 11] = ((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 9]-ttraject[0, 9])**2 +
                             (ttraject[num, 10]-ttraject[0, 10])**2)
        ttraject[num, 12] = ((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 9]-ttraject[0, 9])**2)
        ttraject[num, 13] = ((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 10]-ttraject[0, 10])**2)
        ttraject[num, 14] = ((ttraject[num, 10]-ttraject[0, 10])**2 + (ttraject[num, 9]-ttraject[0, 9])**2)

        ttraject[num, 15] = ttraject[num, 11]/(6*ttraject[num, 1])
        ttraject[num, 16] = ttraject[num, 12]/(4*ttraject[num, 1])
        ttraject[num, 17] = ttraject[num, 13]/(4*ttraject[num, 1])
        ttraject[num, 18] = ttraject[num, 14]/(4*ttraject[num, 1])

    MSD = ttraject[:, 11]
    MSDxy = ttraject[:, 12]
    MSDxz = ttraject[:, 13]
    MSDyz = ttraject[:, 14]

    Deff = ttraject[:, 15]
    Deffxy = ttraject[:, 16]
    Deffxz = ttraject[:, 17]
    Deffyz = ttraject[:, 18]

    ttraject2a = ttraject[0::samp, :]
    ttraject2 = np.vstack([ttraject2a, ttraject[-1, :]])

    return ttraject2


def randall(b, s, f, p, xc, yc, zc, xs, ys, zs, samp):
    """
    Builds a single random trajectory with all of the above. Samp term
    determines how many frames are deleted from overall trajectory.

    b: base magnitude of single step
    s: variation in step size
    f: number of frames or steps to Takes
    p: particle number (should be 1 for now)
    sampl: sampling factor

    Output:
    0 particle number
    1 time or frames
    2 x movement
    3 angle 1 (not used)
    4 angle 2 (note used)
    5 x coordinate
    6 y coordinate
    7 z coordinate
    8 centered x coordinate
    9 centered y coordinate
    10 centered z coordinate
    11 MSD
    12 2D xy MSD
    13 2D xz MSD
    14 2D yz MSD
    15 Diffusion Coefficient (Deff)
    16 2D xy Deff
    17 2D xz Deff
    18 2D yz Deff
    19 y movement
    20 z movement
    """

    base = b
    step = s
    pi = 3.14159265359
    frames = f

    ttraject = np.zeros((frames, 22))

    for num in range(1, frames):

        # Create particle number
        ttraject[num, 0] = p
        ttraject[num-1, 0] = p
        # Create frame
        ttraject[num, 1] = 1 + ttraject[num-1, 1]
        # Create magnitude vectors
        ttraject[num, 2] = xs*base*(random.random()-0.5)
        ttraject[num, 19] = ys*base*(random.random()-0.5)
        ttraject[num, 20] = zs*base*(random.random()-0.5)
        # Create Angle Vectors
        # ttraject[num, 3] = 2 * pi * random.random()
        # ttraject[num, 4] = pi * random.random()
        # Build trajectories
        ttraject[num, 5] = ttraject[num-1, 5] + ttraject[num, 2] + xc
        ttraject[num, 6] = ttraject[num-1, 6] + ttraject[num, 19] + yc
        ttraject[num, 7] = ttraject[num-1, 7] + ttraject[num, 20] + zc

    particle = ttraject[:, 0]
    time = ttraject[:, 1]
    x = ttraject[:, 5]
    y = ttraject[:, 6]
    z = ttraject[:, 7]

    ttraject[:, 8] = x - ((max(x)+min(x))/2)
    cx = ttraject[:, 8]
    ttraject[:, 9] = y - ((max(y)+min(y))/2)
    cy = ttraject[:, 9]
    ttraject[:, 10] = z - ((max(z)+min(z))/2)
    cz = ttraject[:, 10]

    # Calculate MSDs and Deffs
    for num in range(1, frames):

        ttraject[num, 11] = ((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 9]-ttraject[0, 9])**2 +
                             (ttraject[num, 10]-ttraject[0, 10])**2)
        ttraject[num, 12] = ((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 9]-ttraject[0, 9])**2)
        ttraject[num, 13] = ((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 10]-ttraject[0, 10])**2)
        ttraject[num, 14] = ((ttraject[num, 10]-ttraject[0, 10])**2 + (ttraject[num, 9]-ttraject[0, 9])**2)

        ttraject[num, 15] = ttraject[num, 11]/(6*ttraject[num, 1])
        ttraject[num, 16] = ttraject[num, 12]/(4*ttraject[num, 1])
        ttraject[num, 17] = ttraject[num, 13]/(4*ttraject[num, 1])
        ttraject[num, 18] = ttraject[num, 14]/(4*ttraject[num, 1])

    MSD = ttraject[:, 11]
    MSDxy = ttraject[:, 12]
    MSDxz = ttraject[:, 13]
    MSDyz = ttraject[:, 14]

    Deff = ttraject[:, 15]
    Deffxy = ttraject[:, 16]
    Deffxz = ttraject[:, 17]
    Deffyz = ttraject[:, 18]

    ttraject2 = ttraject[0::samp, :]

    return ttraject2
