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


def randplane(b, s, f, p, pnor, ppoi):
    """
    Builds a single random trajectory with a convection term. Magnitude of the
    convection term is determined by xc, yc, and zc. Default is 0.

    b: base magnitude of single step
    s: variation in step size
    f: number of frames or steps to Takes
    p: particle number (should be 1 for now)
    pnor: the normal defining the plane of interest as a 3-element numpy array
    ppoi: a point that falls on the plane of interest as a 3-element numpy array

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

    # Define a plane
    nor = pnor
    poi = ppoi

    ttraject = np.zeros((frames, 22))
    ttraject[0, 0] = p

    for num in range(1, frames):

        # Create particle number
        ttraject[num, 0] = p

        # Create frame
        ttraject[num, 1] = 1 + ttraject[num-1, 1]

        # Create magnitude vectors
        ttraject[num, 2] = base*(random.random()-0.5)
        ttraject[num, 19] = base*(random.random()-0.5)
        ttraject[num, 20] = base*(random.random()-0.5)

        # Build preliminary trajectories
        ttraject[num, 5] = ttraject[num-1, 5] + ttraject[num, 2]
        ttraject[num, 6] = ttraject[num-1, 6] + ttraject[num, 19]
        ttraject[num, 7] = ttraject[num-1, 7] + ttraject[num, 20]

        # Test to see if trajectories cross boundary
        p0 = np.array([ttraject[num-1, 5], ttraject[num-1, 6], ttraject[num-1, 7]])
        p1 = np.array([ttraject[num, 5], ttraject[num, 6], ttraject[num, 7]])

        test1 = np.dot(nor, p0) > np.dot(nor, poi)
        test2 = np.dot(nor, p1) > np.dot(nor, poi)
        test3 = test2 == test1

        if test3 == True:
            # In this case, the trajectories remain unmodified
            tor = 1
        else:
            newp = reflect(p0, p1, nor, poi)
            ttraject[num, 5] = newp[0]
            ttraject[num, 6] = newp[1]
            ttraject[num, 7] = newp[2]

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


def reflect(p0, p1, n, an):
    """
    This function takes a line defined by points p0 and p1 and reflects it about the plane defined by the normal n
    and the point an.  The output is a new point p1new where the particle would be if it "bounced" off the plane.

    Note: the inputs should be defined as numpy arrays.  Ideally, the plane should fall in between p0 and p1.  I will
    need to come up with a test for this in the future.
    """

    d = p1 - p0

    mag = np.linalg.norm(n)
    nn = n/mag
    r = d - 2 * np.dot(d, nn) * nn

    par = (np.dot(n, an) - np.dot(n, p0))/np.dot(n, d)
    inter = p0 + par * d

    rem = np.linalg.norm(p1 - inter)
    p1new = (rem/np.linalg.norm(r)) * r + inter

    return p1new


def multrandplane(b, s, f, p, pnor, ppoi):
    """
    See randplane
    """

    parts = p
    one = randplane(b, s, f, 1, pnor, ppoi)
    counter = 1

    while counter < p + 1:
        counter = counter + 1
        one = np.append(one, randplane(b, s, f, counter, pnor, ppoi), axis=0)

    return one


def rand2planes(b, s, f, p, pnor1, ppoi1, pnor2, ppoi2):
    """
    Builds a single random trajectory with a convection term. Magnitude of the
    convection term is determined by xc, yc, and zc. Default is 0.

    b: base magnitude of single step
    s: variation in step size
    f: number of frames or steps to Takes
    p: particle number (should be 1 for now)
    pnor: the normal defining the plane of interest as a 3-element numpy array
    ppoi: a point that falls on the plane of interest as a 3-element numpy array

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

    # Define a plane
    nor = pnor1
    poi = ppoi1

    nor1 = pnor2
    poi1 = ppoi2

    ttraject = np.zeros((frames, 22))
    ttraject[0, 0] = p

    for num in range(1, frames):

        # Create particle number
        ttraject[num, 0] = p

        # Create frame
        ttraject[num, 1] = 1 + ttraject[num-1, 1]

        # Create magnitude vectors
        ttraject[num, 2] = base*(random.random()-0.5)
        ttraject[num, 19] = base*(random.random()-0.5)
        ttraject[num, 20] = base*(random.random()-0.5)

        # Build preliminary trajectories
        ttraject[num, 5] = ttraject[num-1, 5] + ttraject[num, 2]
        ttraject[num, 6] = ttraject[num-1, 6] + ttraject[num, 19]
        ttraject[num, 7] = ttraject[num-1, 7] + ttraject[num, 20]

        # Test to see if trajectories cross boundaries
        p0 = np.array([ttraject[num-1, 5], ttraject[num-1, 6], ttraject[num-1, 7]])
        p1 = np.array([ttraject[num, 5], ttraject[num, 6], ttraject[num, 7]])

        test1 = np.dot(nor, p0) > np.dot(nor, poi)
        test2 = np.dot(nor, p1) > np.dot(nor, poi)
        test3 = test2 == test1

        test4 = np.dot(nor1, p0) > np.dot(nor1, poi1)
        test5 = np.dot(nor1, p1) > np.dot(nor1, poi1)
        test6 = test4 == test5

        if test3 == False and test6 == False:
            (g, h) = inter2planes(nor, poi, nor1, poi1)
            point = (intersection(p0, p1, nor, poi) + intersection(p0, p1, nor1, poi1))/2
            onpoint = linepoint(g, h, point)
            ttraject[num, 5] = 0.95*onpoint[0]
            ttraject[num, 6] = 0.95*onpoint[1]
            ttraject[num, 7] = 0.95*onpoint[2]

        elif test3 == False:
            # In this case, the trajectories remain unmodified
            newp = reflect(p0, p1, nor, poi)
            ttraject[num, 5] = newp[0]
            ttraject[num, 6] = newp[1]
            ttraject[num, 7] = newp[2]

        elif test6 == False:
            newp1 = reflect(p0, p1, nor1, poi1)
            ttraject[num, 5] = newp1[0]
            ttraject[num, 6] = newp1[1]
            ttraject[num, 7] = newp1[2]

        else:
            tor = 2

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


def multrand2planes(b, s, f, p, pnor, ppoi, pnor2, ppoi2):
    """
    See rand2planes
    """

    parts = p
    one = rand2planes(b, s, f, 1, pnor, ppoi, pnor2, ppoi2)
    counter = 1

    while counter < p + 1:
        counter = counter + 1
        one = np.append(one, rand2planes(b, s, f, counter, pnor, ppoi, pnor2, ppoi2), axis=0)

    return one


def inter2planes(n0, an0, n1, an1):
    """
    Outputs a line in parametric format that results from the intersection of two planes.  Note that the planes
    cannot be parallel.

    The output is two vectors such that line = vector1 + vector2 * t
    """

    nnew = np.cross(n0, n1)
    a = np.array([[n0[0], n0[1]], [n1[0], n1[1]]])
    b = np.array([np.dot(n0, an0), np.dot(n1, an1)])
    x = np.linalg.solve(a, b)
    xn = np.array([x[0], x[1], 0])

    return (xn, nnew)


def linepoint(anchor, vector, p):
    """
    This function takes a parametric line defined by anchor and vector (line = anchor + vector * t) and finds the point
    on the line that is the closest to the point p.
    """

    para = np.dot((anchor - p), vector)/(np.linalg.norm(vector)**2)
    insn = anchor + para * vector
    return insn


def intersection(p0, p1, n, an):
    """
    This has the same functionality as reflect, but returns the intersection point rather than the reflected point.
    """

    d = p1 - p0

    mag = np.linalg.norm(n)
    nn = n/mag
    r = d - 2 * np.dot(d, nn) * nn

    par = (np.dot(n, an) - np.dot(n, p0))/np.dot(n, d)
    inter = p0 + par * d

    rem = np.linalg.norm(p1 - inter)
    p1new = (rem/np.linalg.norm(r)) * r + inter

    return inter


def quadratic(a, b, c):
    # Solves the quadtratic equation.

    d = np.sqrt(b**2 - 4*a*c)

    x1 = (-b + d)/(2*a)
    x2 = (-b - d)/(2*a)

    return (x1, x2)


def tangenttosphere(point, center, radius):
    """
    Outputs a point p and a vector v that define a plane tangent to the sphere at point p.
    """

    x = point[0]
    y = point[1]
    z = point[2]
    v1 = np.array([2*x, 2*y, 2*z])

    mag = np.linalg.norm(v1)
    v = v1/mag

    return point, v


def interlinesphere(point1, point2, center, radius):
    """
    This equation finds the intersection points of a line with a sphere and will output the intersection point closest
    to point2.
    """

    x0 = point1[0]
    y0 = point1[1]
    z0 = point1[2]
    x1 = point2[0]
    y1 = point2[1]
    z1 = point2[2]
    xc = center[0]
    yc = center[1]
    zc = center[2]
    R = radius

    A = (x0 - xc)**2 + (y0 - yc)**2 + (z0 - zc)**2 - R**2
    C = (x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2
    B = (x1 - xc)**2 + (y1 - yc)**2 + (z1 - zc)**2 - A - C - R**2

    t = quadratic(C, B, A)

    xi1 = x0*(1-t[0]) + t[0]*x1
    yi1 = y0*(1-t[0]) + t[0]*y1
    zi1 = z0*(1-t[0]) + t[0]*z1
    xi2 = x0*(1-t[1]) + t[1]*x1
    yi2 = y0*(1-t[1]) + t[1]*y1
    zi2 = z0*(1-t[1]) + t[1]*z1

    short1 = (x1 - xi1)**2 + (y1 - yi1)**2 + (z1 - zi1)**2
    short2 = (x1 - xi2)**2 + (y1 - yi2)**2 + (z1 - zi2)**2

    if short1 < short2:
        (xi, yi, zi) = (xi1, yi1, zi1)
    else:
        (xi, yi, zi) = (xi2, yi2, zi2)


    return (xi, yi, zi)


def reflectmod(p0, p1, n, an):
    """
    Modified to make the magnitude of the reflected vector smaller.

    This function takes a line defined by points p0 and p1 and reflects it about the plane defined by the normal n
    and the point an.  The output is a new point p1new where the particle would be if it "bounced" off the plane.

    Note: the inputs should be defined as numpy arrays.  Ideally, the plane should fall in between p0 and p1.  I will
    need to come up with a test for this in the future.
    """

    d = p1 - p0

    mag = np.linalg.norm(n)
    nn = n/mag
    r = d - 2 * np.dot(d, nn) * nn

    par = (np.dot(n, an) - np.dot(n, p0))/np.dot(n, d)
    inter = p0 + par * d

    rem = 0.15*np.linalg.norm(p1 - inter)
    p1new = (rem/np.linalg.norm(r)) * r + inter

    return p1new


def randsphere(b, s, f, p, center, radius):
    """
    Builds a single random trajectory bounded by sphere centered at point 0 and of radius "radius."

    b: base magnitude of single step
    s: variation in step size
    f: number of frames or steps to Takes
    p: particle number (should be 1 for now)
    center: point defining center of the sphere (x, y, z)
    ppoi: a point that falls on the plane of interest as a 3-element numpy array

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
    ttraject[0, 0] = p

    for num in range(1, frames):

        # Create particle number
        ttraject[num, 0] = p

        # Create frame
        ttraject[num, 1] = 1 + ttraject[num-1, 1]

        # Create magnitude vectors
        ttraject[num, 2] = base*(random.random()-0.5)
        ttraject[num, 19] = base*(random.random()-0.5)
        ttraject[num, 20] = base*(random.random()-0.5)

        # Build preliminary trajectories
        ttraject[num, 5] = ttraject[num-1, 5] + ttraject[num, 2]
        ttraject[num, 6] = ttraject[num-1, 6] + ttraject[num, 19]
        ttraject[num, 7] = ttraject[num-1, 7] + ttraject[num, 20]

        # Test to see if trajectories cross boundary
        p0 = np.array([ttraject[num-1, 5], ttraject[num-1, 6], ttraject[num-1, 7]])
        p1 = np.array([ttraject[num, 5], ttraject[num, 6], ttraject[num, 7]])
        pc = center

        test1 = (p0[0] - pc[0])**2 + (p0[1] - pc[1])**2 + (p0[2] - pc[2])**2 < radius**2
        test2 = (p1[0] - pc[0])**2 + (p1[1] - pc[1])**2 + (p1[2] - pc[2])**2 < radius**2
        test3 = test2 == test1

        if test3 == True:
            # In this case, the trajectories remain unmodified
            tor = 1
        else:
            pint = interlinesphere(p0, p1, pc, radius)
            pint, v = tangenttosphere(pint, pc, radius)
            newp = reflectmod(p0, p1, v, pint)
            ttraject[num, 5] = 0.97*(pint[0] - pc[0]) + pc[0]
            ttraject[num, 6] = 0.97*(pint[1] - pc[1]) + pc[1]
            ttraject[num, 7] = 0.97*(pint[2] - pc[2]) + pc[2]

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


def multrandsphere(b, s, f, p, center, radius):
    """
    See randsphere
    """

    parts = p
    one = randsphere(b, s, f, 1, center, radius)
    counter = 1

    while counter < p + 1:
        counter = counter + 1
        one = np.append(one, randsphere(b, s, f, counter, center, radius), axis=0)

    return one
