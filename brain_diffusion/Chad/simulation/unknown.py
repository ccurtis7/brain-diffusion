def roti(traj, n1, n2, th1, th2, th3)
    """
    This function will rotate any given three-coordinate dataset of trajectories
    (traj).  This is normally a dataset from my Excel tracking program or from
    a random trajectory dataset formatted similarly.  This function works by
    inputting three different rotation angles (th1, th2, th3) which defined a
    rotation matrix.  These are done first around the x axis followed by y and
    z.

    Inputs:
    traj: trajectory dataset
    n1: particle number column (normally 0)
    n2: first column in xyz dataset (others are assumed to follow)
    th1: rotation angle around x axis
    th2: rotation angle around y axis
    th3: rotation angle around z axis

    Outputs:
    A modified form of traj with the columns defined by n2 being replaced with
    new rotated coordinates.
    """


def rotmat(traj, n1, n2, A)
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

def plot_3Dwellip(traj, n1, n2, dec, filename, xr, yr, zr):
    """
    This function creates a single 3D plot from trajectory data.  This dataset
    must include a column of particle numbers as well as the x, y, and z
    coordinates of each particle at each frame. Output will be saved as a .png
    file of the desired name.

    This function also will plot a 3D ellipsoid fit around the dataset using
    the mvee function (minimum volume enclosing ellipse).  I have yet to
    determine how to accurately relate the size of the MVEE to the MSD.  right
    now, the user can input the percentile of maximum trajectory values used
    to construct the ellipse (p1) as well as the percentile to exclude (p2).

    Inputs:
    traj: array of trajectory data e.g. particle #, frames, x, y, z, Deff, MSD
    n1: particle# column
    n2: xyz data start (so x data column, 29 for a normal dataset)
    a range)
    dec: how many decimals you would like to be displayed in the graph.
    filename: what you want to name the file.  Must be in ''.
    xr: defines the range of x
    yr: defines the range of y
    zr: defines the range of z
    Can also use plt.show() afterwards to preview your data, even if it skews the title and legend a bit.
    """
