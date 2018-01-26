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


def download_trajectory_data(file):
    """
    Converts a .csv file dataset to a numpy array.  If the file
    is not a .csv file or the file does not exist, returns an error message.

    Inputs: .csv file.
    Outputs: numpy array
    """

    # Define unit test variables
    file_csv = True
    file_exists = True

    # Check if file is a .csv file
    if file[-4:] == '.csv':

        # Check if file exists and downloads it.\
        try:
            name = np.genfromtxt(file, delimiter=",")
            name = np.delete(name, 0, 0)
        except:
            print('File does not exist')
            file_exists = False
            return (file_csv, file_exists)
        else:
            return name

    else:

        file_csv = False
        print('File is not a .csv file')
        return (file_csv, file_exists)


def csv_writer():
    """
    Creates a sample dataset for unit tests called output.csv
    """
    if os.path.exists('output.csv'):
        print('output.csv', 'already exists')

    else:

        data = ["first_name,last_name,city".split(","),
                "Tyrese,Hirthe,Strackeport".split(","),
                "Jules,Dicki,Lake Nickolasville".split(","),
                "Dedric,Medhurst,Stiedemannberg".split(",")
                ]
        path = "output.csv"
        with open(path, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for line in data:
                writer.writerow(line)


def define_xydata(dataset, sets):
    """
    Takes a large numpy array (presumably trajectory data with frames, x
    coordinates, y coordinates) of n columns and splits into (n-1)/3 numpy
    arrays of x-y data plus a separate array of time data.

    The first column in array is time column.  Afterwards, every set of three
    columns becomes an entry in dictionary (Run i).

    Example: trajectory data over 10-second interval with three particles, each
    with three types of data: frame, x coordinate, y coordinate.  define_xydata
    would output a times array (1 column) plus 3 3-column arrays of trajectory
    data with names Run['Run1'], Run['Run2'], Run['Run3'].  These can be used
    to define variable with desired names later.

    Input: Large numpy array (n columns), number of sets to define from array
    Output: time array (1 column)
            trajectory arrays ((n-1)/3 columns)
    """

    splitsuccess = True
    wholenumber = True
    justrightsets = True
    correctformat = True

    if (dataset.shape[1] - 1) % 3 == 0:

        if sets < (dataset.shape[1] - 1)/3:

            if float(sets).is_integer():

                times = dataset[:, 0]
                Run = dict()

                for num in range(1, sets + 1):
                    Run["Run" + str(num)] = dataset[:, 3 * num - 2: 3 * num + 1]

                return times, Run

            else:
                print("The variable 'sets' must be a whole number.")
                wholenumber = False
                splitsuccess = False
                return (splitsuccess, wholenumber, justrightsets, correctformat)

        else:
            print("Desired number of sets is too high for this dataset.")
            justrightsets = False
            splitsuccess = False
            return (splitsuccess, wholenumber, justrightsets, correctformat)

    else:
        print("Data is not of right format.  Dataset must have 1 time column and sets of 3 columns of trajectory data.")
        correctformat = False
        splitsuccess = False
        return (splitsuccess, wholenumber, justrightsets, correctformat)


def shift_trajectory(xydata):
    """
    Adjusts the coordinate system of x-y trajectory data such that if the data
    were plotted, the center of the plot would be (0,0).

    Inputs:
    xydata: a numpy array of 3 columns: frame, x-coordinate and y-coordinate.
    """

    length = xydata.shape[0]
    width = xydata.shape[1]

    # Define unit test variables
    numerical = True
    justright = True

    # checks for data that's not float64 format
    for num in range(0, width):

        for yes in range(0, length):

            if np.dtype(xydata[yes, num]) == np.dtype('float64'):
                numerical = True
            else:
                numerical = False
                break

    if not numerical:

        print("Array contains data that isn't of type float64.")

    else:

        # Checks if array is correct format (3 columns)
        if width == 3:

            x_mean = (np.min(xydata[:, 1]) + np.max(xydata[:, 1]))/2
            y_mean = (np.min(xydata[:, 2]) + np.max(xydata[:, 2]))/2

            xydata[:, 1] = xydata[:, 1] - x_mean
            xydata[:, 2] = xydata[:, 2] - y_mean

            print("Trajectory successfully shifted.")

        else:

            if width > 3:

                x_mean = (np.min(xydata[:, 1]) + np.max(xydata[:, 1]))/2
                y_mean = (np.min(xydata[:, 2]) + np.max(xydata[:, 2]))/2

                xydata[:, 1] = xydata[:, 1] - x_mean
                xydata[:, 2] = xydata[:, 2] - y_mean

                justright = False

                print("Array has more than three columns. May not yield correct results")

            else:

                justright = False
                print("Array doesn't have enough columns")

    return (numerical, justright)


def shift_trajectory3D(xyzdata):
    """
    Adjusts the coordinate system of x-y-z trajectory data such that if the data
    were plotted, the center of the plot would be (0,0,0).

    Inputs:
    xyzdata: a numpy array of 3 columns: frame, x-coordinate and y-coordinate.
    """

    length = xyzdata.shape[0]
    width = xyzdata.shape[1]

    # Define unit test variables
    numerical = True
    justright = True

    # checks for data that's not float64 format
    for num in range(0, width):

        for yes in range(0, length):

            if np.dtype(xyzdata[yes, num]) == np.dtype('float64'):
                numerical = True
            else:
                numerical = False
                break

    if not numerical:

        print("Array contains data that isn't of type float64.")

    else:

        # Checks if array is correct format (3 columns)
        if width == 3:

            x_mean = (np.min(xyzdata[:, 0]) + np.max(xyzdata[:, 0]))/2
            y_mean = (np.min(xyzdata[:, 1]) + np.max(xyzdata[:, 1]))/2
            z_mean = (np.min(xyzdata[:, 2]) + np.max(xyzdata[:, 2]))/2

            xyzdata[:, 0] = xyzdata[:, 0] - x_mean
            xyzdata[:, 1] = xyzdata[:, 1] - y_mean
            xyzdata[:, 2] = xyzdata[:, 2] - z_mean

            print("Trajectory successfully shifted.")

        else:

            if width > 3:

                x_mean = (np.min(xyzdata[:, 0]) + np.max(xyzdata[:, 0]))/2
                y_mean = (np.min(xyzdata[:, 1]) + np.max(xyzdata[:, 1]))/2
                z_mean = (np.min(xyzdata[:, 2]) + np.max(xyzdata[:, 2]))/2

                xydata[:, 1] = xyzdata[:, 1] - x_mean
                xydata[:, 2] = xyzdata[:, 2] - y_mean

                justright = False

                print("Array has more than three columns. May not yield correct results")

            else:

                justright = False
                print("Array doesn't have enough columns")

    return (numerical, justright)


def plot_trajectory(xydata, charttitle):
    """
    Plots a single 3-column numpy array of trajectory data (frames in column 1,
    x coordinates in column 2, y coordinates in column 3) within iPython
    notebook.

    Note: MUST have run output_notebook in order to run successfully.

    Input: numpy array, chart title (string)
    Output: displays trajectory plot inline
    """

    length = xydata.shape[0]
    width = xydata.shape[1]

    justright = True

    if width == 3:

        x = xydata[:, 1]
        y = xydata[:, 2]
        p = figure(title=charttitle, title_text_font_size='13pt',
                   width=300, height=300)
        p.line(x, y, line_width=2)
        show(p)

    else:

        justright = False

        if width > 3:

            x = xydata[:, 1]
            y = xydata[:, 2]
            p = figure(title=charttitle, title_text_font_size='13pt',
                       width=300, height=300)
            p.line(x, y, line_width=2)
            show(p)

            print("Array has more than three columns.  May not yield correct results")

        else:

            justright = False
            print("Array doesn't have enough columns")

    return justright


def sidebyside(xydata1, xydata2, charttitle1, charttitle2):
    """
    Plots two 3-column numpy arrays of trajectory data (frames in column 1, x
    coordinates in column 2, y coordinates in column 3) next to each other in
    iPython notebook.

    Note: MUST have run output_notebook in order to run successfully.

    Input: 2 numpy arrays, 2 chart titles (strings)
    Output: displays trajectory plots inline.
    """
    length1 = xydata1.shape[0]
    width1 = xydata1.shape[1]

    justright = True

    length2 = xydata2.shape[0]
    width2 = xydata2.shape[1]

    if width1 == 3 and width2 == 3:

        x1 = xydata1[:, 1]
        y1 = xydata1[:, 2]
        x2 = xydata2[:, 1]
        y2 = xydata2[:, 2]

        s1 = figure(title=charttitle1, title_text_font_size='13pt', width=300, height=300)
        s1.line(x1, y1, color='navy', line_width=2)

        s2 = figure(title=charttitle2, title_text_font_size='13pt', width=300, height=300, x_range=s1.x_range, y_range=s1.y_range)
        s2.line(x2, y2, color='firebrick', line_width=2)

        p = gridplot([[s1, s2]])
        show(p)

    else:

        justright = False

        if width1 > 3 or width2 > 3:

            x1 = xydata1[:, 1]
            y1 = xydata1[:, 2]
            x2 = xydata2[:, 1]
            y2 = xydata2[:, 2]

            s1 = figure(title=charttitle1, width=300, height=300, x_axis_label='x', y_axis_label='y')
            s1.line(x1, y1, color='navy', line_width=2)

            s2 = figure(title=charttitle2, width=300, height=300, x_range=s1.x_range, y_range=s1.y_range, x_axis_label='x', y_axis_label='y')
            s2.line(x2, y2, color='firebrick', line_width=2)

            p = hplot(s1, s2)
            show(p)

            print("At least one of the given arrays has more than three columns.  May not yield corect results.")

        else:

            print("One of the given arrays has less than three columns.  Data could not be plotted.")

    return justright


def overlay(xydata1, xydata2, charttitle):
    """
    Plots two 3-column numpy arrays of trajectory data (frames in column 1, x
    coordinates in column 2, y coordinates in column 3) superimposed upon each
    other in iPython notebook.

    Note: MUST have run output_notebook in order to run successfully.

    Input: 2 numpy arrays, 2 chart titles (strings)
    Output: displays trajectory plots inline.
    """

    length1 = xydata1.shape[0]
    width1 = xydata1.shape[1]

    justright = True

    length2 = xydata2.shape[0]
    width2 = xydata2.shape[1]

    if width1 == 3 and width2 == 3:

        x1 = xydata1[:, 1]
        y1 = xydata1[:, 2]
        x2 = xydata2[:, 1]
        y2 = xydata2[:, 2]

        p = figure(title=charttitle, title_text_font_size='9pt', width=300, height=300, x_axis_label='x', y_axis_label='y')

        p.line(x1, y1, line_width=2, color='navy')
        p.line(x2, y2, line_width=2, color='firebrick')

        show(p)

    else:

        justright = False

        if width1 > 3 or width2 > 3:

            x1 = xydata1[:, 1]
            y1 = xydata1[:, 2]
            x2 = xydata2[:, 1]
            y2 = xydata2[:, 2]

            p = figure(title=charttitle, width=300, height=300)

            p.line(x1, y1, line_width=2, color='navy')
            p.line(x2, y2, line_width=2, color='firebrick')

            show(p)

            print("At least one of the given arrays has more than three columns.  May not yield corect results.")

        else:

            print("One of the given arrays has less than three columns.  Data could not be plotted.")

    return justright


def animated_plot(xydata):
    """
    I haven't been able to generate a working code for an animated plot function
    and I haven't been able to find out why.  Whenever I try, I get an error
    message saying that index cannot be defined.
    """
    b = xydata
    xlist = b[:, 1]
    ylist = b[:, 2]

    # create a plot and style its properties
    p = figure(x_range=(min(xlist), max(xlist)), y_range=(min(ylist), max(ylist)), toolbar_location=None)

    # add a text renderer to out plot (no data yet)
    r = p.line(x=[], y=[], line_width=3, color='navy')

    session = push_session(curdoc())

    index = 0
    ds = r.data_source

    # create a callback that will add the next point of the trajectory data
    def callback():
        global index
        ds.data['x'].append(xlist[index])
        ds.data['y'].append(ylist[index])
        ds.trigger('data', ds.data, ds.data)
        index = index + 1

    curdoc().add_periodic_callback(callback, 67)

    # open the document in a browser
    session.show()

    # run forever
    session.loop_until_closed()


def plot_trajectories3D(traj, n1, n2, n3, dec, filename):
    """
    This function creates a multiple 3D plots from trajectory data.  This
    dataset must include a column of particle numbers as well as the x, y, and
    z coordinates of of each particle at each frame. Output will be saved as a
    .png file of the desired name.
    Inputs:
    traj: array of trajectory data e.g. particle #, frames, x, y, z, Deff, MSD
    n1: particle# column
    n2: x data
    n3: z data (with y data, of course, in between x and z.  This just defines a range)
    dec: how many decimals you would like to be displayed in the graph.
    filename: what you want to name the file.  Must be in ''.
    Can also use plt.show() afterwards to preview your data, even if it skews the title and legend a bit.
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    position = traj[:, n2:n3+1]
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

    # Determines arrangement of subplots
    rows = np.sqrt(total)
    if (rows % 1 == 0):
        rows = int(rows)
    else:
        rows = int(rows) + 1

    # Create figure
    fig = plt.figure(figsize=(24, 20), dpi=80)
    ax = dict()

    # Plot trajectories
    for num in range(1, total1):

        number = 100*rows + 10*rows + num
        ax[num] = fig.add_subplot(number, projection='3d')
        ax[num].set_title('Particle {}'.format(num), x=0.5, y=1.08)
        ax[num].plot(path[num][:, 0], path[num][:, 1], path[num][:, 2])
        ax[num].locator_params(nbins=5)
        ax[num].title.set_fontsize(30)

        ax[num].tick_params(direction='out', pad=13)

        ax[num].xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f um'.format(dec)))
        ax[num].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f um'.format(dec)))
        ax[num].zaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f um'.format(dec)))

    # Save figures
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')


def plot_3Doverlay(traj, n1, n2, dec, filename, xr, yr, zr):
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
    xr: defines the range of x
    yr: defines the range of y
    zr: defines the range of z
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

    # Creates figure
    fig = plt.figure(figsize=(24, 24), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_title('Particle Trajectories', x=0.5, y=1.15)

    # Plots individual trajectories
    for num in range(1, total1):

        ax.plot(path[num][:, 0], path[num][:, 1], path[num][:, 2], label='Particle {}'.format(num), linewidth=3)

    axbox = ax.get_position()
    # ax.legend(loc=(0.86, 0.90), prop={'size': 20})
    ax.locator_params(nbins=6)
    ax.view_init(elev=38, azim=72)

    # A few adjustments to prettify the graph
    for item in ([ax.xaxis.label, ax.yaxis.label, ax.zaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
        item.set_fontsize(32)

    plt.xticks(rotation=-30)

    ax.title.set_fontsize(35)
    ax.tick_params(direction='out', pad=20)
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f um'.format(dec)))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f um'.format(dec)))
    plt.gca().zaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f um'.format(dec)))
    plt.gca().set_xlim([-xr, xr])
    plt.gca().set_ylim([-yr, yr])
    plt.gca().set_zlim([-zr, zr])

    # Save your figure
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')


def plot_MSDorDeff(traj, n1, n2, n3, dec, datatype, filename):
    """
    Plots the MSDs from a trajectory dataset.

    n1: particle numbers
    n2: time
    n3: MSDs or Deffs (34 or 35 for my datasets)
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    total = int(max(particles))
    total1 = total + 1
    rawtime = traj[:, n2]
    rawMSD = traj[:, n3]
    MSD = dict()
    time = dict()

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        MSD[num] = (rawMSD[min1:max1])
        time[num] = (rawtime[min1:max1])

    # Creates figure
    fig = plt.figure(figsize=(24, 18), dpi=80)
    ax = fig.add_subplot(111)
    # ax.set_title('Particle Trajectories', x=0.5, y=1.15)

    # Plots individual trajectories
    for num in range(1, total1):

        ax.plot(time[num][:], MSD[num][:], label='Particle {}'.format(num))

    axbox = ax.get_position()
    ax.legend(loc=(0.86, 0.90), prop={'size': 20})
    # ax.locator_params(nbins=4)

    # A few adjustments to prettify the graph
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)

    ax.title.set_fontsize(35)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(datatype)
    ax.tick_params(direction='out', pad=16)
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))

    # Save your figure
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')


def plot_MeanMSDorDeff(traj, n1, n2, n3, dec, datatype, filename):
    """
    Plots the MSDs from a trajectory dataset.

    n1: particle numbers
    n2: time
    n3: MSDs or Deffs
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    total = int(max(particles))
    total1 = total + 1
    rawtime = traj[:, n2]
    rawMSD = traj[:, n3]
    MSD = dict()
    time = dict()

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        MSD[num] = (rawMSD[min1:max1])
        time[num] = (rawtime[min1:max1])

    MMSD = MSD[1]
    for num in range(2, total1):
        MMSD = MMSD + MSD[num]
    MMSD = MMSD/total1

    # Creates figure
    fig = plt.figure(figsize=(24, 18), dpi=80)
    ax = fig.add_subplot(111)
    # ax.set_title('Particle Trajectories', x=0.5, y=1.15)

    ax.plot(time[1][:], MMSD[:])

    # A few adjustments to prettify the graph
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)

    ax.title.set_fontsize(35)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(datatype)
    ax.tick_params(direction='out', pad=16)
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))

    # Save your figure
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')


def randtraj(b, s, f, p):
    """
    Builds a single random trajectory.

    b: base magnitude of single step
    s: variation in step size
    f: number of frames or steps to Takes
    p: particle number (should be 1 for now)

    Output:
    0 particle number
    1 time or frames
    2 magnitude
    3 angle 1
    4 angle 2
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
    """

    base = b
    step = s
    pi = 3.14159265359
    frames = f

    ttraject = np.zeros((frames, 20))

    for num in range(1, frames):

        # Create particle number
        ttraject[num, 0] = p
        ttraject[num-1, 0] = p
        # Create frame
        ttraject[num, 1] = 1 + ttraject[num-1, 1]
        # Create magnitude vector
        ttraject[num, 2] = base + step*random.random()
        # Create Angle Vectors
        ttraject[num, 3] = 2 * pi * random.random()
        ttraject[num, 4] = pi * random.random()
        # Build trajectories
        ttraject[num, 5] = ttraject[num-1, 5] + ttraject[num, 2]*np.sin(ttraject[num, 4])*np.cos(ttraject[num, 3])
        ttraject[num, 6] = ttraject[num-1, 6] + ttraject[num, 2]*np.sin(ttraject[num, 4])*np.sin(ttraject[num, 3])
        ttraject[num, 7] = ttraject[num-1, 7] + ttraject[num, 2]*np.cos(ttraject[num, 4])

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


def randtraj2(b, s, f, p):
    """
    Builds a single random trajectory without using spherical coordinates, as
    randtraj does.

    b: base magnitude of single step
    s: variation in step size
    f: number of frames or steps to Takes
    p: particle number (should be 1 for now)

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

    return ttraject


def multrandtraj(b, s, f, p):
    """
    Builds an array of multiple trajectories appended to each other. Number of
    trajectories is determined by p.
    """

    parts = p
    one = randtraj2(b, s, f, 1)
    counter = 1

    while counter < p + 1:
        counter = counter + 1
        one = np.append(one, randtraj2(b, s, f, counter), axis=0)

    return one


def plot_Mean2DMSDsorDeff(traj, n1, n2, n3, dec1, dec2, datatype, filename, limit1, limit2, tick1, tick2):
    """
    Plots the MSDs or Deffs from a trajectory dataset.

    n1: particle numbers (typically 0)
    n2: time (typically 1)
    n3: MSDs or Deffs (11 or 15 typically)
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
        MSD[num] = (raw2DMSDs[min1 + 2:max1, :])
        time[num] = (rawtime[min1 + 2:max1])

    MMSD = MSD[1]
    for num in range(2, total1):
        MMSD = MMSD + MSD[num]
    MMSD = MMSD/total1

    # Creates figure
    fig = plt.figure(figsize=(24, 18), dpi=80)
    ax = fig.add_subplot(111)
    # ax.set_title('Particle Trajectories', x=0.5, y=1.15)

    ax.plot(time[1][:], MMSD[:, 0], linewidth=10, label='3D')
    ax.plot(time[1][:], MMSD[:, 1], linewidth=10, label='2D xy')
    ax.plot(time[1][:], MMSD[:, 2], linewidth=10, label='2D xz')
    ax.plot(time[1][:], MMSD[:, 3], linewidth=10, label='2D yz')

    # A few adjustments to prettify the graph
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(70)

    xmajor_ticks = np.arange(0, limit1, tick1)
    ymajor_ticks = np.arange(0, limit2, tick2)

    ax.set_xticks(xmajor_ticks)
    ax.set_yticks(ymajor_ticks)
    ax.title.set_fontsize(70)
    ax.set_xlabel('Time (s)', fontsize=95)
    ax.set_ylabel(datatype, fontsize=95)
    ax.tick_params(direction='out', pad=16)
    ax.legend(loc=(0.65, 0.05), prop={'size': 70})
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec1)))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec2)))

    plt.gca().set_xlim([0, limit1])
    plt.gca().set_ylim([0, limit2])

    # Save your figure
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')
    return MMSD


def plot_MSDorDeffLR(traj, n1, n2, n3, dec, datatype, filename):
    """
    Plots the MSDs from a trajectory dataset. Also performs linear regression
    to find average MSD.  This is normally used for data from my Excel
    spreadsheets, and not for random trajectories, as random trajectories can
    be plotted using an average function rather than linear regression.  This
    also shouldn't actually be used for diffusion data, as this would be a poor
    linear regression model for that set.  I will make another function
    that will calculate Deff from that.

    It must also be noted that this function will not work with datasets that
    have trajectories beginning at times other than t=0.  This, for now, must
    be manually adjusted in the spreadsheet.  I will ensure that my spreadsheets
    are made this way before using the data.

    n1: particle numbers
    n2: time
    n3: MSDs or Deffs (34 or 35 for my datasets)
    datatype: must be either "MSD" or "Deff" in order to work
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    total = int(max(particles))
    total1 = total + 1
    rawtime = traj[:, n2]
    rawMSD = traj[:, n3]
    MSD = dict()
    time = dict()

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        MSD[num] = (rawMSD[min1:max1])
        time[num] = (rawtime[min1:max1])

    # Insert to perform linear regression
    small = 10**-10

    def func(x, a, b, c, d, g):
        return a*x + b*x**0.5 + c*x**2 + d*x**3 + g*np.log(x+small)

    xdata = traj[:, n2]
    ydata = traj[:, n3]
    x0 = [0.1, 0.1, 0.1, 0.1, 0.1]
    params, other = opt.curve_fit(func, xdata, ydata, x0)

    time1 = np.linspace(min(xdata), max(xdata), num=100)
    MSD1 = np.zeros(np.shape(time1)[0])

    for num in range(0, np.shape(time1)[0]):
        MSD1[num] = params[0]*time1[num] + params[1]*time1[num]**0.5 + params[2]*time1[num]**2 + params[3]*time1[num]**3 + params[4]*np.log(time1[num] + small)

    # Creates figure
    fig = plt.figure(figsize=(24, 18), dpi=80)
    ax = fig.add_subplot(111)
    # ax.set_title('Particle Trajectories', x=0.5, y=1.15)

    # Plots individual trajectories
    for num in range(1, total1):

        ax.plot(time[num][:], MSD[num][:], label='Particle {}'.format(num))

    ax.plot(time1, MSD1, linewidth=2.5, label='Average')
    axbox = ax.get_position()
    #ax.legend(loc=(0.82, 0.82), prop={'size': 30})
    # ax.locator_params(nbins=4)

    # A few adjustments to prettify the graph
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)

    ax.title.set_fontsize(35)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(datatype)
    ax.tick_params(direction='out', pad=16)
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))

    # Save your figure
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')


def LRfor3D2D(traj, n1, n2, n3, n4, n5, n6, dec, datatype, filename):
    """
    n1: particle numbers
    n2: time
    n3: 3DMSDs
    n4: 2D xy MSDs
    n5: 2D xz MSDs
    n6: 2D yz MSDs
    """

    # Insert to perform linear regression
    small = 10**-10

    def func(x, a, b, c, d, g):
        return a*x + b*x**0.5 + c*x**2 + d*x**3 + g*np.log(x+small)

    # Linear regression for 3D MSDs
    xdata = traj[:, n2]
    ydata = np.c_[np.c_[np.c_[traj[:, n3], traj[:, n4]], traj[:, n5]], traj[:, n6]]
    x0 = [0.1, 0.1, 0.1, 0.1, 0.1]
    params = dict()
    MSD1 = dict()

    for num in range(0, 4):
        params[num], other = opt.curve_fit(func, xdata, ydata[:, num], x0)

    time1 = np.linspace(min(xdata), max(xdata), num=100)

    for b in range(0, 4):
        MSD1[b] = np.zeros(np.shape(time1)[0])
        for num in range(0, np.shape(time1)[0]):
            MSD1[b][num] = params[b][0]*time1[num] + params[b][1]*time1[num]**0.5 + params[b][2]*time1[num]**2 + params[b][3]*time1[num]**3 + params[b][4]*np.log(time1[num] + small)

    Deff = np.divide(MSD1[0], 6*time1)
    Dxy = np.divide(MSD1[1], 4*time1)
    Dxz = np.divide(MSD1[2], 4*time1)
    Dyz = np.divide(MSD1[3], 4*time1)

    # Creates figure
    fig = plt.figure(figsize=(24, 18), dpi=80)
    ax = fig.add_subplot(111)
    # ax.set_title('Particle Trajectories', x=0.5, y=1.15)

    if datatype == 'MSD':
        ax.plot(time1, MSD1[0], linewidth=2.5, label='3D MSD')
        ax.plot(time1, MSD1[1], linewidth=2.5, label='2D xy MSD')
        ax.plot(time1, MSD1[2], linewidth=2.5, label='2D xz MSD')
        ax.plot(time1, MSD1[3], linewidth=2.5, label='2D yz MSD')
    else:
        ax.plot(time1, Deff, linewidth=2.5, label='3D D')
        ax.plot(time1, Dxy, linewidth=2.5, label='2D xy D')
        ax.plot(time1, Dxz, linewidth=2.5, label='2D xz D')
        ax.plot(time1, Dyz, linewidth=2.5, label='2D yz D')

    # A few adjustments to prettify the graph
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)

    ax.title.set_fontsize(35)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(datatype)
    ax.tick_params(direction='out', pad=16)
    ax.legend(loc=(0.79, 0.79), prop={'size': 30})
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))

    # Save your figure
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')


def fillin(data):
    """
    This function is perfect.  It shifts the frames by the startframe and fills in any blank frames.
    """
    def startshift(data1):
        startframe = data1[0, 1]
        data1[:, 1] = data1[:, 1] - startframe
        return data1

    data = startshift(data)

    shap = int(max(data[:, 1])) + 1
    filledin = np.zeros((shap, 5))
    filledin[0, :] = data[0, :]

    count = 0
    new = 0
    other = 0
    tot = 0

    for num in range(1, shap):
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


def prettify(traj, cut, lim, umppx, fps, umps):
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
    rawdataset[:, 2:4] = umppx * rawdataset[:, 2:4]
    rawdataset[:, 4] = umps * rawdataset[:, 4]

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        dataset[num] = (rawdataset[min1:max1, 0:5])

    flee = dict()
    for num in range(1, total1):
        flee[num] = fillin(dataset[num])

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
