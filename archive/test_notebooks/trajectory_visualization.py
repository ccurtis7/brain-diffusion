from bokeh.io import output_notebook
from bokeh.plotting import figure, show, gridplot, hplot, vplot, curdoc
import numpy as np
import os
import csv
from bokeh.client import push_session


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
