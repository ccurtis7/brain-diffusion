"""
This py file creates an animated plot of 2 trajectory datasets each in the
format of a 3-column numpy array (frames, x-coordinate, y-coordinate).  In order
to run the code, users must input their own data into the file.  I have included
some code that calls my data from a csv file locally and converts it to the
desired numpy format, but this must be replaced with users' own data in order to
work. These portions have been annotated below.

In order to run the file once the correct data has been included, open up a
command line and change to the correct directory.  Run "bokeh serve --show
animated_overlay.py" and the trajectory should be plotted with points advancing
every 67 ms.  The rate of advancement can also be adjusted at the user's
discretion as annotated below.
"""

import numpy as np
from bokeh.plotting import figure, show, gridplot, vplot, hplot, curdoc
from bokeh.io import output_notebook
from bokeh.client import push_session


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

# Uploaded data from my local drive. Converts csv file to numpy array
trajectory = np.genfromtxt('../sample_data/Sample_Trajectory_Data1.csv',
                           delimiter=",")

# Deletes the first row (the titles of the columns)
trajectory = np.delete(trajectory, 0, 0)

# Converts My large dataset into individual 3-column datasets and a time column
time = trajectory[:, 0]

PLGA_4A_UP_R1_P25 = trajectory[:, 1:4]
PLGA_4A_P80_R1_P22 = trajectory[:, 4:7]
PLGA_15k_UP_R3_P5 = trajectory[:, 7:10]
PLGA_15k_P80_R3_P36 = trajectory[:, 10:13]
PLGA_4A_F68_R3_P15 = trajectory[:, 13:16]
PLGA_15k_F68_R3_P10 = trajectory[:, 16:19]
PLGA4A_5CHA_R3_P124 = trajectory[:, 19:22]
PLGA_15k_5CHA_R3_P50 = trajectory[:, 22:25]
PLGA_15k_2CHA_R2_P44 = trajectory[:, 25:28]
PLGA15k_05CHA_R3_P22 = trajectory[:, 28:31]
PEG_PLGA15k_P80_R3_P127 = trajectory[:, 31:34]
PEG_PLGA4A_F68_R3_P111 = trajectory[:, 34:37]
PEG_PLGA4A_P80_R1_P84 = trajectory[:, 37:40]
PEG_PLGA4A_UP_R1_P53 = trajectory[:, 40:43]
PEG_PLGA15k_2CHA_R2_P26 = trajectory[:, 43:46]
PEG_PLGA15k_5CHA_R2_P52 = trajectory[:, 46:49]
PEG_PLGA15k_UP_R3_P61 = trajectory[:, 49:52]
PEG_PLGA58k_5CHA_R2_P15 = trajectory[:, 52:55]
PEG_PLGA15k_F68_R2_P81 = trajectory[:, 55:58]
PLGA_15k_2CHA_R2_P37 = trajectory[:, 58:61]
PLGA_15k_PEG_2CHA_R2_P81 = trajectory[:, 61:64]
PLGA_15k_05CHA_R1_P45 = trajectory[:, 64:67]
PLGA_15k_PEG_05CHA_R3_P61 = trajectory[:, 67:70]

shift_trajectory(PLGA_4A_UP_R1_P25)
shift_trajectory(PEG_PLGA4A_UP_R1_P53)

shift_trajectory(PLGA_4A_P80_R1_P22)
shift_trajectory(PEG_PLGA4A_P80_R1_P84)

shift_trajectory(PLGA_15k_UP_R3_P5)
shift_trajectory(PEG_PLGA15k_UP_R3_P61)

shift_trajectory(PLGA_15k_P80_R3_P36)
shift_trajectory(PEG_PLGA15k_P80_R3_P127)

shift_trajectory(PLGA_4A_F68_R3_P15)
shift_trajectory(PEG_PLGA4A_F68_R3_P111)

shift_trajectory(PLGA_15k_F68_R3_P10)
shift_trajectory(PEG_PLGA15k_F68_R2_P81)

shift_trajectory(PLGA4A_5CHA_R3_P124)
shift_trajectory(PEG_PLGA58k_5CHA_R2_P15)

shift_trajectory(PLGA_15k_5CHA_R3_P50)
shift_trajectory(PEG_PLGA15k_5CHA_R2_P52)
shift_trajectory(PLGA_15k_2CHA_R2_P37)
shift_trajectory(PLGA_15k_PEG_2CHA_R2_P81)

shift_trajectory(PEG_PLGA15k_2CHA_R2_P26)
shift_trajectory(PLGA_15k_2CHA_R2_P44)

shift_trajectory(PLGA15k_05CHA_R3_P22)
shift_trajectory(PLGA_15k_05CHA_R1_P45)
shift_trajectory(PLGA_15k_PEG_05CHA_R3_P61)

# This is where the actual coding begins.
a = PLGA_15k_F68_R3_P10
b = PEG_PLGA58k_5CHA_R2_P15
xlist1 = a[:, 1]
ylist1 = a[:, 2]
xlist2 = b[:, 1]
ylist2 = b[:, 2]

xlow = min(min(xlist1), min(xlist2))
xhigh = max(max(xlist1), max(xlist2))
ylow = min(min(ylist1), min(ylist2))
yhigh = max(max(ylist1), max(ylist2))

# create a plot and style its properties.  Change chart title here.
p = figure(title='PLGA_15k_UP_R3_P5 and PEG_PLGA15k_UP_R3_P61',
           title_text_font_size='13pt', x_range=(xlow, xhigh),
           y_range=(ylow, yhigh),)

# add a text renderer to out plot (no data yet)
r1 = p.line(x=[], y=[], line_width=3, color='navy')
r2 = p.line(x=[], y=[], line_width=3, color='firebrick')

# session = push_session(curdoc())

i = 0
ds1 = r1.data_source
ds2 = r2.data_source


# create a callback that will add a number in a random location
def callback():
    global i
    ds1.data['x'].append(xlist1[i])
    ds1.data['y'].append(ylist1[i])
    ds1.trigger('data', ds1.data, ds1.data)
    ds2.data['x'].append(xlist2[i])
    ds2.data['y'].append(ylist2[i])
    ds2.trigger('data', ds2.data, ds2.data)
    if i < xlist1.shape[0] - 1:
        i = i + 1
    else:
        i = 0
        ds1.data['x'] = []
        ds1.data['y'] = []
        ds2.data['x'] = []
        ds2.data['y'] = []

# Adds a new data point every 67 ms.  Change at user's discretion.
curdoc().add_periodic_callback(callback, 67)

# session.show()

# session.loop_until_closed()
