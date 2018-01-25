"""
A script for calculating and visualizing Deff (effective diffusion
coefficient) based on MSD (mean squared displacement) data from t=0 to
a user-selected timepoint or range of timepoints (in which case Deff is
a geometric mean).  Make sure to import the entire file: this performs
the time-intensive cleaning of the data independent of the
plotting/computing function, making for faster plot tweaks.
"""
import numpy as np
import os
import pandas as pd
import scipy
from scipy import stats

from bokeh.charts import Histogram
from bokeh.models import Range1d
from bokeh.plotting import figure, hplot, output_file, show
import zipfile


ALPHA = 1


"""
These lines are included as script (not within a function) in order to
execute their relatively time-intensive data cleaning when the file is
imported, not when the graphing function is called.  This allows for
faster tweaking of graph parameters.
"""
if os.path.exists('brain-diffusion_data.zip') == False:
    print 'Error: data do not exist or are not in working directory'
zf = zipfile.ZipFile('brain-diffusion_data.zip')
file_handle = zf.open('Mean_Square_Displacement_Data.csv')
msd = pd.read_csv(file_handle)
# Rename columns without biological replicate number (this allows for
# averaging of biological replicates later).  "Biological replicates"
# refers to different brain tissue samples.
msd.columns = msd.iloc[0]
msd = msd.iloc[1:]
# Create a list of column names without replicates
columns = msd.columns
columns = columns[1:67]
columns2 = [ii for n, ii in enumerate(columns) if ii not in columns[:n]]


def compute_geomean(df, column_name):
    """
    Appends a dataframe with the geometric mean of three replicates.

    This function combines MSDs of three biological replicates
    (different brain tissue samples) using a geometric mean.  It first
    splits the three replicates (grouped by common column titles) into
    their own dataframe, converting to float values if necessary, and
    then appends a scipy-calculated geometric mean to the original
    dataframe in a new column.  The new column is titled as the
    original replicate columns plus the word 'geo'.

    Inputs:
    df: a pandas dataframe with replicates for MSD at a range of
    timepoints
    column_name: a string matching the column title of the particle
    chemistry for which a mean MSD is desired to be calculated

    Outputs:
    df: the same pandas dataframe as input, but with an appended
    column containing the geometric mean of the replicate MSD values
    for each timepoint within column_name
    """
    geomeans = [0]
    # Grab the three MSD values, one timepoint at a time
    for i in range(2, len(df) + 1):
        timepoint = df[column_name].ix[i]
        bioreps = list(timepoint[range(len(timepoint))])
        if type(bioreps[0]) != float:
            # Convert strings to floats if necessary
            biorepsfloat = []
            for j in bioreps:
                floatMSD = float(j)
                biorepsfloat.append(floatMSD)
        else:
            biorepsfloat = bioreps
        # Append to a growing list of geometric means, then add as
        # a new column in the original dataframe
        geomeans.append(scipy.stats.gmean(biorepsfloat))
    df[column_name + ' geo'] = geomeans


"""
These lines are included as script (not within a function) in order to
execute their relatively time-intensive data cleaning when the file is
imported, not when the graphing function is called.  This allows for
faster tweaking of graph parameters.
"""
# Create a column of mean MSDs for each particle type
for title in columns2:
    compute_geomean(msd, title)
# Reset the index to timepoints, converting from string to float
msd = msd.set_index('Particle')
tempindex = [0.0]
for i in range(1, len(msd)):
    tempindex.append(float(msd.index[i]))
msd['index'] = tempindex
msd = msd.set_index('index')


def compute_hist_Deff(particle_chemistry, tmin, tmax):
    """
    Calculates and plots Deff in timepoint range for a given chemistry.

    This function trims the cleaned MSD pandas dataframe to the user-
    selected timepoint range, calculates a Deff from the mean MSD for
    each timepoint within the range, plots a histogram of the list of
    Deffs, and gives a geometric mean Deff value for the time range
    specified.  This is all within the specified particle chemistry.
    Note: tmin must be less than tmax.

    Inputs:
    particle_chemistry: a string matching the column title of the
    particle chemistry for which a Deff is to be plotted and calculated
    tmin: a float representing the minimum timepoint the user wishes to
    consider in the Deff calculation
    tmax: a float representing the maximum timepoint the user wishes to
    consider in the Deff calculation

    Outputs:
    A bokeh charts histogram of Deff values calculated from the
    particle chemistry's MSD of each timepoint within the range
    Deff: a single float representing the geometric mean Deff value for
    the timepoint range specified

    Side effects:
    The trimmed MSD dataset is appended (with a new column) to include
    the list of Deffs for the given particle chemistry.
    """
    # Verify time range validity
    if tmin < 0 or tmax > round(max(msd.index)):
        return "Error: input time range between 0 and your data's tmax"
    else:
        if tmin == 0:
            print 'Divide by 0 error: tmin=0 changed to tmin=0.01'
            tmin = 0.01
        # Trim out-of-time-range rows
        temp1_msd = msd[msd.index >= tmin]
        temp2_msd = temp1_msd[temp1_msd.index <= tmax]
        # Calculate Deffs for only the timepoints needed and add as a
        # new column
        Deff_list = []
        for i in range(0, len(temp2_msd)):
            index = temp2_msd.index[i]
            # Calculate Deff using the conventional relationship
            # between MSD and Deff
            Deff_list.append(temp2_msd[particle_chemistry + ' geo'][index]/(
                4*index**ALPHA))
        temp2_msd[particle_chemistry + ' Deff'] = Deff_list
        # Plot histogram and print mean Deff value
        output_file('Deffs_hist.html')
        p = Histogram(
            temp2_msd[particle_chemistry + ' Deff'], bins=15, legend=False)
        # Set range of x axis to reflect approximate range of all Deff
        # values
        p.x_range = Range1d(0, 0.015)
        show(p)
        # There's only one Deff column from this function: calculate
        # geometric mean Deff from that column
        Deff = scipy.stats.gmean(temp2_msd[particle_chemistry + ' Deff'])
        return Deff


def compute_plot_all_Deff(tmin, tmax, particle_chemistry):
    """
    Calculates and plots all Deffs in the timepoint range.

    This function trims the cleaned MSD pandas dataframe to the user-
    selected timepoint range, calculates a Deff from the mean MSD for
    each timepoint within the range and for each particle chemistry,
    plots a line graph of all Deffs across the timepoint range, and
    gives geometric mean Deff values for each chemistry for the time
    range specified.
    Note: tmin must be less than tmax to get a timepoint range.  If,
    however, the user requires a Deff calculated from a single
    timepoint, he or she can input an equal tmin and tmax, and the
    function will consider only the single closest timepoint (on the
    later side) to the input.

    Inputs:
    tmin: a float representing the minimum timepoint the user wishes to
    consider in the Deff calculation
    tmax: a float representing the maximum timepoint the user wishes to
    consider in the Deff calculation (make this the same as tmin if a
    single-timepoint Deff is desired)
    particle_chemistry: a string matching the column title of the
    particle chemistry which is to be emphasized: the Deffs of this
    chemistry will be plotted on the histogram, bolded/dashed on the
    line plot, and printed in the form of a geometric mean.

    Outputs:
    A bokeh histogram of Deffs for the inputted chemistry, on the same
    figure as the below line plot (this figure defaults to fit nicely
    in a 1280x800 screen, but plots can be resized as needed)
    A bokeh line plot of Deffs for all particle chemistries across the
    timepoint range, with emphasized chemistry bolded/dashed
    Deff of inputted chemistry: a geometric mean of the Deffs between
    tmin and tmax for the inputted chemistry, printed in the form of
    a string within an explanatory sentence
    avg_Deffs: a fresh pandas dataframe indexed with the columns
    (particle chemistries) of the MSD dataframe, containing single
    geometric mean Deff values for each particle chemistry
    """
    # Verify time range validity
    if tmin < 0 or tmax > round(max(msd.index)):
        return "Error: input time range between 0 and your data's tmax"
    else:
        if tmin == 0:
            print 'Divide by 0 error: tmin=0 changed to tmin=0.01'
            tmin = 0.01
        # Trim out-of-time-range rows
        temp1_msd = msd[msd.index >= tmin]
        # Trim to a length of 1 if user desires single-timepoint Deff
        if tmin == tmax:
            temp2_msd = temp1_msd.head(1)
        else:
            temp2_msd = temp1_msd[temp1_msd.index <= tmax]
            # Calculate Deffs for only the timepoints needed and add as
            # a new column to a new dataframe
            index = temp2_msd.index
            Deffs = pd.DataFrame(index=index, columns=columns2)
            avg_Deffs = pd.DataFrame(index=columns2)
            avg_Deffs_temp = []
            h = Histogram([1], bins=50, width=300, height=250)
            # maxes will eventually be used for sizing the histogram
            maxes = []
            # Lay the foundation for the bokeh figure
            output_file('Deffs_hist_and_line_plot.html')
            p = figure(
                tools='resize,pan,box_zoom,wheel_zoom,reset,save', width=950,
                height=650, x_axis_label='MSD timepoint', y_axis_label='Deff')
            # Cycle through each particle chemistry and fill in Deffs
            # and avg_Deffs
            for title in columns2:
                single_Deff_list = []
                for i in range(0, len(temp2_msd)):
                    index = temp2_msd.index[i]
                    single_Deff_list.append(temp2_msd[title + ' geo'][index]/(
                        4*index**ALPHA))
                # Add particle-chemistry-specific Deff list to Deffs
                # dataframe
                Deffs[title] = single_Deff_list
                maxes.append(np.max(single_Deff_list))
                # Add geometric mean Deff to what will become avg_Deffs
                avg_Deffs_temp.append(scipy.stats.gmean(Deffs[title]))
                # Create a special bold/dashed line for the inputted
                # chemistry only, and generate the printed output
                if title == particle_chemistry:
                    p.line(
                        Deffs.index, Deffs[title], line_width=5,
                        line_dash=(15, 10), legend=title, line_color=(
                            np.random.randint(256), np.random.randint(256),
                            np.random.randint(256))
                        )
                    h = Histogram(Deffs[particle_chemistry], width=300,
                        height=250)
                    print particle_chemistry + ' Deff = ' + str(
                        scipy.stats.gmean(Deffs[title]))
                else:
                    p.line(
                        Deffs.index, Deffs[title], line_width=1, legend=title,
                        line_color=(
                            np.random.randint(256), np.random.randint(256),
                            np.random.randint(256))
                        )
            avg_Deffs['Deff'] = avg_Deffs_temp
            p.legend.label_text_font_size = '6pt'
            # The line plot x-axis range is calibrated to include all
            # of the desired data and the legend with no overlap
            p.x_range = Range1d(tmin, tmax + (tmax - tmin)/5)
            # The histogram x-axis is calibrated to remain at a
            # constant scale for each given tmin/tmax combination--this
            # gives a sense of scale for this particular histogram
            # within the possible Deff values for all the chemistries
            h.x_range = Range1d(0, np.max(maxes))
            f = hplot(h, p)
            show(f)
            return avg_Deffs
