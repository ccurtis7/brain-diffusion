"""
A script for calculating Deff (effective diffusion coefficient) based
on MSD (mean squared displacement) data from t=0 to a user-selected
timepoint or range of timepoints (in which case Deff is a geometric
mean).  Make sure to import the entire file: this performs the time-
intensive cleaning of the data independent of the plotting/computing
function, making for faster plot tweaks.
"""
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats

from bokeh.plotting import figure, output_file, show
import zipfile


ALPHA = 1


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
    for i in range(2, len(df)+1):
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
    df[column_name+' geo'] = geomeans


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


def compute_hist_Deff(particle_chemistry,tmin,tmax):
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
    A matplotlib histogram of Deff values calculated from the particle
    chemistry's MSD of each timepoint within the range
    Deff: a single float representing the geometric mean Deff value for
    the timepoint range specified

    Side effects:
    The trimmed MSD dataset is appended to include the list of Deffs
    for the given particle chemistry.
    """
    # Verify time range validity
    if tmin < 0 or tmax > 20:
        return 'Error: input time range between 0 and 20 sec'
    else:
        if tmin == 0:
            print 'Divide by 0 error: tmin=0 changed to tmin=0.01'
            tmin = 0.01
        # Trim out-of-time-range rows
        temp1_msd = msd[msd.index >= tmin]
        temp2_msd = temp1_msd[temp1_msd.index <= tmax]
        # Calculate Deffs for only the timepoints needed and add as a new
        # column
        Deff_list = []
        for i in range(0, len(temp2_msd)):
            index = temp2_msd.index[i]
            # Calculate Deff using the conventional relationship between
            # MSD and Deff
            Deff_list.append(temp2_msd[particle_chemistry + ' geo'][index]/(4*index**ALPHA))
        temp2_msd[particle_chemistry + 'Deff'] = Deff_list
        # Plot histogram and print mean Deff value
        # NOTE: Eventually I'll migrate the plot to bokeh; I'm using
        # matplotlib temporarily for ease of testing
        plt.hist(temp2_msd[particle_chemistry + 'Deff'], bins=15)
        plt.xlabel('Calculated Deffs')
        plt.ylabel('Count')
        plt.show()
        Deff = scipy.stats.gmean(temp2_msd[particle_chemistry + 'Deff'])
        return Deff


def compute_plot_all_Deff(tmin,tmax):
    """
    Calculates and plots all Deffs in the timepoint range.

    This function trims the cleaned MSD pandas dataframe to the user-
    selected timepoint range, calculates a Deff from the mean MSD for
    each timepoint within the range and for each particle chemistry,
    plots a line graph of all Deffs across the timepoint range, and
    gives geometric mean Deff values for each chemistry for the time
    range specified.
    Note: tmin must be less than tmax.

    Inputs:
    tmin: a float representing the minimum timepoint the user wishes to
    consider in the Deff calculation
    tmax: a float representing the maximum timepoint the user wishes to
    consider in the Deff calculation

    Outputs:
    A bokeh line plot of Deffs for all particle chemistries across the
    timepoint range
    Deffs (printed): a fresh pandas dataframe of the same dimensions
    (timepoints [trimmed] and columns) as the MSD dataframe, containing
    Deffs instead of MSDs
    avg_Deffs: a fresh pandas dataframe indexed with the columns
    (particle chemistries) of the MSD dataframe, containing single
    geometric mean Deff values for each particle chemistry
    """
    # Verify time range validity
    if tmin < 0 or tmax > 20:
        return 'Error: input time range between 0 and 20 sec'
    else:
        if tmin == 0:
            print 'Divide by 0 error: tmin=0 changed to tmin=0.01'
            tmin = 0.01
        # Trim out-of-time-range rows
        temp1_msd = msd[msd.index >= tmin]
        temp2_msd = temp1_msd[temp1_msd.index <= tmax]
        # Calculate Deffs for only the timepoints needed and add as a new
        # column to a new dataframe
        index = temp2_msd.index
        Deffs = pd.DataFrame(index=index, columns=columns2)
        avg_Deffs = pd.DataFrame(index=columns2)
        avg_Deffs_temp = []
        output_file('Deffs_plot.html')
        p = figure(tools='resize,pan,box_zoom,wheel_zoom,reset,save', x_axis_label='MSD timepoint', y_axis_label='Deff')
        for title in columns2:
            single_Deff_list = []
            for i in range(0, len(temp2_msd)):
                index = temp2_msd.index[i]
                single_Deff_list.append(temp2_msd[title + ' geo'][index]/(4*index**ALPHA))
            # Add paricle-chemistry-specific Deff list to Deffs dataframe
            Deffs[title] = single_Deff_list
            # Add geometric mean Deff to what will become avg_Deffs
            avg_Deffs_temp.append(scipy.stats.gmean(Deffs[title]))
            p.line(Deffs.index, Deffs[title], legend=title, line_color=(np.random.randint(256),np.random.randint(256),np.random.randint(256)))
        avg_Deffs['Deff'] = avg_Deffs_temp
        p.legend.label_text_font_size = '6pt'
        # p.legend.label_width = 50
        # p.legend.label_height = 6
        show(p)
        # puke Deff lists while returning the much prettier avg_Deffs table
        print Deffs
        return avg_Deffs