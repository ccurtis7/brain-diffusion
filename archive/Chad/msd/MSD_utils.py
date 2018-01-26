import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.ma as ma
import scipy.stats as stat


def get_data_gels(channels, surface_functionalities, slices, base, path):
    """
    Loads data from csv files and outputs a dictionary following a specified
        sample naming convection determined by the input

    Parameters:
    channels, surface functionalities, media, and concentrations, and replicates
        can take ranges or lists.
    path is string with substition placeholders for concentration and sample
        name (built from channels, surface_functionalities, media,
        concentrations, and replicates).

    Example:
    path = "./{concentration}_ACSF/geoM2xy_{sample_name}.csv";
    get_data(["RED", "YG"], ["PEG", "noPEG"], ["in_agarose"],
    ["0_1x", "1x", "10x"], ["S1", "S2", "S3", "S4"], path)
    """

    data = {}
    avg_sets = {}
    counter = 0

    for channel in channels:
        for surface_functionality in surface_functionalities:
            test_value = "{}_{}_{}".format(channel, surface_functionality, base)
            avg_sets[counter] = test_value
            counter = counter + 1
            for slic in slices:
                sample_name = "{}_{}_{}_{}".format(channel, surface_functionality, base, slic)
                filename = path.format(channel=channel, functionality=surface_functionality, slic=slic, sample_name=sample_name)
                data[sample_name] = np.genfromtxt(filename, delimiter=",")

    return data, avg_sets


def data_prep_for_plotting_gels(path, frames, SD_frames, conversion, to_frame, parameters, base):
    """
    A summary function that preps my mean mean squared displacement data for graphing by performing averages over slices,
    creates time arrays, calculates standard deviations, and organized nomenclature.

    Inputs:

    path: string. A path to where the MMSD data resides e.g. path = "./{genotype}/geoM2xy_{sample_name}.csv"
    frames: integer.  Number of frames expected in MMSD datasets e.g. frames = 60
    SD_frames: array of integers.  Desired frames at which to plot standard deviation bars e.g. SD_frames = [1, 7, 14, 15]
    conversion: list of three floats.  First is the microns per pixel, second is the frames per second, and third is the microns
        per slice e.g. conversion = (0.3, 3.95, 1)
    to_frame: integer.  Frame to which you which to plot.
    parameters: a dictionary of the form:

    parameters = {}
    parameters["channels"] = ["RED"]
    parameters["genotypes"] = ["WT"]
    parameters["pups"] = ["P1", "P2", "P3"]
    parameters["surface functionalities"] = ["PEG"]
    parameters["slices"] = ["S1", "S2", "S3"]
    parameters["regions"] = ["cortex", "mid"]
    parameters["replicates"] = [1, 2, 3, 4, 5]

    Outputs:

    data: dictionary of arrays, with keys being names of individual datasets and entries being the MSDs of the datasets.
    avg_over_slices: numbered dictionary with entries being names of datasets averaged over slices.
    names_with_replicates: numbered dictionaries with entries being raw names of datasets (original names without
        averaging over replicates.)
    time: array.  Time array calculated based on frames and frames per second conversion.
    time_SD: array.  Time array with entries at times when standard deviations are calculated.
    average_over_slices: dictionary of arrays containing data averaged over slices.
    all_SD_over_slices: dictionary of arrays containing standard deviations calculated over slices.

    """

    data, avg_over_slices = get_data_gels(parameters["channels"], parameters["surface functionalities"], parameters["slices"], base, path)
    time, time_SD = build_time_array(frames, conversion, SD_frames)
    average_over_slices = avg_all(data, frames, avg_over_slices)
    all_SD_over_slices = SD_all(data, frames, SD_frames, avg_over_slices)

    return data, avg_over_slices, time, time_SD, average_over_slices, all_SD_over_slices


def plot_all_MSD_histograms_gels(parameters, base, folder, dataset, time, bins, desired_time, diffusion_data=False, dimension="2D",
                                 set_y_limit=False, y_range=40, set_x_limit=False, x_range=40):
    """
    This function plots histograms for all datasets in an experiment.  The output from calculate_MMSDs
    is used as an input to this function.

    Inputs:
    parameters: dictionary of form:

    parameters = {}
    parameters["channels"] = ["RED", "YG"]
    parameters["genotypes"] = ["KO"]
    parameters["pups"] = ["P1", "P2", "P3"]
    parameters["surface functionalities"] = ["PEG"]
    parameters["slices"] = ["S1", "S2", "S3"]
    parameters["regions"] = ["cortex", "mid"]
    parameters["replicates"] = [1, 2, 3, 4, 5]

    dataset: dictionary of dictionaries of arrays of floats.  Contains MSD data with keys corresponding to
        (1) sample names and (2) particle numbers. Use output from calculate_MMSDs preferably.
    time: array of floats.  Contains corresponding time data. Must be same for all particles in MSD dictionary.
        Must also be one unit longer than MSD datasets.
    bins: integer or array. desired number of bins.
    filename: string.  desired name of file.  File will automatically be saved as a PNG.
    desired_time: float.  Time at which to measure MSDs for histogram.
    diffusion_data: True/False.  If true, will plot diffusion data instead of MSD data.
    dimension: string.  1D, 2D, or 3D.
    set_y_limit: True/False.  Option to manually set the y limit of graph.
    y_range: float.  Manually set y limit.
    set_x_limit: True/False.  Option to manually set the x limit of graph.
    x_range: float. Manually set x limit.
    """

    channels = parameters["channels"]
    surface_functionalities = parameters["surface functionalities"]
    slices = parameters["slices"]

    for channel in channels:
            for surface_functionality in surface_functionalities:
                slice_counter = 0
                for slic in slices:

                    sample_name = "{}_{}_{}_{}".format(channel, surface_functionality, base, slic)
                    # print("name is", sample_name)
                    Hplot = folder.format(functionality=surface_functionality, slic=slic)+'{}_Hplot'.format(sample_name)

                    plt.gcf().clear()
                    plot_MSD_histogram(dataset[sample_name], time, bins, Hplot, desired_time, diffusion_data=diffusion_data,
                                       dimension=dimension, set_y_limit=set_y_limit, y_range=y_range,
                                       set_x_limit=set_x_limit, x_range=x_range)


def quality_control_gels(path2, folder, frames, conversion, parameters, base, interv, cut):
    """
    This function plots a histogram of trajectory lengths (in units of frames) and two types of plots of
    trajectories (original and overlay).

    Inputs:
    path2: string.  Name of input trajectory csv files.
    folder: string. Name of folder to which to save results.
    frames: integer.  Total number of frames in videos.
    conversion: array.  Contains microns per pixel and frames per second of videos.
    parameters:

    parameters = {}
    parameters["channels"] = ["RED"]
    parameters["surface functionalities"] = ["PEG"]
    parameters["slices"] = ["S1", "S2", "S3"]
    parameters["replicates"] = [1, 2, 3, 4]

    cut: integer.  Minimum number of frames a trajectory must have in order to be plotted.
    """

    channels = parameters["channels"]
    surface_functionalities = parameters["surface functionalities"]
    slices = parameters["slices"]
    replicates = parameters["replicates"]
    SD_frames = [1, 7, 14, 15]

    trajectory = {}
    names_with_replicates = {}
    data = {}

    particles_unfiltered = {}
    framed_unfiltered = {}
    x_data_unfiltered = {}
    y_data_unfiltered = {}
    total_unfiltered = {}
    particles = {}
    framed = {}
    x_data = {}
    y_data = {}
    total = {}
    tlength = {}
    x_microns = {}
    y_microns = {}
    x_particle = {}
    y_particle = {}

    x_original_frames = {}
    y_original_frames = {}

    x_adjusted_frames = {}
    y_adjusted_frames = {}

    counter2 = 0

    for channel in channels:
            for surface_functionality in surface_functionalities:
                        for slic in slices:
                            for replicate in replicates:
                                # Establishing sample names and extracting data from csv files.
                                counter2 = counter2 + 1
                                sample_name_long = "{}_{}_{}_{}_{}".format(channel, surface_functionality, base, slic, replicate)
                                names_with_replicates[counter2] = sample_name_long

                                filename = path2.format(functionality=surface_functionality, slic=slic, sample_name=sample_name_long)
                                data[sample_name_long] = np.genfromtxt(filename, delimiter=",")
                                data[sample_name_long] = np.delete(data[sample_name_long], 0, 1)

                                # Names of output plots
                                fold = folder.format(functionality=surface_functionality, slic=slic)
                                logplot = fold + '{}_logplot'.format(sample_name_long)
                                Mplot = fold + '{}_Mplot'.format(sample_name_long)
                                Dplot = fold + '{}_Dplot'.format(sample_name_long)
                                Hplot = fold + '{}_Hplot'.format(sample_name_long)
                                Hlogplot = fold + '{}_Hlogplot'.format(sample_name_long)
                                Cplot = fold + '{}_Cplot'.format(sample_name_long)
                                Tplot = fold + '{}_Tplot'.format(sample_name_long)
                                T2plot = fold + '{}_T2plot'.format(sample_name_long)
                                lenplot = fold + '{}_lenplot'.format(sample_name_long)

                                # Fill in data and split into individual trajectories
                                particles_unfiltered[counter2], framed_unfiltered[counter2], x_data_unfiltered[counter2],\
                                    y_data_unfiltered[counter2] = fill_in_and_split(data[names_with_replicates[counter2]])

                                total_unfiltered[counter2] = int(max(particles_unfiltered[counter2]))

                                # Filter out short trajectories
                                particles[counter2], framed[counter2], x_data[counter2], y_data[counter2] =\
                                    filter_out_short_traj(particles_unfiltered[counter2], framed_unfiltered[counter2],
                                                          x_data_unfiltered[counter2], y_data_unfiltered[counter2], cut)

                                # Convert to microns and seconds
                                time, time_SD = build_time_array(frames, conversion, SD_frames)
                                framen = np.linspace(0, frames, frames+1).astype(np.int64)
                                total[counter2] = int(max(particles[counter2]))
                                tlength[counter2] = np.zeros(total[counter2])

                                x_microns[counter2] = x_data[counter2]*conversion[0]
                                y_microns[counter2] = y_data[counter2]*conversion[0]

                                # Adjust frames (probably unneccesary, but I did it...)
                                x_particle[counter2] = {}
                                y_particle[counter2] = {}

                                x_original_frames[counter2] = {}
                                y_original_frames[counter2] = {}

                                x_adjusted_frames[counter2] = {}
                                y_adjusted_frames[counter2] = {}

                                for num in range(1, total[counter2] + 1):
                                    hold = np.where(particles[counter2] == num)
                                    itindex = hold[0]
                                    min1 = min(itindex)
                                    max1 = max(itindex)
                                    x_particle[counter2][num] = x_microns[counter2][min1:max1+1]
                                    y_particle[counter2][num] = y_microns[counter2][min1:max1+1]

                                    x_original_frames[counter2][num] = np.zeros(frames + 1)
                                    y_original_frames[counter2][num] = np.zeros(frames + 1)
                                    x_adjusted_frames[counter2][num] = np.zeros(frames + 1)
                                    y_adjusted_frames[counter2][num] = np.zeros(frames + 1)

                                    x_original_frames[counter2][num][framed[counter2][min1]:framed[counter2][max1]+1]\
                                        = x_microns[counter2][min1:max1+1]
                                    y_original_frames[counter2][num][framed[counter2][min1]:framed[counter2][max1]+1]\
                                        = y_microns[counter2][min1:max1+1]

                                    x_adjusted_frames[counter2][num][0:max1+1-min1] = x_microns[counter2][min1:max1+1]
                                    y_adjusted_frames[counter2][num][0:max1+1-min1] = y_microns[counter2][min1:max1+1]

                                    x_original_frames[counter2][num] = ma.masked_equal(x_original_frames[counter2][num], 0)
                                    y_original_frames[counter2][num] = ma.masked_equal(y_original_frames[counter2][num], 0)
                                    x_adjusted_frames[counter2][num] = ma.masked_equal(x_adjusted_frames[counter2][num], 0)
                                    y_adjusted_frames[counter2][num] = ma.masked_equal(y_adjusted_frames[counter2][num], 0)

                                    tlength[counter2][num - 1] = ma.count(x_adjusted_frames[counter2][num])

                                plot_traj(x_original_frames[counter2], y_original_frames[counter2], total[counter2], interv, T2plot)
                                plt.gcf().clear()
                                plot_trajectory_overlay(x_original_frames[counter2], y_original_frames[counter2], 6, 2, 6, Tplot)
                                plt.gcf().clear()
                                plot_traj_length_histogram(tlength[counter2], max(tlength[counter2]), lenplot)
                                plt.gcf().clear()


def calculate_diffusion_coefficients_gels(channels, surface_functionalities, slices, path, time, time_to_calculate, to_frame, dimension):

    """
    Loads data from csv files and outputs a dictionary following a specified
        sample naming convection determined by the input

    Parameters:
    channels, surface functionalities, media, and concentrations, and replicates
        can take ranges or lists.
    path is string with substition placeholders for concentration and sample
        name (built from channels, surface_functionalities, media,
        concentrations, and replicates).

    Example:
    path = "./{genotype}/{pup}/{region}/{channel}/geoM2xy_{sample_name}.csv";
    get_data(["RED", "YG"], ["WT", "KO", "HET"], ["P1", "P2", "P3", "P4"],
    ["PEG", "noPEG"], ["S1", "S2", "S3", "S4"], ["cortex", "hipp", "mid"],
    [1, 2, 3, 4, 5], path)
    """

    data = {}
    avg_over_slices_raw = {}
    avg_over_pups_raw = {}
    names_with_replicates = {}
    counter = 0
    counter2 = 0

    diffusion_coef_point_derivative = {}
    diffusion_coef_linear_fit = {}

    for channel in channels:
        for surface_functionality in surface_functionalities:
            for slic in slices:
                test_value = "{}_{}_0-4p_agarose_{}".format(channel, surface_functionality, slic)
                avg_over_slices_raw[counter] = test_value
                counter = counter + 1
                sample_name = test_value
                for replicate in replicates:
                    sample_name_long = test_value + "_{}".format(replicate)
                    names_with_replicates[counter2] = sample_name_long
                    counter2 = counter2 + 1
                filename = path.format(functionality=surface_functionality, slic=slic, sample_name=sample_name)
                data[sample_name] = np.genfromtxt(filename, delimiter=",")

                diffusion_coef_point_derivative[sample_name] =\
                    diffusion_coefficient_point_derivative(data[sample_name], time, time_to_calculate, dimension)
                diffusion_coef_linear_fit[sample_name] =\
                    diffusion_coefficient_linear_regression(data[sample_name], time, to_frame, dimension)

    return diffusion_coef_point_derivative, diffusion_coef_linear_fit


def build_time_array(frames=90, conversion=(0.16, 9.89, 1), SD_frames=[1, 7, 14]):
    """
    Builds (1) a time array based on the desire number of frames and the fps and (2) a shortened time array at which the
    standard deviations are going to be calculated.

    The default is 100 frames, a conversion factor of 0.16 microns per pixel, 9.89 frames per second, and 1 micron per z-stack.

    Example:
    build_time_array(90, (0.16, 9.89, 1), [1, 7, 14, 15])
    """

    frames_1 = np.linspace(0, frames, frames+1).astype(np.int64)
    time = frames_1/conversion[1]
    time_SD = np.zeros(np.size(SD_frames))

    for i in range(0, np.size(SD_frames)):
        time_SD[i] = time[SD_frames[i]]

    return time, time_SD


def return_average(data, frames=90, to_average='YG_nPEG_in_agarose_1x'):
    """
    Averages over replicates within a sample.

    Parameters:
    to_average is a string.

    Example:
    return_average('RED_PEG_in_agarose_10x')
    """

    to_avg = {}
    counter = 0
    for keys in data:
        if to_average in keys:
            to_avg[counter] = keys
            counter = counter + 1

    if not isinstance(data[to_avg[0]], float):

        to_avg_num = np.zeros((frames, counter))
        for i in range(0, counter):
            for j in range(0, frames):
                to_avg_num[j, i] = data[to_avg[i]][j]

        answer = np.mean(to_avg_num, axis=1)
    else:
        to_avg_num = np.zeros(counter)
        for i in range(0, counter):
            to_avg_num[i] = data[to_avg[i]]

        answer = np.mean(to_avg_num)

    return answer


def avg_all(data, frames, avg_sets):
    """
    Averages over all replicates in a dataset.

    data is a dictionary defined in function get_data.
    frames is the number of frames in each experiment.
    avg_sets is a dictionary of strings (keys are 0-N)

    Example:
    avg_all(data, 90, {0: 'RED_PEG_in_agarose_0_1x', 1: 'RED_PEG_in_agarose_1x'})
    """

    all_avg = {}
    for keys in avg_sets:
        all_avg[avg_sets[keys]] = return_average(data, frames, avg_sets[keys])
    return all_avg


def get_data_pups(channels, genotypes, pups, surface_functionalities, slices, regions, replicates, path):

    """
    Loads data from csv files and outputs a dictionary following a specified
        sample naming convection determined by the input

    Parameters:
    channels, surface functionalities, media, and concentrations, and replicates
        can take ranges or lists.
    path is string with substition placeholders for concentration and sample
        name (built from channels, surface_functionalities, media,
        concentrations, and replicates).

    Example:
    path = "./{genotype}/{pup}/{region}/{channel}/geoM2xy_{sample_name}.csv";
    get_data(["RED", "YG"], ["WT", "KO", "HET"], ["P1", "P2", "P3", "P4"],
    ["PEG", "noPEG"], ["S1", "S2", "S3", "S4"], ["cortex", "hipp", "mid"],
    [1, 2, 3, 4, 5], path)
    """

    data = {}
    avg_over_slices_raw = {}
    avg_over_pups_raw = {}
    names_with_replicates = {}
    counter = 0
    counter2 = 0

    for channel in channels:
        for genotype in genotypes:
            for surface_functionality in surface_functionalities:
                for region in regions:
                    for pup in pups:
                        for slic in slices:
                            test_value = "{}_{}_{}_{}_{}".format(channel, genotype, surface_functionality, region, pup)
                            avg_over_slices_raw[counter] = test_value
                            test_value2 = "{}_{}_{}_{}".format(channel, genotype, surface_functionality, region)
                            avg_over_pups_raw[counter] = test_value2
                            counter = counter + 1
                            sample_name = "{}_{}_{}_{}_{}_{}".format(channel, genotype, surface_functionality, region, pup, slic)
                            for replicate in replicates:
                                sample_name_long = "{}_{}_{}_{}_{}_{}_{}".format(channel, genotype, surface_functionality,
                                                                                 region, pup, slic, replicate)
                                names_with_replicates[counter2] = sample_name_long
                                counter2 = counter2 + 1
                            filename = path.format(channel=channel, genotype=genotype, pup=pup, region=region, sample_name=sample_name)
                            data[sample_name] = np.genfromtxt(filename, delimiter=",")

    avg_over_slices = {}
    avg_over_pups = {}

    counter = 0
    for key, value in avg_over_slices_raw.items():
        if value not in avg_over_slices.values():
            avg_over_slices[counter] = value
            counter = counter + 1

    counter = 0
    for key, value in avg_over_pups_raw.items():
        if value not in avg_over_pups.values():
            avg_over_pups[counter] = value
            counter = counter + 1

    return data, avg_over_slices, avg_over_pups, names_with_replicates


def return_SD(data, frames=90, SD_frames=[1, 7, 14, 15], to_stdev='YG_nPEG_in_agarose_1x'):
    """
    Finds standard deviation over replicates within a sample at frames specified by SD_frames.

    Parameters:
    to_average is a string.

    Example:
    return_SD(data, 90, [1, 7, 14, 15], 'RED_PEG_in_agarose_10x')
    """

    to_SD = {}
    counter = 0
    for keys in data:
        if to_stdev in keys:
            to_SD[counter] = keys
            counter = counter + 1

    if not isinstance(data[to_SD[0]], float):
        to_SD_num = np.zeros((frames, counter))
        for i in range(0, counter):
            for j in range(0, frames):
                to_SD_num[j, i] = data[to_SD[i]][j]

        answer = np.std(to_SD_num, axis=1)
        answer_sep = answer[SD_frames[:]]

    else:
        to_SD_num = np.zeros(counter)
        for i in range(0, counter):
            to_SD_num[i] = data[to_SD[i]]

        answer_sep = np.std(to_SD_num)

    return answer_sep


def SD_all(data, frames, SD_frames, avg_sets):
    """
    Finds standard deviations over all replicates in a dataset.

    data is a dictionary defined in function get_data.
    frames is the number of frames in each experiment.
    avg_sets is a dictionary of strings (keys are 0-N)

    Example:
    SD_all(data, 90, [1, 7, 14, 15], {0: 'RED_PEG_in_agarose_0_1x', 1: 'RED_PEG_in_agarose_1x'})
    """

    all_SD = {}
    for keys in avg_sets:
        all_SD[avg_sets[keys]] = return_SD(data, frames, SD_frames, avg_sets[keys])
    return all_SD


def range_and_ticks(range_to_graph, to_frame, manual_decimals=False,
                    manual_decimals_val=2):
    """
    This is a useful function that calculates the best range, tick mark interval
    size, and decimals displayed for a given input dataset.

    Inputs:
    range_to_graph: numpy array of data to be graphed.
    to_frame: integer, limits data to be graphed to the range [0, to_frame].
    manual_decimals: True/False.  Allow the user to manually adjust the number of decimals displayed.
    manual_decimals_val: integer.  Number of decimals the user desires.

    Outputs:
    y_range: upper boundary containing the entire range of range_to_graph[0, to_frame].
    ticks: tick interval.
    decimals: number of decimals to display.

    This function must be modified if I need to plot graphs with negative values.
    I should also include symmetrical capabilities when I want to plot negative values.
    """

    graph_max = np.nanmax(range_to_graph[0:to_frame])  # Define max within range

    if np.ceil(np.log10(graph_max)) >= 0:  # Find nearest 10^x that contains graph_max
        base = np.ceil(np.log10(graph_max))
        raw_max = 10**base
        decimals = int(1)
    else:
        base = np.ceil(np.log10(1/graph_max))
        raw_max = 10**(-base)
        decimals = int(base + 1)

    range_correct = False
    cut_down_max = 0.1*raw_max
    counter = -1
    while range_correct is False:  # Make sure that most of the graph space is used efficiently (75%).
        if graph_max/raw_max > 0.75:
            y_range = raw_max
            range_correct = True
        else:
            raw_max = raw_max - cut_down_max
            range_correct = False
        counter = counter + 1

    if graph_max > y_range:  # Make sure that graph_max doesn't exceed limits
        y_range = y_range + cut_down_max

    if counter % 2 == 0:  # Define tick size. I based it off of 0.1*raw_max, which is always a 10^x
        ticks = cut_down_max * 2
    else:
        ticks = cut_down_max

    range_correct = False  # Modifies the tick size if there aren't enough ticks on the graph.
    while range_correct is False:
        if ticks/y_range >= 0.24:
            ticks = ticks/2
            range_correct = False
        else:
            range_correct = True

    if manual_decimals:  # Allow user to set manual decimal values.
        decimals = manual_decimals_val

    return y_range, ticks, decimals


def choose_y_axis_params(all_avg, in_name1, in_name2, to_frame):
    """
    When plotting multiple trajectories, I need to choose plot parameters based
    on multiple datasets.  This function uses the function ranges_and_ticks to
    choose the best parameters based on multiple trajectories.

    Inputs:
    all_avg: dictionary of numpy arrays.  A subset of this data will be plotted
        based on the parameters in_name1 and in_name2.
    in_name1: string.  in_name1 should be found within the keys in all_avg of
        all datasets to be plotted e.g. if I want to plot all RED datasets, the
        I could use in_name1=RED.
    in_name2: similar criteria to in_name1.  If I want to narrow the data to be
        plotted to PEG datasets, then, in_name2 could be "_PEG" (if you only use
        PEG, it will plot all nPEG and PEG datasets, due to my chosen
        nomenclature. Just be aware of your naming system).
    to_frame: integer, limits data to be graphed to the range [0, to_frame].

    Outputs:
    y_range_final: upper boundary containing the entire range of
        range_to_graph[0, to_frame].
    ticks_final: tick interval.
    decimals_final: number of decimals to display.

    Example:
    choose_y_axis_params(all_avg, "RED", "_PEG", 15)
    """

    to_graph = {}
    counter = 0

    for keys in all_avg:
        if in_name1 in keys and in_name2 in keys:
            to_graph[counter] = keys
            counter = counter + 1

    y_range = np.zeros(counter)
    ticks = np.zeros(counter)
    decimals = np.zeros(counter)
    base = np.zeros(counter)

    for keys in to_graph:
        y_range[keys], ticks[keys], decimals[keys] = range_and_ticks(all_avg[to_graph[keys]], to_frame)

    one_to_graph = np.argmax(y_range)
    y_range_final = y_range[one_to_graph]
    ticks_final = ticks[one_to_graph]
    decimals_final = int(decimals[one_to_graph])

    return y_range_final, ticks_final, decimals_final


def data_prep_for_plotting_pups(path, frames, SD_frames, conversion, to_frame, parameters):
    """
    A summary function that preps my mean mean squared displacement data for graphing by performing averages over slices and
    pups, creates time arrays, calculates standard deviations, and organized nomenclature.

    Inputs:

    path: string. A path to where the MMSD data resides e.g. path = "./{genotype}/geoM2xy_{sample_name}.csv"
    frames: integer.  Number of frames expected in MMSD datasets e.g. frames = 60
    SD_frames: array of integers.  Desired frames at which to plot standard deviation bars e.g. SD_frames = [1, 7, 14, 15]
    conversion: list of three floats.  First is the microns per pixel, second is the frames per second, and third is the microns
        per slice e.g. conversion = (0.3, 3.95, 1)
    to_frame: integer.  Frame to which you which to plot.
    parameters: a dictionary of the form:

    parameters = {}
    parameters["channels"] = ["RED"]
    parameters["genotypes"] = ["WT"]
    parameters["pups"] = ["P1", "P2", "P3"]
    parameters["surface functionalities"] = ["PEG"]
    parameters["slices"] = ["S1", "S2", "S3"]
    parameters["regions"] = ["cortex", "mid"]
    parameters["replicates"] = [1, 2, 3, 4, 5]

    Outputs:

    data: dictionary of arrays, with keys being names of individual datasets and entries being the MSDs of the datasets.
    avg_over_slices: numbered dictionary with entries being names of datasets averaged over slices.
    avg_over_pups: numbered dictionary with entries being names of datasets averages over pups.
    names_with_replicates: numbered dictionaries with entries being raw names of datasets (original names without
        averaging over replicates.)
    time: array.  Time array calculated based on frames and frames per second conversion.
    time_SD: array.  Time array with entries at times when standard deviations are calculated.
    average_over_slices: dictionary of arrays containing data averaged over slices.
    average_over_pups: dictionary of arrays containing data averaged over pups.
    all_SD_over_slices: dictionary of arrays containing standard deviations calculated over slices.
    all_SD_over_pups: dictionary of arrays containing standard deviations calculated over pups.

    """

    data = {}
    avg_sets = {}
    all_avg = {}
    all_SD = {}
    names_with_replicates = {}

    data, avg_over_slices, avg_over_pups, names_with_replicates = get_data_pups(parameters["channels"], parameters["genotypes"], parameters["pups"],
                                                                                parameters["surface functionalities"], parameters["slices"],
                                                                                parameters["regions"], parameters["replicates"], path)
    time, time_SD = build_time_array(frames, conversion, SD_frames)
    average_over_slices = avg_all(data, frames, avg_over_slices)
    average_over_pups = avg_all(average_over_slices, frames, avg_over_pups)
    all_SD_over_slices = SD_all(data, frames, SD_frames, avg_over_slices)
    all_SD_over_pups = SD_all(average_over_slices, frames, SD_frames, avg_over_pups)

    return data, avg_over_slices, avg_over_pups, names_with_replicates, time, time_SD,\
        average_over_slices, average_over_pups, all_SD_over_slices, all_SD_over_pups


def graph_single_variable(all_avg, all_SD, time, time_SD, SD_frames, in_name1, in_name2, to_frame=15,
                          line_colors=['g', 'r', 'b', 'c', 'm', 'k'], line_kind='-', x_manual=False,
                          y_range=10, ticks_y=2, dec_y=0, x_range=5, ticks_x=1, dec_x=0, label_size=95,
                          legend_size=40, tick_size=50, line_width=10, fig_size=(20, 18),
                          modify_labels=False, label_identifier="agarose_", base_name="KO"):
    """
    A handy plotting function to examine individual datasets within the larger dataset.

    Inputs:
    all_avg: dictionary of numpy arrays.  A subset of this data will be plotted
        based on the parameters in_name1 and in_name2. Keys contain the name of
        the dataset.
    all_SD: dictionary of numpy arrays.  Mirrors the all_avg dataset.
    time: numpy array.  Assumes time array is the same between datasets
        contained in all_avg.
    time_SD: numpy array.  Contains timepoints at which SDs are desired for the
        plots.
    SD_frames: numpy array.  Contains frames at which SDs are desired for the
        plots.
    in_name1: string.  in_name1 should be found within the keys in all_avg of
        all datasets to be plotted e.g. if I want to plot all RED datasets, the
        I could use in_name1=RED.
    in_name2: similar criteria to in_name1.  If I want to narrow the data to be
        plotted to PEG datasets, then, in_name2 could be "_PEG" (if you only use
        PEG, it will plot all nPEG and PEG datasets, due to my chosen
        nomenclature. Just be aware of your naming system).

    to_frame: integer, limits data to be graphed to the range [0, to_frame].
    line_colors: list of strings.  Contains colors desired for plot.
    line_kind: string.  Contains a single string for the desired line format
        of all lines in graph.
    x_manual: True/False.  If True, the user must define range, ticks, and
        decimals for both x and y.
    y_range: y range of the plot.
    ticks_y: interval size on y axis.
    dec_y: number of decimals displayed on y axis.
    x_range" x range of the plot.
    ticks_x: intervals size on x axis.
    dec_x: number of decimals displayed on the x axis.
    label_size: Size of labels on x and y axes.
    legend_size: size of text in legend.
    tick_size: size of text of ticks.
    line_width: width of lines on plot.
    fig_size: x,y size of plot e.g. (20, 18).
    modify_labels: True/False.  If True, the user must give a label_identifier.
    label_identifier: string.  User puts in a string that is contained in all keys
        of all_avg.  Labels of data will now be whatever follows label_identifier
        e.g. if a key contains "agarose_S1" and I use "agarose_", the legend
        will read "S1".

    To do:
    I need to make my labelling more flexible eventually.  The troubl is correctly
    lining up my labels with the data in all_avg.
    I also need to make it possible to plot in a desired order e.g. 0.1x, 1x, 10x.
    """

    to_graph = {}
    line_type = {}
    counter = 0
    labels = {}

    if not x_manual:
        y_range, ticks_y, dec_y = choose_y_axis_params(all_avg, in_name1, in_name2, to_frame)
        x_range, ticks_x, dec_x = range_and_ticks(time, to_frame)
    else:
        y_range, ticks_y, dec_y = y_range, ticks_y, dec_y
        x_range, ticks_x, dec_x = x_range, ticks_x, dec_x

    filename = base_name + "_" + in_name1 + "_" + in_name2 + ".png"

    # Creates figure
    fig = plt.figure(figsize=fig_size, dpi=80)
    ax = fig.add_subplot(111)

    for keys in all_avg:
        if in_name1 in keys and in_name2 in keys:
            to_graph[counter] = keys
            counter = counter + 1

    for keys in to_graph:
        if modify_labels:
            labels[keys] = to_graph[keys].split(label_identifier, 1)[1]
        else:
            labels[keys] = to_graph[keys]
        line_type[keys] = line_colors[keys]+line_kind
        ax.plot(time[0:to_frame], all_avg[to_graph[keys]][0:to_frame], line_type[keys], linewidth=line_width, label=labels[keys])
        ax.errorbar(time_SD, all_avg[to_graph[keys]][SD_frames], all_SD[to_graph[keys]], fmt='', linestyle='',
                    capsize=7, capthick=2, elinewidth=2, color=line_colors[keys])

    # A few adjustments to prettify the graph
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(tick_size)

    xmajor_ticks = np.arange(0, x_range+0.0001, ticks_x)
    ymajor_ticks = np.arange(0, y_range+0.0001, ticks_y)

    ax.set_xticks(xmajor_ticks)
    plt.xticks(rotation=-30)
    ax.set_yticks(ymajor_ticks)
    ax.title.set_fontsize(tick_size)
    ax.set_xlabel('Time (s)', fontsize=label_size)
    ax.set_ylabel(r'MSD ($\mu$m$^2$)', fontsize=label_size)
    ax.tick_params(direction='out', pad=16)
    ax.legend(loc=(0.02, 0.75), prop={'size': legend_size})
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec_x)))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec_y)))

    # plt.yscale('log')
    # plt.xscale('log')
    plt.gca().set_xlim([0, x_range+0.0001])
    plt.gca().set_ylim([0, y_range+0.0001])

    # Save your figure
    plt.savefig('{}'.format(filename), bbox_inches='tight')
    return y_range, ticks_y, dec_y, x_range, ticks_x, dec_x


def fill_in_and_split(data):
    """
    This function takes a raw trajectory datafile (with the first column deleted),
    splits it into particles, frames, x data, and y data arrays, and fills in
    any skipped frames.

    Inputs:
    data: array.  First column are particle numbers, second frames, third x data,
        forth, y data.

    Outputs:
    particles_new
    framed_new
    x_data_new
    y_data_new
    """

    particles = data[:, 0]
    framed = data[:, 1].astype(np.int64)
    x_data = data[:, 2]
    y_data = data[:, 3]

    counter = 0
    original_size = framed.size

    for num in range(0, original_size - 1):
        if framed[num+1] > framed[num] and particles[num+1] == particles[num]:
            if not framed[num+1] - framed[num] == 1:
                new_frames = framed[num+1] - framed[num] - 1
                counter = counter + new_frames

    new_size = original_size + counter

    if not counter == 0:
        particles_new = np.zeros(new_size)
        framed_new = np.zeros(new_size)
        x_data_new = np.zeros(new_size)
        y_data_new = np.zeros(new_size)

        counter2 = 0

        for num in range(0, original_size - 1):
            if framed[num+1] > framed[num] and particles[num+1] == particles[num]:
                if not framed[num+1] - framed[num] == 1:
                    new_frames = framed[num+1] - framed[num] - 1
                    for num2 in range(1, new_frames):
                        particles_new[num+num2+counter2] = particles[num]
                        framed_new[num+num2+counter2] = framed[num]
                        x_data_new[num+num2+counter2] = x_data[num]
                        y_data_new[num+num2+counter2] = y_data[num]
                    counter2 = counter2 + new_frames
                else:
                    particles_new[num+counter2] = particles[num]
                    framed_new[num+counter2] = framed[num]
                    x_data_new[num+counter2] = x_data[num]
                    y_data_new[num+counter2] = y_data[num]
            else:
                particles_new[num+counter2] = particles[num]
                framed_new[num+counter2] = framed[num]
                x_data_new[num+counter2] = x_data[num]
                y_data_new[num+counter2] = y_data[num]

        particles_new[original_size+counter2 - 1] = particles[original_size - 1]
        framed_new[original_size+counter2 - 1] = framed[original_size - 1]
        x_data_new[original_size+counter2 - 1] = x_data[original_size - 1]
        y_data_new[original_size+counter2 - 1] = y_data[original_size - 1]
    else:
        particles_new = particles
        framed_new = framed
        x_data_new = x_data
        y_data_new = y_data

    return particles_new, framed_new, x_data_new, y_data_new


def plot_traj_length_histogram(length, total, lenplot):

    total1 = total
    hist = length

    hist = [x for x in hist if str(x) != 'nan']

    plot, bins = np.histogram(hist, bins=total1)
    width = 1 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:])/2
    plt.bar(center, plot, align='center', width=width)
    plt.xlabel('Trajectory lengths', fontsize=20)

    plt.savefig('{}.png'.format(lenplot), bbox_inches='tight')
    return hist, total1


def plot_traj(xts, yts, total, interv, T2plot):
    """
    This function plots the trajectories of the particles as they were originally caught on the
    microscope (no centering performed).

    Inputs:
    xts: dictionary of arrays.  Keys are the particle numbers.
    yts: dictionary of arrays.  Keys are the particle numbers.
    total: total particles in dictionary.
    """
    middle_max_x = np.zeros(total)
    middle_max_y = np.zeros(total)

    for num in range(1, total+1):
        middle_max_x[num-1] = np.nanmax(xts[num])
        middle_max_y[num-1] = np.nanmax(yts[num])

    x_max = np.round(max(middle_max_x)/10, 0)*10+0.1
    y_max = np.round(max(middle_max_y)/10, 0)*10+0.1

    both_max = max(x_max, y_max)

    # Creates figure
    fig = plt.figure(figsize=(24, 18), dpi=80)
    ax = fig.add_subplot(111)

    for num in range(1, total+1):
        ax.plot(xts[num], yts[num], linewidth=3, label='Particle {}'.format(num))

    # A few adjustments to prettify the graph
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(70)

    xmajor_ticks = np.arange(0, both_max, interv)
    ymajor_ticks = np.arange(0, both_max, interv)

    ax.set_xticks(xmajor_ticks)
    ax.set_yticks(ymajor_ticks)
    ax.title.set_fontsize(70)
    ax.set_xlabel(r'x ($\mu$m)', fontsize=95)
    ax.set_ylabel(r'y ($\mu$m)', fontsize=95)
    # ax.tick_params(direction='out', pad=16)
    # ax.legend(loc=(0.60, 0.76), prop={'size': 40})
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(0)))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(0)))

    # plt.yscale('log')
    # plt.xscale('log')
    plt.gca().set_xlim([0, both_max])
    plt.gca().set_ylim([0, both_max])

    # Save your figure
    plt.savefig('{}.png'.format(T2plot), bbox_inches='tight')


def filter_out_short_traj(particles_unfiltered, framed_unfiltered, x_data_unfiltered, y_data_unfiltered, cut):
    """
    Filtered out particles trajectories from a dataset that are shorter than the variable cut.

    Inputs:
    particles_unfiltered
    framed_unfiltered
    x_data_unfiltered
    y_data_unfiltered
    cut: integer.

    Outputs:
    particles_filtered
    framed_filtered
    x_data_filtered
    y_data_filtered
    """
    total_unfiltered = int(max(particles_unfiltered))

    counter = 0

    for num in range(1, total_unfiltered+1):

        hold = np.where(particles_unfiltered == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)

        if max1 - min1 + 1 < cut:
            counter = counter + max1 - min1 + 1

    new_shape = particles_unfiltered.shape[0] - counter
    particles_filtered = np.zeros(new_shape)
    framed_filtered = np.zeros(new_shape)
    x_data_filtered = np.zeros(new_shape)
    y_data_filtered = np.zeros(new_shape)

    counter = 0
    counter2 = 0

    for num in range(1, total_unfiltered+1):

        hold = np.where(particles_unfiltered == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)

        if max1 - min1 + 1 >= cut:
            particles_filtered[min1 - counter:max1+1 - counter] = particles_unfiltered[min1:max1+1] - counter2
            framed_filtered[min1 - counter:max1+1 - counter] = framed_unfiltered[min1:max1+1]
            x_data_filtered[min1 - counter:max1+1 - counter] = x_data_unfiltered[min1:max1+1]
            y_data_filtered[min1 - counter:max1+1 - counter] = y_data_unfiltered[min1:max1+1]
        else:
            counter = counter + max1 - min1 + 1
            counter2 = counter2 + 1

    return particles_filtered, framed_filtered, x_data_filtered, y_data_filtered


def plot_trajectory_overlay(x, y, graph_size, ticks, number_of_trajectories, Tplot):
    """
    This function plots a random selection of trajectories from an x,y trajectory dataset. The user can
    manipulate size of the graph, tick interval size, and the number of trajectories to be plotted.
    """
    unit = graph_size

    maxx = unit + 0.1
    minx = -unit - 0.1
    maxy = unit + 0.1
    miny = -unit - 0.1

    random_particles = np.zeros(number_of_trajectories)
    for num in range(0, number_of_trajectories):
        random_particles[num] = np.random.random_integers(1, max(x))

    dec = 0
    xc = dict()
    yc = dict()
    xcmask = dict()

    # Creates figure
    fig1 = plt.figure(figsize=(24, 18), dpi=80)
    ax1 = fig1.add_subplot(111)

    for num in range(1, random_particles.shape[0]+1):
        lowx = ma.min(x[random_particles[num-1]])
        highx = ma.max(x[random_particles[num-1]])
        lowy = ma.min(y[random_particles[num-1]])
        highy = ma.max(y[random_particles[num-1]])

        xc[random_particles[num-1]] = np.array([x[random_particles[num-1]] - ((highx+lowx)/2)])
        yc[random_particles[num-1]] = np.array([y[random_particles[num-1]] - ((highy+lowy)/2)])

        xcmask[random_particles[num-1]] = x[random_particles[num-1]].recordmask
        xc[random_particles[num-1]] = ma.array(xc[random_particles[num-1]], mask=xcmask[random_particles[num-1]])
        yc[random_particles[num-1]] = ma.array(yc[random_particles[num-1]], mask=xcmask[random_particles[num-1]])

        ax1.plot(xc[random_particles[num-1]][0, :], yc[random_particles[num-1]][0, :],
                 linewidth=10, label='Particle {}'.format(random_particles[num-1]))

    # A few adjustments to prettify the graph
    for item in ([ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(70)

    xmajor_ticks = np.arange(minx, maxx, ticks)
    ymajor_ticks = np.arange(miny, maxy, ticks)

    ax1.set_xticks(xmajor_ticks)
    ax1.set_yticks(ymajor_ticks)
    ax1.title.set_fontsize(70)
    ax1.set_xlabel(r'x ($\mu$m)', fontsize=95)
    ax1.set_ylabel(r'y ($\mu$m)', fontsize=95)
    ax1.tick_params(direction='out', pad=16)
    plt.xticks(rotation=-30)
    # ax1.legend(loc=(0.60, 0.46), prop={'size': 40})
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))

    # plt.yscale('log')
    # plt.xscale('log')
    plt.gca().set_xlim([minx, maxx])
    plt.gca().set_ylim([miny, maxy])

    # Save your figure
    plt.savefig('{}.png'.format(Tplot), bbox_inches='tight')


def quality_control(path2, folder, frames, conversion, parameters, interv, cut):
    """
    This function plots a histogram of trajectory lengths (in units of frames) and two types of plots of
    trajectories (original and overlay).

    Inputs:
    path2: string.  Name of input trajectory csv files.
    folder: string. Name of folder to which to save results.
    frames: integer.  Total number of frames in videos.
    conversion: array.  Contains microns per pixel and frames per second of videos.
    parameters:

    parameters = {}
    parameters["channels"] = ["RED"]
    parameters["genotypes"] = ["WT"]
    parameters["pups"] = ["P1", "P2", "P3"]
    parameters["surface functionalities"] = ["PEG"]
    parameters["slices"] = ["S1", "S2", "S3"]
    parameters["regions"] = ["cortex", "mid"]
    parameters["replicates"] = [1, 2, 3, 4]

    cut: integer.  Minimum number of frames a trajectory must have in order to be plotted.
    """

    channels = parameters["channels"]
    genotypes = parameters["genotypes"]
    pups = parameters["pups"]
    surface_functionalities = parameters["surface functionalities"]
    slices = parameters["slices"]
    regions = parameters["regions"]
    replicates = parameters["replicates"]
    SD_frames = [1, 7, 14, 15]

    trajectory = {}
    names_with_replicates = {}
    data = {}

    particles_unfiltered = {}
    framed_unfiltered = {}
    x_data_unfiltered = {}
    y_data_unfiltered = {}
    total_unfiltered = {}
    particles = {}
    framed = {}
    x_data = {}
    y_data = {}
    total = {}
    tlength = {}
    x_microns = {}
    y_microns = {}
    x_particle = {}
    y_particle = {}

    x_original_frames = {}
    y_original_frames = {}

    x_adjusted_frames = {}
    y_adjusted_frames = {}

    counter2 = 0

    for channel in channels:
        for genotype in genotypes:
            for surface_functionality in surface_functionalities:
                for region in regions:
                    for pup in pups:
                        for slic in slices:
                            for replicate in replicates:

                                # Establishing sample names and extracting data from csv files.
                                counter2 = counter2 + 1
                                sample_name_long = "{}_{}_{}_{}_{}_{}_{}".format(channel, genotype, surface_functionality,
                                                                                 region, pup, slic, replicate)
                                names_with_replicates[counter2] = sample_name_long

                                filename = path2.format(channel=channel, genotype=genotype, pup=pup, region=region, sample_name=sample_name_long)
                                data[sample_name_long] = np.genfromtxt(filename, delimiter=",")
                                data[sample_name_long] = np.delete(data[sample_name_long], 0, 1)

                                # Names of output plots
                                logplot = folder.format(channel=channel, genotype=genotype,
                                                        pup=pup, region=region)+'{}_logplot'.format(sample_name_long)
                                Mplot = folder.format(channel=channel, genotype=genotype, pup=pup, region=region)+'{}_Mplot'.format(sample_name_long)
                                Dplot = folder.format(channel=channel, genotype=genotype, pup=pup, region=region)+'{}_Dplot'.format(sample_name_long)
                                Hplot = folder.format(channel=channel, genotype=genotype, pup=pup, region=region)+'{}_Hplot'.format(sample_name_long)
                                Hlogplot = folder.format(channel=channel, genotype=genotype,
                                                         pup=pup, region=region)+'{}_Hlogplot'.format(sample_name_long)
                                Cplot = folder.format(channel=channel, genotype=genotype,
                                                      pup=pup, region=region)+'{}_Cplot'.format(sample_name_long)
                                Tplot = folder.format(channel=channel, genotype=genotype,
                                                      pup=pup, region=region)+'{}_Tplot'.format(sample_name_long)
                                T2plot = folder.format(channel=channel, genotype=genotype,
                                                       pup=pup, region=region)+'{}_T2plot'.format(sample_name_long)
                                lenplot = folder.format(channel=channel, genotype=genotype,
                                                        pup=pup, region=region)+'{}_lenplot'.format(sample_name_long)

                                # Fill in data and split into individual trajectories
                                particles_unfiltered[counter2], framed_unfiltered[counter2], x_data_unfiltered[counter2],\
                                    y_data_unfiltered[counter2] = fill_in_and_split(data[names_with_replicates[counter2]])

                                total_unfiltered[counter2] = int(max(particles_unfiltered[counter2]))

                                # Filter out short trajectories
                                particles[counter2], framed[counter2], x_data[counter2], y_data[counter2] =\
                                    filter_out_short_traj(particles_unfiltered[counter2], framed_unfiltered[counter2],
                                                          x_data_unfiltered[counter2], y_data_unfiltered[counter2], cut)

                                # Convert to microns and seconds
                                time, time_SD = build_time_array(frames, conversion, SD_frames)
                                framen = np.linspace(0, frames, frames+1).astype(np.int64)
                                total[counter2] = int(max(particles[counter2]))
                                tlength[counter2] = np.zeros(total[counter2])

                                x_microns[counter2] = x_data[counter2]*conversion[0]
                                y_microns[counter2] = y_data[counter2]*conversion[0]

                                # Adjust frames (probably unneccesary, but I did it...)
                                x_particle[counter2] = {}
                                y_particle[counter2] = {}

                                x_original_frames[counter2] = {}
                                y_original_frames[counter2] = {}

                                x_adjusted_frames[counter2] = {}
                                y_adjusted_frames[counter2] = {}

                                for num in range(1, total[counter2] + 1):
                                    hold = np.where(particles[counter2] == num)
                                    itindex = hold[0]
                                    min1 = min(itindex)
                                    max1 = max(itindex)
                                    x_particle[counter2][num] = x_microns[counter2][min1:max1+1]
                                    y_particle[counter2][num] = y_microns[counter2][min1:max1+1]

                                    x_original_frames[counter2][num] = np.zeros(frames + 1)
                                    y_original_frames[counter2][num] = np.zeros(frames + 1)
                                    x_adjusted_frames[counter2][num] = np.zeros(frames + 1)
                                    y_adjusted_frames[counter2][num] = np.zeros(frames + 1)

                                    x_original_frames[counter2][num][framed[counter2][min1]:framed[counter2][max1]+1]\
                                        = x_microns[counter2][min1:max1+1]
                                    y_original_frames[counter2][num][framed[counter2][min1]:framed[counter2][max1]+1]\
                                        = y_microns[counter2][min1:max1+1]

                                    x_adjusted_frames[counter2][num][0:max1+1-min1] = x_microns[counter2][min1:max1+1]
                                    y_adjusted_frames[counter2][num][0:max1+1-min1] = y_microns[counter2][min1:max1+1]

                                    x_original_frames[counter2][num] = ma.masked_equal(x_original_frames[counter2][num], 0)
                                    y_original_frames[counter2][num] = ma.masked_equal(y_original_frames[counter2][num], 0)
                                    x_adjusted_frames[counter2][num] = ma.masked_equal(x_adjusted_frames[counter2][num], 0)
                                    y_adjusted_frames[counter2][num] = ma.masked_equal(y_adjusted_frames[counter2][num], 0)

                                    tlength[counter2][num - 1] = ma.count(x_adjusted_frames[counter2][num])

                                plot_traj(x_original_frames[counter2], y_original_frames[counter2], total[counter2], interv, T2plot)
                                plt.gcf().clear()
                                plot_trajectory_overlay(x_original_frames[counter2], y_original_frames[counter2], 6, 2, 6, Tplot)
                                plt.gcf().clear()
                                plot_traj_length_histogram(tlength[counter2], max(tlength[counter2]), lenplot)
                                plt.gcf().clear()


def diffusion_coefficient_point_derivative(MSDs, time, time_to_calculate, dimension):
    """
    This function calculates the diffusion coefficient from a mean squared displacement dataset.

    Inputs:
    MSDs: array.
    time: array.  Must be one unit longer than the MSDs array.
    time_to_calculate: float.  Time at which the diffusion coefficient is to be calculated.
    dimension: string, either "1D", "2D", or "3D."

    Outputs:
    diffusion_coef_point_derivative: float.
    """

    if dimension == "1D":
        dimension_coefficient = 2
    elif dimension == "2D":
        dimension_coefficient = 4
    elif dimension == "3D":
        dimension_coefficient = 6
    else:
        print("Error: dimension must be string 1D, 2D, or 3D.")

    diffusion_coefficients = MSDs/(dimension_coefficient*time[:-1])

    def find_nearest(array, value):
        idx = (np.abs(array-value)).argmin()
        return array[idx], idx

    time_close, time_at_diffusion_calc = find_nearest(time, time_to_calculate)
    diffusion_coef_point_derivative = diffusion_coefficients[time_at_diffusion_calc]

    return diffusion_coef_point_derivative


def diffusion_coefficient_linear_regression(MSDs, time, to_frame, dimension):
    """
    This function calculates the diffusion coefficient from a mean squared displacement dataset
    by using linear regression.

    Inputs:
    MSDs: array.
    time: array.  Must be one unit longer than the MSDs array.
    to_frame: frame at which to cut MSD data for linear regression calculation.
    dimension: string, either "1D", "2D", or "3D."

    Outputs:
    diffusion_coef_point_linear_fit.

    Example:

    grouped_variables = ["RED", "YG", "B"]
    subgrouped_variables = ["P1", "P2", "P3", "P4"]

    data_to_graph = {"RED_P1": 0.5, "RED_P2": 1, "RED_P3": 1.5, "YG_P1": 1, "YG_P2": 1, "YG_P3": 1, "B_P1":
    0.2, "B_P2": 5, "B_P3": 0.1, "RED_P4": 0.3, "YG_P4": 1.8, "B_P4": 3.6}
    stds_to_graph = {"RED_P1": 0.1, "RED_P2": 0.05, "RED_P3": 0.1, "YG_P1": 0.05, "YG_P2": 0.1, "YG_P3": 0.05,
    "B_P1": 0.5, "B_P2": 0.5, "B_P3": 0.5, "RED_P4": 0.3, "YG_P4": 0.5, "B_P4": 0.6}

    sample_size = 3

    filename = "test.png"
    legend_location = "top_left"
    """

    if dimension == "1D":
        dimension_coefficient = 2
    elif dimension == "2D":
        dimension_coefficient = 4
    elif dimension == "3D":
        dimension_coefficient = 6
    else:
        print("Error: dimension must be string 1D, 2D, or 3D.")

    fit_coefficients = dict()
    residuals = dict()
    line = dict()
    A1 = np.ones((np.shape(time[:-1])[0], 2))
    A1[:, 0] = time[:-1]

    fit_coefficients[0], residuals[0] = np.linalg.lstsq(A1[1:to_frame, :], MSDs[1:to_frame])[0:2]

    line[0] = fit_coefficients[0][0]*time[:-1] + fit_coefficients[0][1]

    diffusion_coefficient_linear_fit = fit_coefficients[0][0]/dimension_coefficient
    return diffusion_coefficient_linear_fit


def calculate_diffusion_coefficients(channels, genotypes, pups, surface_functionalities, slices, regions, replicates, path,
                                     time, time_to_calculate, to_frame, dimension):

    """
    Loads data from csv files and outputs a dictionary following a specified
        sample naming convection determined by the input

    Parameters:
    channels, surface functionalities, media, and concentrations, and replicates
        can take ranges or lists.
    path is string with substition placeholders for concentration and sample
        name (built from channels, surface_functionalities, media,
        concentrations, and replicates).

    Example:
    path = "./{genotype}/{pup}/{region}/{channel}/geoM2xy_{sample_name}.csv";
    get_data(["RED", "YG"], ["WT", "KO", "HET"], ["P1", "P2", "P3", "P4"],
    ["PEG", "noPEG"], ["S1", "S2", "S3", "S4"], ["cortex", "hipp", "mid"],
    [1, 2, 3, 4, 5], path)
    """

    data = {}
    avg_over_slices_raw = {}
    avg_over_pups_raw = {}
    names_with_replicates = {}
    counter = 0
    counter2 = 0

    diffusion_coef_point_derivative = {}
    diffusion_coef_linear_fit = {}

    for channel in channels:
        for genotype in genotypes:
            for surface_functionality in surface_functionalities:
                for region in regions:
                    for pup in pups:
                        for slic in slices:
                            test_value = "{}_{}_{}_{}_{}".format(channel, genotype, surface_functionality, region, pup)
                            avg_over_slices_raw[counter] = test_value
                            test_value2 = "{}_{}_{}_{}".format(channel, genotype, surface_functionality, region)
                            avg_over_pups_raw[counter] = test_value2
                            counter = counter + 1
                            sample_name = "{}_{}_{}_{}_{}_{}".format(channel, genotype, surface_functionality, region, pup, slic)
                            for replicate in replicates:
                                sample_name_long = "{}_{}_{}_{}_{}_{}_{}".format(channel, genotype, surface_functionality,
                                                                                 region, pup, slic, replicate)
                                names_with_replicates[counter2] = sample_name_long
                                counter2 = counter2 + 1
                            filename = path.format(channel=channel, genotype=genotype, pup=pup, region=region, sample_name=sample_name)
                            data[sample_name] = np.genfromtxt(filename, delimiter=",")

                            diffusion_coef_point_derivative[sample_name] =\
                                diffusion_coefficient_point_derivative(data[sample_name], time, time_to_calculate, dimension)
                            diffusion_coef_linear_fit[sample_name] =\
                                diffusion_coefficient_linear_regression(data[sample_name], time, to_frame, dimension)

    avg_over_slices = {}
    avg_over_pups = {}

    counter = 0
    for key, value in avg_over_slices_raw.items():
        if value not in avg_over_slices.values():
            avg_over_slices[counter] = value
            counter = counter + 1

    counter = 0
    for key, value in avg_over_pups_raw.items():
        if value not in avg_over_pups.values():
            avg_over_pups[counter] = value
            counter = counter + 1

    p_derivative = {}
    lin_fit = {}

    p_derivative["average_over_slices"] = {}
    p_derivative["average_over_pups"] = {}
    p_derivative["all_SD_over_slices"] = {}
    p_derivative["all_SD_over_pups"] = {}

    lin_fit["average_over_slices"] = {}
    lin_fit["average_over_pups"] = {}
    lin_fit["all_SD_over_slices"] = {}
    lin_fit["all_SD_over_pups"] = {}

    p_derivative["average_over_slices"] = avg_all(diffusion_coef_point_derivative, 2, avg_over_slices)
    p_derivative["average_over_pups"] = avg_all(p_derivative["average_over_slices"], 2, avg_over_pups)
    p_derivative["all_SD_over_slices"] = SD_all(diffusion_coef_point_derivative, 2, [], avg_over_slices)
    p_derivative["all_SD_over_pups"] = SD_all(p_derivative["average_over_slices"], 2, [], avg_over_pups)

    lin_fit["average_over_slices"] = avg_all(diffusion_coef_linear_fit, 2, avg_over_slices)
    lin_fit["average_over_pups"] = avg_all(lin_fit["average_over_slices"], 2, avg_over_pups)
    lin_fit["all_SD_over_slices"] = SD_all(diffusion_coef_linear_fit, 2, [], avg_over_slices)
    lin_fit["all_SD_over_pups"] = SD_all(lin_fit["average_over_slices"], 2, [], avg_over_pups)

    return p_derivative, lin_fit


def diffusion_bar_chart(grouped_variables, subgrouped_variables, data_to_graph, stds_to_graph, sample_size, filename, legend_position="top_left",
                        manual_y_axis=False, y_max=1.5):
    """
    This is a fairly comprehensive function for plotting diffusion bar charts complete with error bars.

    Inputs:

    grouped_variables: array.  Corresponds to variables that will appear on x axis.
    subgrouped_variables: array.  Corresponds to variables that will appear in legend.
    data_to_graph: dictionary.  Keys must contain both grouped_variables and subgrouped variables.
    stds_to_graph: dictionary.  Keys must contain both grouped_variables and sub_grouped variables.
    sample_size: scalar.  Number of samples used to find each datapoint.
    filename: string. Desired name of file with file extension.

    legend_position: string.  top_left, top_right, left_shifted_one, right_shifted_one.
    """

    if manual_y_axis:
        graph_max = y_max
    else:
        graph_max = max(data_to_graph.values())
    line_colors = ['gray', 'lightgray', 'brown', 'g', 'r', 'b', 'c', 'm', 'k']
    legend_size = 40
    width = 0.2
    N_groups = len(grouped_variables)
    N_subgroups = len(subgrouped_variables)

    legend_location = {"top_left": (0.0048*legend_size, 1), "top_right": (1, 1), "left_shifted_one": (0.0048*legend_size+width, 1),
                       "right_shifted_one": (1-width, 1)}
    bar_values = {}
    bar_stds = {}
    rects = {}
    error_kws = dict(elinewidth=10, ecolor='k', capsize=15, markeredgewidth=10)
    label_size = 70

    plt.rc('axes', linewidth=4)
    plt.rc('font', weight='bold')
    fig, ax = plt.subplots(figsize=(26, 18))

    ind = np.arange(N_groups)
    labels = N_groups*["None"]
    legendees = N_subgroups*["None"]
    legend_labels = N_subgroups*["None"]

    counter_subgroup = 0
    for subgroup in subgrouped_variables:
        bar_values[subgroup] = np.zeros(N_groups)
        bar_stds[subgroup] = np.zeros(N_groups)

        counter_group = 0
        for group in grouped_variables:

            for keys in data_to_graph:
                if subgroup in keys:
                    if group in keys:
                        bar_values[subgroup][counter_group] = data_to_graph[keys]
                        bar_stds[subgroup][counter_group] = stds_to_graph[keys]
            labels[counter_group] = group
            counter_group = counter_group + 1

        rects[subgroup] = ax.bar(ind + counter_subgroup*width - width/2, bar_values[subgroup], width, yerr=bar_stds[subgroup]/np.sqrt(sample_size),
                                 color=line_colors[counter_subgroup], linewidth=5, error_kw=error_kws)
        legendees[counter_subgroup] = rects[subgroup][0]
        legend_labels[counter_subgroup] = subgroup
        counter_subgroup = counter_subgroup + 1

    ax.set_ylabel(r'Diffusion Coefficients ($\mu$m$^2$/s)', size=label_size)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(labels)
    legend = ax.legend(legendees, legend_labels, prop={'size': legend_size}, bbox_to_anchor=legend_location[legend_position])
    legend.get_frame().set_linewidth(5)

    ax.tick_params(axis="both", labelsize="40", direction='out', pad=20)
    plt.xlim([min(ind)-0.4, max(ind)+0.75])
    plt.ylim([0, graph_max+0.1*graph_max])

    plt.savefig(filename, bbox_inches='tight')


def summary_barcharts(diffusion_dataset, parameters):
    """
    This uses the general function diffusion_bar_chart to plot specific output from
    calculate_diffusion_coefficients.  diffusion_dataset must be of the correct form
    specific to calculate_diffusion_coefficients.
    """

    relevant_dataset = diffusion_dataset["average_over_slices"]
    relevant_stds = diffusion_dataset["all_SD_over_slices"]
    examined_variable = parameters["channels"]

    data_to_graph = {}
    stds_to_graph = {}
    grouped_variables = parameters["regions"]
    subgrouped_variables = parameters["pups"]
    sample_size = len(parameters["slices"])
    graph_max = max(relevant_dataset.values())

    counter = 0
    for variable in examined_variable:
        data_to_graph[variable] = {}
        stds_to_graph[variable] = {}

        for keys in relevant_dataset:
            if variable in keys:
                data_to_graph[variable][keys] = relevant_dataset[keys]
                stds_to_graph[variable][keys] = relevant_stds[keys]

        filename = "diffusion_barchart_{}.png".format(variable)
        diffusion_bar_chart(grouped_variables, subgrouped_variables, data_to_graph[variable], stds_to_graph[variable], sample_size, filename,
                            manual_y_axis=True, y_max=graph_max, legend_position="top_right")
        counter = counter + 1

    outer_average = diffusion_dataset["average_over_pups"]
    outer_stds = diffusion_dataset["all_SD_over_pups"]
    sample_size2 = len(parameters["pups"])

    filename2 = "diffusion_barchart.png"
    diffusion_bar_chart(parameters["channels"], parameters["regions"], outer_average, outer_stds, sample_size2, filename2,
                        manual_y_axis=True, y_max=graph_max, legend_position="top_right")


def calculate_MMSDs(parameters, folder, size, if_localn_error=True):
    """
    Calculates the mean squared displacement based on the output of my Hyak code.
    In order to run, must have a csv files with prefixes "pM1x", "pM1y", and "pM2xy"
    in the called upon folders.

    Inputs:
    parameters: dictionary of strings.
    folder: string.  Identifies location of csv files to be used for MMSD calculations.
    size: float.  Equal to the number of nodes Hyak used to break up dataset.
    """

    channels = parameters["channels"]
    genotypes = parameters["genotypes"]
    pups = parameters["pups"]
    surface_functionalities = parameters["surface functionalities"]
    slices = parameters["slices"]
    regions = parameters["regions"]
    replicates = parameters["replicates"]

    SM1x = {}
    SM1y = {}
    SM2xy = {}

    for channel in channels:
        for genotype in genotypes:
            for surface_functionality in surface_functionalities:
                for region in regions:
                    for pup in pups:
                        slice_counter = 0
                        for slic in slices:

                            suffix = parameters["slice_suffixes"][slice_counter]  # Define slice suffixes (currently a, b, c)
                            with open(folder.format(genotype=genotype, pup=pup, region=region,
                                                    channel=channel)+'/pM1x'+suffix+'_0.csv', "rb") as f_handle:
                                interim = np.genfromtxt(f_handle, delimiter=",")  # interim defines local_n.

                            sample_name = "{}_{}_{}_{}_{}_{}".format(channel, genotype, surface_functionality, region, pup, slic)
                            # print("name is", sample_name)

                            # For some reason, I often get an error if local_n is exactly interim.shape. This avoids that error, if needed.
                            if if_localn_error:
                                local_n = interim.shape[1] - 1
                            else:
                                local_n = interim.shape[1]

                            SM1x[sample_name] = {}
                            SM1y[sample_name] = {}
                            SM2xy[sample_name] = {}

                            # This defines shifted (S) MSDs for each particle.
                            for num in range(0, size):
                                with open(folder.format(genotype=genotype, pup=pup, region=region,
                                          channel=channel)+'/pM1x'+suffix+'_{}.csv'.format(num), "rb") as f_handle:
                                    interim = np.genfromtxt(f_handle, delimiter=",")
                                for i in range(1, local_n+1):
                                    current = local_n*num + i
                                    SM1x[sample_name][current] = interim[:, i-1]
                                with open(folder.format(genotype=genotype, pup=pup, region=region,
                                          channel=channel)+'/pM1y'+suffix+'_{}.csv'.format(num), "rb") as f_handle:
                                    interim1 = np.genfromtxt(f_handle, delimiter=",")
                                for i in range(1, local_n+1):
                                    current = local_n*num + i
                                    SM1y[sample_name][current] = interim1[:, i-1]
                                with open(folder.format(genotype=genotype, pup=pup, region=region,
                                          channel=channel)+'/pM2xy'+suffix+'_{}.csv'.format(num), "rb") as f_handle:
                                    interim2 = np.genfromtxt(f_handle, delimiter=",")
                                for i in range(1, local_n+1):
                                    current = local_n*num + i
                                    SM2xy[sample_name][current] = interim2[:, i-1]

                            total_check = local_n*size
                            tots = total_check

                            arM1x = np.zeros(SM1x[sample_name][1].shape[0])
                            arM1y = np.zeros(SM1x[sample_name][1].shape[0])
                            arM2xy = np.zeros(SM1x[sample_name][1].shape[0])

                            st_arM1x = np.zeros(SM1x[sample_name][1].shape[0])
                            st_arM1y = np.zeros(SM1x[sample_name][1].shape[0])
                            st_arM2xy = np.zeros(SM1x[sample_name][1].shape[0])

                            gM1x = dict()
                            gM1y = dict()
                            gM2xy = dict()

                            log_gM1x = dict()
                            log_gM1y = dict()
                            log_gM2xy = dict()

                            geoM1x = np.zeros(SM1x[sample_name][1].shape[0])
                            geoM1y = np.zeros(SM1x[sample_name][1].shape[0])
                            geoM2xy = np.zeros(SM1x[sample_name][1].shape[0])

                            st_geoM1x = np.zeros(SM1x[sample_name][1].shape[0])
                            st_geoM1y = np.zeros(SM1x[sample_name][1].shape[0])
                            st_geoM2xy = np.zeros(SM1x[sample_name][1].shape[0])

                            # Calculating geometric and arithmetic means
                            for num2 in range(0, SM1x[sample_name][1].shape[0]):
                                gM1x[num2+1] = np.zeros(tots)
                                gM1y[num2+1] = np.zeros(tots)
                                gM2xy[num2+1] = np.zeros(tots)

                                for num in range(1, tots+1):
                                    gM1x[num2+1][num-1] = SM1x[sample_name][num][num2]
                                    gM1y[num2+1][num-1] = SM1y[sample_name][num][num2]
                                    gM2xy[num2+1][num-1] = SM2xy[sample_name][num][num2]

                                gM1x[num2+1] = ma.masked_invalid(gM1x[num2+1])
                                gM1y[num2+1] = ma.masked_invalid(gM1y[num2+1])
                                gM2xy[num2+1] = ma.masked_invalid(gM2xy[num2+1])

                                gM1x[num2+1] = ma.masked_equal(gM1x[num2+1], 0)
                                gM1y[num2+1] = ma.masked_equal(gM1y[num2+1], 0)
                                gM2xy[num2+1] = ma.masked_equal(gM2xy[num2+1], 0)

                                log_gM1x[num2+1] = np.log(gM1x[num2+1])
                                log_gM1y[num2+1] = np.log(gM1y[num2+1])
                                log_gM2xy[num2+1] = np.log(gM2xy[num2+1])

                                geoM1x[num2] = stat.gmean(gM1x[num2+1])
                                geoM1y[num2] = stat.gmean(gM1y[num2+1])
                                geoM2xy[num2] = stat.gmean(gM2xy[num2+1])

                                st_geoM1x[num2] = np.abs(geoM1x[num2]-np.exp(np.mean(
                                    np.log(gM1x[num2+1]))-np.std(np.log(gM1x[num2+1]))/np.sqrt(gM1x[num2+1].shape[0])))
                                st_geoM1y[num2] = np.abs(geoM1y[num2]-np.exp(np.mean(
                                    np.log(gM1y[num2+1]))-np.std(np.log(gM1y[num2+1]))/np.sqrt(gM1y[num2+1].shape[0])))
                                st_geoM2xy[num2] = np.abs(geoM2xy[num2]-np.exp(np.mean(np.log(
                                    gM2xy[num2+1]))-np.std(np.log(gM2xy[num2+1]))/np.sqrt(gM2xy[num2+1].shape[0])))

                                arM1x[num2] = np.mean(gM1x[num2+1])
                                arM1y[num2] = np.mean(gM1y[num2+1])
                                arM2xy[num2] = np.mean(gM2xy[num2+1])

                                st_arM1x[num2] = np.std(gM1x[num2+1])
                                st_arM1y[num2] = np.std(gM1y[num2+1])
                                st_arM2xy[num2] = np.std(gM2xy[num2+1])

                            # Saves the geometric and arithmetic mean data.
                            np.savetxt(folder.format(genotype=genotype, pup=pup, region=region,
                                                     channel=channel)+'geoM2xy_{}.csv'.format(sample_name), geoM2xy, delimiter=',')
                            np.savetxt(folder.format(genotype=genotype, pup=pup, region=region,
                                                     channel=channel)+'arM2xy_{}.csv'.format(sample_name), arM2xy, delimiter=',')

                            np.savetxt(folder.format(genotype=genotype, pup=pup, region=region,
                                                     channel=channel)+'geoM1x_{}.csv'.format(sample_name), geoM1x, delimiter=',')
                            np.savetxt(folder.format(genotype=genotype, pup=pup, region=region,
                                                     channel=channel)+'arM1x_{}.csv'.format(sample_name), arM1x, delimiter=',')
                            np.savetxt(folder.format(genotype=genotype, pup=pup, region=region,
                                                     channel=channel)+'geoM1y_{}.csv'.format(sample_name), geoM1y, delimiter=',')
                            np.savetxt(folder.format(genotype=genotype, pup=pup, region=region,
                                                     channel=channel)+'arM1y_{}.csv'.format(sample_name), arM1y, delimiter=',')

                            slice_counter = slice_counter + 1

    return SM1x, SM1y, SM2xy


def plot_general_histogram(dataset, bins, label, filename, set_y_limit=False, y_range=40, set_x_limit=False, x_range=40):
    """
    This function plots a histogram of the input dataset.

    Inputs:
    dataset: array of floats.  Contains data used to generate histogram.
    bins: integer or array. desired number of bins.
    label: string. label used along x axis.
    filename: string.  desired name of file.  File will automatically be saved as a PNG.
    set_y_limit: True/False.  Option to manually set the y limit of graph.
    y_range: float.  Manually set y limit.
    set_x_limit: True/False.  Option to manually set the x limit of graph.
    x_range: float. Manually set x limit.
    """

    plot, bins = np.histogram(dataset, bins=bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:])/2
    plt.bar(center, plot, align='center', width=width)
    plt.xlabel(label, fontsize=20)

    if set_y_limit:
        plt.gca().set_ylim([0, y_range])

    if set_x_limit:
        plt.gca().set_xlim([0, x_range])

    plt.savefig('{}.png'.format(filename), bbox_inches='tight')


def plot_MSD_histogram(MSD_dataset, time, bins, filename, desired_time, diffusion_data=False, dimension="2D", set_y_limit=False, y_range=40,
                       set_x_limit=False, x_range=40):
    """
    This function plots a histogram of a mean squared displacement dataset. The MSD data
    can be converted to a diffusion dataset by using the diffusion_data option.

    Inputs:
    MSD_dataset: dictionary of arrays of floats.  Contains MSD data with keys corresponding to particle numbers.
    time: array of floats.  Contains corresponding time data. Must be same for all particles in MSD dictionary.
        Must also be one unit longer than MSD datasets.
    bins: integer or array. desired number of bins.
    filename: string.  desired name of file.  File will automatically be saved as a PNG.
    desired_time: float.  Time at which to measure MSDs for histogram.
    diffusion_data: True/False.  If true, will plot diffusion data instead of MSD data.
    dimension: string.  1D, 2D, or 3D.
    set_y_limit: True/False.  Option to manually set the y limit of graph.
    y_range: float.  Manually set y limit.
    set_x_limit: True/False.  Option to manually set the x limit of graph.
    x_range: float. Manually set x limit.
    """

    def find_nearest(array, value):
        idx = (np.abs(array-value)).argmin()
        return array[idx], idx

    td, idx = find_nearest(time, desired_time)

    if type(MSD_dataset) == type({}):
        total = len(MSD_dataset)

        if diffusion_data:
            label = r'Deffs ($\mu$m$^2$/s) at $\tau$ = {}s'.format(desired_time)

            if dimension is "1D":
                dimension_factor = 2
            elif dimension is "2D":
                dimension_factor = 4
            else:
                dimension_factor = 6

            dataset = {}
            for num in range(1, total+1):
                dataset[num] = MSD_dataset[num]/(dimension_factor*time[1:])

        else:
            label = r'MSDs ($\mu$m$^2$) at $\tau$ = {}s'.format(desired_time)
            dataset = MSD_dataset

        td, idx = find_nearest(time[1:], desired_time)

        hist = np.zeros(total)
        for num in range(1, total+1):
            hist[num-1] = dataset[num][idx]

        hist = [x for x in hist if str(x) != 'nan']

        plot_general_histogram(hist, bins, label, filename, set_y_limit=set_y_limit, y_range=y_range, set_x_limit=set_x_limit, x_range=x_range)

    else:
        total = MSD_dataset.shape[1]

        if diffusion_data:
            label = r'Deffs ($\mu$m$^2$/s) at $\tau$ = {}s'.format(desired_time)

            if dimension is "1D":
                dimension_factor = 2
            elif dimension is "2D":
                dimension_factor = 4
            else:
                dimension_factor = 6

            dataset = np.zeros((frames.shape[0], total))
            for num in range(0, total):
                dataset[:, num] = MSD_dataset[:, num]/(dimension_factor*time[1:])

        else:
            label = r'MSDs ($\mu$m$^2$) at $\tau$ = {}s'.format(desired_time)
            dataset = MSD_dataset

        hist = np.zeros(total)
        for num in range(0, total):
            hist[num] = dataset[idx, num]

        hist = [x for x in hist if str(x) != 'nan']

        plot_general_histogram(hist, bins, label, filename, set_y_limit=set_y_limit, y_range=y_range, set_x_limit=set_x_limit, x_range=x_range)


def plot_all_MSD_histograms(parameters, folder, dataset, time, bins, desired_time, diffusion_data=False, dimension="2D",
                            set_y_limit=False, y_range=40, set_x_limit=False, x_range=40):
    """
    This function plots histograms for all datasets in an experiment.  The output from calculate_MMSDs
    is used as an input to this function.

    Inputs:
    parameters: dictionary of form:

    parameters = {}
    parameters["channels"] = ["RED", "YG"]
    parameters["genotypes"] = ["KO"]
    parameters["pups"] = ["P1", "P2", "P3"]
    parameters["surface functionalities"] = ["PEG"]
    parameters["slices"] = ["S1", "S2", "S3"]
    parameters["regions"] = ["cortex", "mid"]
    parameters["replicates"] = [1, 2, 3, 4, 5]

    dataset: dictionary of dictionaries of arrays of floats.  Contains MSD data with keys corresponding to
        (1) sample names and (2) particle numbers. Use output from calculate_MMSDs preferably.
    time: array of floats.  Contains corresponding time data. Must be same for all particles in MSD dictionary.
        Must also be one unit longer than MSD datasets.
    bins: integer or array. desired number of bins.
    filename: string.  desired name of file.  File will automatically be saved as a PNG.
    desired_time: float.  Time at which to measure MSDs for histogram.
    diffusion_data: True/False.  If true, will plot diffusion data instead of MSD data.
    dimension: string.  1D, 2D, or 3D.
    set_y_limit: True/False.  Option to manually set the y limit of graph.
    y_range: float.  Manually set y limit.
    set_x_limit: True/False.  Option to manually set the x limit of graph.
    x_range: float. Manually set x limit.
    """

    channels = parameters["channels"]
    genotypes = parameters["genotypes"]
    pups = parameters["pups"]
    surface_functionalities = parameters["surface functionalities"]
    slices = parameters["slices"]
    regions = parameters["regions"]
    replicates = parameters["replicates"]

    for channel in channels:
        for genotype in genotypes:
            for surface_functionality in surface_functionalities:
                for region in regions:
                    for pup in pups:
                        slice_counter = 0
                        for slic in slices:

                            sample_name = "{}_{}_{}_{}_{}_{}".format(channel, genotype, surface_functionality, region, pup, slic)
                            # print("name is", sample_name)
                            Hplot = folder.format(channel=channel, genotype=genotype, pup=pup, region=region)+'{}_Hplot'.format(sample_name)

                            plt.gcf().clear()
                            plot_MSD_histogram(dataset[sample_name], time, bins, Hplot, desired_time, diffusion_data=diffusion_data,
                                               dimension=dimension, set_y_limit=set_y_limit, y_range=y_range,
                                               set_x_limit=set_x_limit, x_range=x_range)


def fillin2(data):
    """
    Fills in blanks of arrays without shifting frames by the starting frame.  Compare to fillin.

    Input: trajectory dataset from MOSAIC tracking software read into a numpy array
    Output: modified numpy array with missing frames filled in.
    """

    shap = int(max(data[:, 1])) + 1
    shape1 = int(min(data[:, 1]))
    newshap = shap - shape1
    filledin = np.zeros((newshap, 5))
    filledin[0, :] = data[0, :]

    count = 0
    new = 0
    other = 0
    tot = 0

    for num in range(1, newshap):
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


def MSD_iteration(folder, name, cut, totvids, conversion, frames):
    """
    Cleans up data for MSD analysis from csv files.  Outputs in form of
    dictionaries.
    """

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

    x = np.zeros((frames, total1 - 1))
    y = np.zeros((frames, total1 - 1))
    xs = np.zeros((frames, total1 - 1))
    ys = np.zeros((frames, total1 - 1))

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
    x_m = x[:, :total1-1]
    y_m = y[:, :total1-1]
    xs_m = xs[:, :total1-1]
    ys_m = ys[:, :total1-1]

    print('Total particles after merging datasets and filtering short trajectories:', total1)
    return total1, xs_m, ys_m, x_m, y_m


def vectorized_MMSD_calcs(frames, total1, xs_m, ys_m, x_m, y_m, frame_m):

    SM1x = np.zeros((frames, total1-1))
    SM1y = np.zeros((frames, total1-1))
    SM2xy = np.zeros((frames, total1-1))

    xs_m = ma.masked_equal(xs_m, 0)
    ys_m = ma.masked_equal(ys_m, 0)

    x_m = ma.masked_equal(x_m, 0)
    y_m = ma.masked_equal(y_m, 0)

    geoM1x = np.zeros(frame_m)
    geoM1y = np.zeros(frame_m)

    for frame in range(1, frame_m):
        bx = xs_m[frame, :]
        cx = xs_m[:-frame, :]
        Mx = (bx - cx)**2

        Mxa = np.mean(Mx, axis=0)
        Mxab = stat.gmean(Mxa, axis=0)

        geoM1x[frame] = Mxab

        by = ys_m[frame, :]
        cy = ys_m[:-frame, :]
        My = (by - cy)**2

        Mya = np.mean(My, axis=0)
        Myab = stat.gmean(Mya, axis=0)

        geoM1y[frame] = Myab
        SM1x[frame, :] = Mxa
        SM1y[frame, :] = Mya

    geoM2xy = geoM1x + geoM1y
    SM2xy = SM1x + SM1y

    return geoM1x, geoM1y, geoM2xy, SM1x, SM1y, SM2xy
