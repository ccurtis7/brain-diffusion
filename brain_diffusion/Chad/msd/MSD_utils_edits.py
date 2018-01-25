import numpy as np


def get_data_gels(channels, surface_functionalities, media, concentrations, replicates, path):

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
            for medium in media:
                for concentration in concentrations:
                    test_value = "{}_{}_{}_{}".format(channel, surface_functionality, medium, concentration)
                    avg_sets[counter] = test_value
                    counter = counter + 1
                    for replicate in replicates:
                        sample_name = "{}_{}_{}_{}_{}".format(channel, surface_functionality, medium, concentration, replicate)
                        filename = path.format(channel=channel, surface_functionality=surface_functionality, medium=medium,
                                               concentration=concentration, replicate=replicate, sample_name=sample_name)
                        data[sample_name] = np.genfromtxt(filename,
                                                          delimiter=",")

    return data, avg_sets


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


def return_average(data, frames, to_average='YG_nPEG_in_agarose_1x'):
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

    to_avg_num = np.zeros((frames, counter))
    for i in range(0, counter):
        for j in range(0, frames):
            to_avg_num[j, i] = data[to_avg[i]][j]

    answer = np.mean(to_avg_num, axis=1)

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
    avg_sets = {}
    counter = 0

    for channel in channels:
        for genotype in genotypes:
            for pup in pups:
                for surface_functionality in surface_functionalities:
                    for slic in slices:
                        for region in regions:
                            test_value = "{}_{}_{}_{}_{}_{}".format(channel, genotype, pup, surface_functionality, slic, region)
                            avg_sets[counter] = test_value
                            counter = counter + 1
                            for replicate in replicates:
                                sample_name = "{}_{}_{}_{}_{}_{}_{}".format(channel, genotype, pup, surface_functionality, slic, region, replicate)
                                filename = path.format(channel=channel, genotype=genotype, pup=pup, region=region, sample_name=sample_name)
                        data[sample_name] = np.genfromtxt(filename, delimiter=",")

    return data, avg_sets


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

    to_SD_num = np.zeros((frames, counter))
    for i in range(0, counter):
        for j in range(0, frames):
            to_SD_num[j, i] = data[to_SD[i]][j]

    answer = np.std(to_SD_num, axis=1)
    answer_sep = answer[SD_frames[:]]

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
