import numpy as np
import numpy.ma as ma
import scipy.stats as stat
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stat


def histogram_by_video(SMfilename, xlabel='Log Diffusion Coefficient Dist', ylabel='Trajectory Count', fps=100.02,
                       y_range=5000, frame_range=range(5, 30, 5), analysis='log', theta='D'):
    """
    Plots a histogram of mean squared displacements or diffusion coefficients from input data.

    Parameters
    ----------
    SMfilename : string
        Filename of particle MSDs.  Must be a csv file, comma delimited. Must be
        organized as frames x particles.
    xlabel : string
        X label of the output graph.
    ylabel : string
        Y label of the output graph.
    fps : float or int
        The frame rate of the video being analyzed.  Only required if graphing
        diffusion coefficients rather than mean squared displacements.
    y_range : int
        Y range of the output graph.
    frame_range : range
        Range containing which frames the user wishes to be plotted.
    analysis : string
        Desired type of data to be plotted.  If input is 'log', plots the natural
        logarithm of the data.  Any other input will plot the raw input data.
    theta : string
        Desired type of data to be plotted.  If input is 'D', plots diffusion
        coefficients.  Any other input will plot the mean squared displacements.

    Returns
    -------
    Returns 'Graph completed successfully' if function is successful.

    Examples
    --------
    >>> nframe = 51
    >>> npar = 1000
    >>> SMxy = np.zeros((nframe, npar))
    >>> for frame in range(0, nframe):
            SMxy[frame, :] = np.random.normal(loc=0.5*frame, scale=0.5, size=npar)
    >>> np.savetxt('sample_file.csv', SMxy, delimiter=',')
    >>> histogram_by_video('sample_file.csv', y_range=500, analysis="nlog", theta="MSD")
    >>> os.remove('sample_file.csv')
    >>> os.remove('sample_file_hist.png')
    """

    assert type(SMfilename) is str, "SMfilename must be a string"
    assert SMfilename.split('.')[1] == 'csv', "SMfilename must be a csv file."
    # assert os.path.isfile(SMfilename), "SMfilename must exist."
    assert type(np.genfromtxt(SMfilename, delimiter=",")) == np.ndarray, "SMfilename must be comma delimited."
    assert type(xlabel) is str, "xlabel must be a string"
    assert type(ylabel) is str, "ylabel must be a string"
    assert type(fps) is float or int, "fps must be float or int"
    assert type(y_range) is int, "y_range must be int"
    assert type(frame_range) is range, "frame_range must be a range"
    assert type(analysis) is str, "analysis must be string"
    assert type(theta) is str, "theta must be string"

    # load data
    SM2xy = np.genfromtxt(SMfilename, delimiter=",")

    # generate keys for legend
    bar = {}
    keys = []
    entries = []
    for i in range(0, 5):
        keys.append(i)
        entries.append(str(50*(i+1)) + 'ms')

    set_x_limit = False
    set_y_limit = True
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig = plt.figure(figsize=(16, 6))

    counter = 0
    for i in frame_range:
        toi = i/fps
        if theta == "MSD":
            factor = 1
        else:
            factor = 4*toi

        if analysis == 'log':
            dist = ma.log(SM2xy[i, :]/factor)
            test_bins = np.linspace(-5, 5, 76)
        else:
            dist = ma.masked_equal(SM2xy[i, :], 0)/factor
            test_bins = np.linspace(0, 20, 76)

        unmask = np.invert(ma.getmask(dist))
        dist = dist[unmask]
        histogram, test_bins = np.histogram(dist, bins=test_bins)

        # Plot_general_histogram_code
        avg = np.mean(dist)

        plt.rc('axes', linewidth=2)
        plot = histogram
        bins = test_bins
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:])/2
        bar[keys[counter]] = plt.bar(center, plot, align='center', width=width, color=colors[counter], label=entries[counter])
        plt.axvline(avg, color=colors[counter])
        plt.xlabel(xlabel, fontsize=30)
        plt.ylabel(ylabel, fontsize=30)
        plt.tick_params(axis='both', which='major', labelsize=20)

        counter = counter + 1
        if set_y_limit:
            plt.gca().set_ylim([0, y_range])

        if set_x_limit:
            plt.gca().set_xlim([0, x_range])

        plt.legend(fontsize=20, frameon=False)
    plt.savefig(SMfilename.split('.csv')[0]+'_hist.png', bbox_inches='tight')
    return 'Graph completed successfully'
