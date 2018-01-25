import numpy as np
import numpy.ma as ma
import scipy.stats as stat
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stat


def histogram_by_video(SMfilename, xlabel='Log Diffusion Coefficient Dist', ylabel='Trajectory Count', fps=100.02, frames=651,
                       y_range=5000, frame_range=range(5, 30, 5), analysis='log', theta='D'):

    """

    """
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
