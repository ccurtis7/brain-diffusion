import numpy as np
import numpy.ma as ma
import scipy.stats as stat
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stat
from histogram_utils import histogram_by_video

def main():
    """
    Function that allows the user to graph histograms of all files in a folder
    from the command line.
    """
    script = sys.argv[0]
    for filename in sys.argv[1:]:
        histogram_by_video(filename)

main()
