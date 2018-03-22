import numpy as np
import numpy.ma as ma
import scipy.stats as stat
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stat
import numpy as np
import sys
import os
from brain_diffusion.histogram_utils import histogram_by_video
import pytest

is_travis = "CI" in os.environ.keys()


@pytest.mark.skipif(is_travis, reason="This doesn't work on Travis yet.")
def test_histogram_by_video():
    nframe = 51
    npar = 1000
    SMxy = np.zeros((nframe, npar))
    for frame in range(0, nframe):
        SMxy[frame, :] = np.random.normal(loc=0.5*frame, scale=0.5, size=npar)

    np.savetxt('sample_file.csv', SMxy, delimiter=',')
    histogram_by_video('sample_file.csv', y_range=500, analysis="nlog", theta="MSD")

    assert os.path.isfile('sample_file_hist.png'), "No plot was generated."

    os.remove('sample_file.csv')
    os.remove('sample_file_hist.png')
