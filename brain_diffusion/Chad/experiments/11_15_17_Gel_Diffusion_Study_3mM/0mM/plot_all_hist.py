import numpy as np
import numpy.ma as ma
import scipy.stats as stat
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stat

import sys
sys.path.insert(0, '/c/Users/koolk/Desktop/brain-diffusion/Chad_functions_and_unittests/11_15_17_Gel_Diffusion_Study_3mM/0mM/')

from histogram_utils import histogram_by_video

def main():
    script = sys.argv[0]
    for filename in sys.argv[1:]:
        histogram_by_video(filename)

main()