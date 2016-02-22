"""
This file extracts the effective diffusion (Deff) and particle property (PP) data from the
zipfile and assigns them to variables. Size and zeta potential are then
categorized into designated increments, adding 2 columns to the particle
property data. The Deff and PP data are then indexed by NP type and joined
into one data set.
"""

import numpy as np
import pandas as pd
import zipfile

# Unzip the csv data and assign it to variables
zf = zipfile.ZipFile('brain-diffusion_data.zip')
file_handle1 = zf.open('Effective_Diffusion_1s_Data.csv')
deff = pd.read_csv(file_handle1)
file_handle2 = zf.open('Particle_Properties_Data.csv')
prop = pd.read_csv(file_handle2)

# Function with input of central size and tick range to create categories of
# particle size
def set_size_range(center,tick):
    low = center
    while low > min(prop['Size']):
        low = low-tick
    high = center
    while high < max(prop['Size']):
        high = high+tick
    rows = (high-low)/tick
    rows = int(rows)
    # The extra row and column of sizes_array are used as indices when the
    # array is converted to DataFrame
    sizes_array = np.zeros((rows+1,4))
    sizes = pd.DataFrame(sizes_array[1:,1:])
    sizes.columns = ['Low','High','Size_Range']
    for x in range(0,rows):
        sizes.loc[[x],['Low']] = low+tick*x
        sizes.loc[[x],['High']] = low+tick*(x+1)
        sizes.loc[[x],['Size_Range']] = '%i to %i' %
                (sizes['Low'][x], sizes['High'][x])
    return sizes

# Function with input of tick range to create categories of zeta potential from
# lowest value to zero
def set_zp_range(tick):
    low = 0
    while low > min(prop['Zeta_Potential']):
        low = low-tick
    rows = (0-low)/tick
    rows = int(rows)
    zp_array = np.zeros((rows+1,4))
    zp = pd.DataFrame(zp_array[1:,1:])
    zp.columns = ['Low','High','ZP_Range']
    for x in range(0,rows):
        zp.loc[[x],['Low']] = low+tick*x
        zp.loc[[x],['High']] = low+tick*(x+1)
        zp.loc[[x],['ZP_Range']] = '%i to %i' % (zp['Low'][x], zp['High'][x])
    return zp

# function assigning size and zeta potential categories to each particle
# property row that then merges Deff and PP to create a working data set
def working_data(size_center,size_tick,zp_tick):
    prop['Size_Range'] = 0
    prop['ZP_Range'] = 0
    size_range = set_size_range(size_center,size_tick)
    zp_range = set_zp_range(zp_tick)
    for x in range(0, len(prop)):
        for y in range(0, len(size_range)):
            if prop['Size'][x] >= size_range['Low'][y] and prop['Size'][x] <
                    size_range['High'][y]:
                prop.loc[[x],['Size_Range']] = size_range['Size_Range'][y]
                break
        for z in range(0,len(zp_range)):
            if prop['Zeta_Potential'][x] >= zp_range['Low'][z] and
            prop['Zeta_Potential'][x] < zp_range['High'][z]:
                prop.loc[[x],['ZP_Range']] = zp_range['ZP_Range'][z]
                break
    deff2 = deff.set_index('Particle')
    prop2 = prop.set_index('Sample')
    data = prop2.join(deff2)
    return data