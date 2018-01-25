"""
This file extracts the effective diffusion (Deff) and particle property (PP)
data from the zipfile and assigns them to variables. Size and zeta potential
are then categorized into designated increments, adding 2 columns to the
particle property data. The Deff and PP data are then indexed by NP type and
joined into one data set.
"""

import numpy as np
import pandas as pd
import zipfile
from ipywidgets import interact, interactive, Dropdown, widgets
from IPython.display import clear_output, display, HTML
from bokeh.charts import Histogram, output_notebook, show, defaults


# Unzip the csv data and assign it to variables
def obtain_data():
    zf = zipfile.ZipFile('brain-diffusion_data.zip')
    file_handle1 = zf.open('Effective_Diffusion_1s_Data.csv')
    deff = pd.read_csv(file_handle1)
    deff = deff.set_index('Particle')
    deff = deff.transpose()
    file_handle2 = zf.open('Particle_Properties_Data.csv')
    prop = pd.read_csv(file_handle2)
    return (deff, prop)


def set_size_range(center, tick):
    """
    Function with input of "central" and "tick" as float  to create categories
    of particle size.
    "center" designates the central measure of the NP in nanometers.
    "tick" designates the range in nanometers between two ticks of the x-axis.
    For example, "center: 100, tick: 10" sets 100 as the central measurement,
    and creates category divides at 90, 80, 70...110, 120, 130...etc.
    Function returns "sizes": a new dataframe with the low and high size limits
    of each tick range, and a third column of categories.
    For example, 'Low': 70, 'High': 80, 'Size_Range': '70 to 80'
    """
    deff, prop = obtain_data()
    if 'Size' in prop.columns:
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
        sizes_array = np.zeros((rows+1, 4))
        sizes = pd.DataFrame(sizes_array[1:, 1:])
        sizes.columns = ['Low', 'High', 'Size_Range']
        for x in range(0, rows):
            sizes.loc[[x], ['Low']] = low+tick*x
            sizes.loc[[x], ['High']] = low+tick*(x+1)
            sizes.loc[[x], ['Size_Range']] = '%i to %i' % (sizes['Low'][x],
                                                           sizes['High'][x])
        return sizes
    else:
        print('Please change your column describing size in your properties '
              'data set to "Size".')


def set_zp_range(tick):
    """
    Function with input of tick range to create categories of zeta potential
    from lowest value to zero. These range from some negative value to zero.
    "tick" designates the range in zeta potential (zp) between two ticks of the
    x-axis. For example, "tick: 2" creates category divides at 0, -2, -4, -6...
    etc. Function returns "zp": a new dataframe with the low and high zp limits
    of each tick range, and a third column of categories.
    For example, 'Low': -4, 'High': -2, 'ZP_Range': '-4 to -2'
    """
    deff, prop = obtain_data()
    if 'Zeta_Potential' in prop.columns:
        low = 0
        while low > min(prop['Zeta_Potential']):
            low = low-tick
        rows = (0-low)/tick
        rows = int(rows)
        zp_array = np.zeros((rows+1, 4))
        zp = pd.DataFrame(zp_array[1:, 1:])
        zp.columns = ['Low', 'High', 'ZP_Range']
        for x in range(0, rows):
            zp.loc[[x], ['Low']] = low+tick*x
            zp.loc[[x], ['High']] = low+tick*(x+1)
            zp.loc[[x], ['ZP_Range']] = '%i to %i' % (zp['Low'][x],
                                                      zp['High'][x])
        return zp
    else:
        print('Please change your column describing ZP in your properties data'
              ' set to "Zeta_Potential".')


def prop_data(size_center, size_tick, zp_tick):
    """
    This function assigns size and zeta potential categories to each particle
    property row by calling on set_zp_range and set_size_range.
    """
    deff, prop = obtain_data()
    prop['Size_Range'] = 0
    prop['ZP_Range'] = 0
    size_range = set_size_range(size_center, size_tick)
    zp_range = set_zp_range(zp_tick)
    for x in range(0, len(prop)):
        for y in range(0, len(size_range)):
            if (prop['Size'][x] >= size_range['Low'][y] and
                    prop['Size'][x] < size_range['High'][y]):
                prop.loc[[x], ['Size_Range']] = size_range['Size_Range'][y]
                break
        for z in range(0, len(zp_range)):
            if (prop['Zeta_Potential'][x] >= zp_range['Low'][z] and
                    prop['Zeta_Potential'][x] < zp_range['High'][z]):
                prop.loc[[x], ['ZP_Range']] = zp_range['ZP_Range'][z]
                break
    prop2 = prop.set_index('Sample')
    return prop2


def working_data(size_center, size_tick, zp_tick):
    """
    This function takes deff and converts it into new dataframes for each
    particle with every particle deff matched with the particle type in the
    next row. The particle dataframes are then appended together, and then the
    particle properties are added for each row.
    """
    deff, prop = obtain_data()
    p1 = pd.melt(deff, value_vars=['PLGA58k UP'])
    p2 = pd.melt(deff, value_vars=['PLGA58k P80'])
    p3 = pd.melt(deff, value_vars=['PLGA58k F68'])
    p4 = pd.melt(deff, value_vars=['PLGA58k 5CHA'])
    p5 = pd.melt(deff, value_vars=['PLGA15k UP'])
    p6 = pd.melt(deff, value_vars=['PLGA15k P80'])
    p7 = pd.melt(deff, value_vars=['PLGA15k F68'])
    p8 = pd.melt(deff, value_vars=['PLGA15k 5CHA'])
    p9 = pd.melt(deff, value_vars=['PLGA15k 2CHA'])
    p10 = pd.melt(deff, value_vars=['PLGA15k 0.5CHA'])
    p11 = pd.melt(deff, value_vars=['PEG-PLGA45k UP'])
    p12 = pd.melt(deff, value_vars=['PEG-PLGA45k P80'])
    p13 = pd.melt(deff, value_vars=['PEG-PLGA45k F68'])
    p14 = pd.melt(deff, value_vars=['PEG-PLGA45k 5CHA'])
    p15 = pd.melt(deff, value_vars=['PEG-PLGA15k UP'])
    p16 = pd.melt(deff, value_vars=['PEG-PLGA15k P80'])
    p17 = pd.melt(deff, value_vars=['PEG-PLGA15k F68'])
    p18 = pd.melt(deff, value_vars=['PEG-PLGA15k 5CHA'])
    p19 = pd.melt(deff, value_vars=['PEG-PLGA15k 2CHA'])
    p20 = pd.melt(deff, value_vars=['PEG-PLGA15k 0.5CHA'])
    p_all = p1.append([p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14,
                       p15, p16, p17, p18, p19, p20])
    properties = prop_data(size_center, size_tick, zp_tick)
    properties = properties.reset_index()
    headers = ['Particle_Type', 'Surfactant', 'PEG', 'Size_Range', 'ZP_Range']
    master = list()
    for i in range(0, 5):
        for j in range(0, len(properties)):
            holder = list([properties[headers[i]][j]]*484)
            master = master + holder
        masterseries = pd.Series(master)
        p_all[headers[i]] = pd.Series(master, index=p_all.index)
        master = list()
        holder = list()
    p_all.rename(columns={'value': 'Deff'}, inplace=True)
    p_all = p_all[p_all.Deff > 0]
    return p_all


def plot_Deff(bins_num, PEG, Particle_Type, Surfactant, Size_Range, ZP_Range):
    """
    This function plots the particles depending on the selected particle
    properties input. It has been customized to allow interactive input from
    the function below. The list of variables were created manually and would
    have to be changed for a different data set.
    The unittests for plot_Deff alone don't make sense, because it is only
    called upon with the interact function and UI input.
    """
    # Here size_center, size_tick, and zp_tick are set because these values
    # determine the Size_Range and ZP_Range categories. They must be manually
    # changed here, as well as in interact_plot_deff below.
    size_center = 100
    size_tick = 10
    zp_tick = 5
    data = working_data(size_center, size_tick, zp_tick)
    data2 = data
    sizes = set_size_range(size_center, size_tick)
    zp = set_zp_range(zp_tick)
    list_vars_Size_Range = list()
    list_vars_ZP_Range = list()
    # Here the lists for size range and zp range are taken from the dataframes
    # created from the set size and zp range functions.
    for x in range(0, len(sizes)):
        list_vars_Size_Range.insert(x, sizes['Size_Range'][x])
    list_vars_Size_Range.insert(0, 'All')
    for x in range(0, len(zp)):
        list_vars_ZP_Range.insert(x, zp['ZP_Range'][x])
    list_vars_ZP_Range.insert(0, 'All')
    list_vars_Surfactant = ['All', 'UP', 'P80', 'F68', '5CHA', '2CHA',
                            '0.5CHA']
    list_vars_Particle_Type = ['All', '58k', '45k', '15k']
    if PEG == 'No':
        data2 = data[data.PEG == 'No']
    if PEG == 'Yes':
        data2 = data[data.PEG == 'Yes']
    data = data2
    data2 = pd.DataFrame()
    for x in range(0, len(list_vars_Surfactant)):
        for y in range(0, len(Surfactant)):
            if Surfactant[y] == list_vars_Surfactant[x]:
                data2 = data2.append(data[data.Surfactant == Surfactant[y]])
    if Surfactant[0] == 'All':
        data2 = data
    data = data2
    data2 = pd.DataFrame()
    for x in range(0, len(list_vars_Particle_Type)):
        for y in range(0, len(Particle_Type)):
            if Particle_Type[y] == list_vars_Particle_Type[x]:
                data2 = data2.append(data[data.Particle_Type ==
                                          Particle_Type[y]])
    if Particle_Type[0] == 'All':
        data2 = data
    data = data2
    data2 = pd.DataFrame()
    for x in range(0, len(list_vars_Size_Range)):
        for y in range(0, len(Size_Range)):
            if Size_Range[y] == list_vars_Size_Range[x]:
                data2 = data2.append(data[data.Size_Range == Size_Range[y]])
    if Size_Range[0] == 'All':
        data2 = data
    data = data2
    data2 = pd.DataFrame()
    for x in range(0, len(list_vars_ZP_Range)):
        for y in range(0, len(ZP_Range)):
            if ZP_Range[y] == list_vars_ZP_Range[x]:
                data2 = data2.append(data[data.ZP_Range == ZP_Range[y]])
    if ZP_Range[0] == 'All':
        data2 = data
    data = data2
    # An if statement is present to prevent the function from attempting to
    # plot if no particles are selected to prevent encountering an error.
    if data.empty is True:
        print('No particles meet the selected parameters. Please broaden '
              'your filters.')
    else:
        defaults.width = 1000
        p = Histogram(
                data, values='Deff', color='Particle', bins=bins_num,
                title="Deff Distribution of Particles", legend='top_right')
        output_notebook()
        show(p)


def interact_plot_deff():
    """
    This function is to interact with the plot_deff function. Several widgets
    are made, allowing several variables to be toggled to change the plotted
    bokeh output. The entire functionality of these functions can be done by
    simply calling this function.
    It's awkward to make unittests for interact_plot_deff since the input is
    variable from user interaction. Best way to see if it works is simply
    running the function and checking if the displayed particles varies
    depending on selected parameters.
    """
    vars_PEG = ['All', 'Yes', 'No']
    vars_Surfactant = widgets.SelectMultiple(
            description="Surfactant", options=['All', 'UP', 'P80', 'F68',
                                               '5CHA', '2CHA', '0.5CHA']
        )
    vars_Particle_Type = widgets.SelectMultiple(
            description="Particle_Type", options=['All', '58k', '45k', '15k'])
    size_center = 100
    size_tick = 10
    zp_tick = 5
    sizes = set_size_range(100, 10)
    zp = set_zp_range(5)
    list_vars_Size_Range = list()
    list_vars_ZP_Range = list()
    for x in range(0, len(sizes)):
        list_vars_Size_Range.insert(x, sizes['Size_Range'][x])
    list_vars_Size_Range.insert(0, 'All')
    vars_Size_Range = widgets.SelectMultiple(description="Size_Range",
                                             options=list_vars_Size_Range)
    for x in range(0, len(zp)):
        list_vars_ZP_Range.insert(x, zp['ZP_Range'][x])
    list_vars_ZP_Range.insert(0, 'All')
    vars_ZP_Range = widgets.SelectMultiple(description="ZP_Range",
                                           options=list_vars_ZP_Range)
    interact(plot_Deff, bins_num=(1, 20), PEG=vars_PEG,
             Particle_Type=vars_Particle_Type, Surfactant=vars_Surfactant,
             Size_Range=vars_Size_Range, ZP_Range=vars_ZP_Range)


"""
set_deff_bin and set_bin_counts were made to categorize the deff values. They
are no longer needed since I figured out Bokeh.charts automatically sets the
bins of the graph a little too late... They could potentially be of use if
trying to plot them with Bokeh.plotting with more specified characteristics
different than those of Bokeh Histograms.
"""


def set_deff_bin():
    deff['MaxDeff'] = deff[:].max(axis=1)
    high = max(deff['MaxDeff'])
    bin_size = high/10
    deff_bin_array = np.zeros((11, 4))
    deff_bin = pd.DataFrame(deff_bin_array[1:, 1:])
    deff_bin.columns = ['Low', 'High', 'Deff_Range']
    for x in range(0, 10):
        deff_bin.loc[[x], ['Low']] = bin_size*x
        deff_bin.loc[[x], ['High']] = bin_size*(x+1)
        deff_bin.loc[[x], ['Deff_Range']] = '%f to %f' % (deff_bin['Low'][x],
                                                          deff_bin['High'][x])
    return deff_bin


def set_bin_counts():
    deffnew = deff.ix[:, 0:11]
    deff_bin = set_deff_bin()
    n = 0
    total = 0
    for x in range(0, 20):
        for y in range(1, 11):
            deffnew.loc[[x], ['%i' % (y)]] = 0
    deff_bin = set_deff_bin()
    for x in range(0, len(deffnew)):
        for y in range(0, len(deff_bin)):
            n = 0
            total = 0
            for z in range(1, len(deff.columns)-1):
                if (deff['%i' % (z)][x] >= deff_bin['Low'][y] and
                        deff['%i' % (z)][x] < deff_bin['High'][y]):
                    n = n+1
                if deff['%i' % (z)][x] >= 0:
                    total = total+1
            deffnew.loc[[x], ['%i' % (y+1)]] = n/total
    for x in range(0, len(deff_bin)):
        deffnew = deffnew.rename(columns=({'%i' % (x+1):
                                          deff_bin['Deff_Range'][x]}))
    return deffnew
