import os
import shutil

"""
This snippet of code is used to rename raw data files from teh Nikon microscope
according to my naming convention, because I couldn't figure out how to do it
with the Nikon macro capabilities.  In order to correctly rename files, the
user must change (1) parameters according to desired naming convention and
(2) folder according to the used folder hierarchy and (3) file_type to be
renamed.

This function will rename all files in a folder that are of file_type and
serialize them, so be careful that only desired files are in each folder.
"""
# DIR = './'

# Define parameters
parameters = {}
parameters["channels"] = ["RED"]
parameters["genotypes"] = ["HET"]
parameters["pups"] = ["P1"]
parameters["surface functionalities"] = ["PEG"]
parameters["slices"] = ["S1", "S2"]
parameters["regions"] = ["cortex", "hipp", "mid"]
parameters["replicates"] = [1, 2, 3, 4, 5]

file_type = '.nd2'
folder = "./{slices}/{region}/"
# -----------------------------------------------------------------------------
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
                    for slic in slices:

                        sample_name = "{}_{}_{}_{}_{}_{}".format(channel, genotype, surface_functionality, region, pup, slic)
                        #folder = DIR
                        # folder = "./{genotype}/{pup}/{region}/{channel}/"
                        replicate_counter = 1
                        for name in os.listdir(folder.format(slices=slic, region=region)):
                            if file_type in name:  # and not 'ipynb' in name:
                                if not os.path.isfile(folder.format(slices=slic, region=region)+sample_name + '_' + str(replicate_counter) + file_type):
                                    shutil.move(folder.format(slices=slic, region=region)+name,
                                                folder.format(slices=slic, region=region)+sample_name + '_' + str(replicate_counter) + file_type)
                                    replicate_counter = replicate_counter + 1
                                else:
                                    print('File already exists')
