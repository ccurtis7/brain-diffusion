from __future__ import absolute_import, division, print_function

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = 0  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
#  _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: Python"]

# Description should be a one-liner:
description = "brain_diffusion: a tool for calculating and analyzing MSDs from trajectory datasets"

# Long description will go up on the pypi page
long_description = """

brain_diffusion
========
Brain_diffusion is a tool for calculating and analyzing MSDs from trajectory
datasets.

It can calculate MSDs from input trajectories, perform averaging over large
datasets, and visualization tools.  It also provides templates for parallel
computing.

To get started using these components in your own software, please go to the
repository README_.

.. _README: https://github.com/ccurtis7/brain_diffusion/blob/master/README.md

License
=======
``brain_diffusion`` is licensed under the terms of the BSD 2-Clause license. See
the file "LICENSE" for information on the history of this software, terms &
conditions for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2015--, Chad Curtis, The University of Washington.

"""

NAME = "brain_diffusion"
MAINTAINER = "Chad Curtis"
MAINTAINER_EMAIL = "ccurtis7@uw.edu"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/ccurtis7/brain_diffusion"
DOWNLOAD_URL = ""
LICENSE = "BSD"
AUTHOR = "Chad Curtis"
AUTHOR_EMAIL = "ccurtis7@uw.edu"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
REQUIRES = ["numpy", "scipy", "matplotlib"]
