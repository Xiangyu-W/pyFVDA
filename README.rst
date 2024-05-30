.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/pyFVDA.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/pyFVDA
    .. image:: https://readthedocs.org/projects/pyFVDA/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://pyFVDA.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/pyFVDA/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/pyFVDA
    .. image:: https://img.shields.io/pypi/v/pyFVDA.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/pyFVDA/
    .. image:: https://img.shields.io/conda/vn/conda-forge/pyFVDA.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/pyFVDA
    .. image:: https://pepy.tech/badge/pyFVDA/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/pyFVDA
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/pyFVDA

.. .. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
..     :alt: Project generated with PyScaffold
..     :target: https://pyscaffold.org/

.. |

======
pyFVDA
======

Intro
=====

This is a Python package of the Fractional Velocity Dispersion Analysis (FVDA) for analyzing Solar Energetic Electron (SEE) events.

Dependencies
============

pyFVDA utilizes a python interface of Coordinated Data Analysis Web (CDAWeb), namely cdasws, to obtain data from the web server.
The data is usually in the Common Data Format (CDF), which requires `cdflib <https://pypi.org/project/cdflib/>`_ or `CDF NASA Library <https://cdf.gsfc.nasa.gov/>`_ to read/write. Therefore, please ensure you have at least one of these libraries before using the pyFVDA.

Installation
============

Once the CDF Library is installed, the pyFVDA can be installed from Github using::

    pip install git+https://github.com/Xiangyu-W/pyFVDA.git

This will install pyFVDA and all of its dependencies. We recommend installing pyFVDA in a new virtual environment in case of conflicts between the packages. 


The recommended version of Python::

    python_requires = >=3.8.13



Documentation
=============

A running example and the brief introduction of the package can be found `here <https://colab.research.google.com/github/Xiangyu-W/pyFVDA/blob/main/docs/example_pyFVDA.ipynb>`_. 
The detailed description document for the pyFVDA will be updated soon. 

A detailed explanation of the FVDA method can be found in the following papers:

    * Lulu Zhao, Gang Li, Ming Zhang, et al. `Statistical Analysis of Interplanetary Magnetic Field Path Lengths from Solar Energetic Electron Events Observed by WIND <https://doi.org/10.3847/1538-4357/ab2041>`_. The Astrophysical Journal, 878:107 (9pp), 2019 June 20.*

    * Xiangyu Wu, Gang Li, Lulu Zhao, et al. `Statistical Study of Release Time and Its Energy Dependence of In Situ Energetic Electrons in Impulsive Solar Flares <https://doi.org/10.1029/2022JA030939>`_. Journal of Geophysical Research: Space Physics,128, e2022JA030939.*
    

