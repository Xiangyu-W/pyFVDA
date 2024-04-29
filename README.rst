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

pyFVDA utilizes a python interface of Coordinated Data Analysis Web (CDAWeb), namely cdasws, to obtain data from the web server. cdasws relies on two libraries, spacepy and cdflib. Therefore, before installing pyFVDA, please first check whether the installation `requirements of cdasws <https://pypi.org/project/cdasws/>`_ are met.

Installation
============

Currently, pyFVDA can be installed from Github using::

    pip install git+https://github.com/Xiangyu-W/pyFVDA.git

This will install pyFVDA and all of its dependencies. Since this package is currently in an early version and is still under development, We recommend installing pyFVDA in a new virtual environment.

Documentation
=============

Detailed descriptions & examples will be added soon.

