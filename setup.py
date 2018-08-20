#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='ksptrack',
    version='1.0',
    description='KSPTrack Labeling',
    packages=find_packages(),
    install_requires=[
        "networkx==1.10",
        "Cython>=0.22",
        "joblib>=0.8.4",
        "matplotlib>=1.4.3",
        "numpy>=1.9.2",
        "PyYAML>=3.11",
        "scikit-image>=0.11.3",
        "scikit-learn>=0.16.0",
        "scipy>=0.15.1",
        "pytest>=2.7.0",
        "pyflow"
        ],
)
