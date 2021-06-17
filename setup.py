#!/usr/bin/env python
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(name='ksptrack',
      version='1.0',
      description='KSPTrack Labeling',
      install_requires=install_requires,
      packages=find_packages())
