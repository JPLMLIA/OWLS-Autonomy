#!/usr/bin/env python

from setuptools import find_packages
from distutils.core import setup

version = {}
with open('src/version.py', 'r') as fp:
    exec(fp.read(), version)

with open('requirements.txt', 'r') as fp:
    required = fp.read().splitlines()

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(name='OWLS-Autonomy',
      version=version['__version__'],
      description='Autonomy tools for the Ocean Worlds Life Surveyor (OWLS) instrument suite.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Machine Learning and Instrument Autonomy (MLIA) group at JPL',
      author_email='',
      url='https://github.com/JPLMLIA/OWLS-Autonomy',
      package_dir={'': 'src'},
      packages=find_packages(),
      install_requires=required,
      python_requires='>=3.6',
     )