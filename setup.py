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
      url='https://github.jpl.nasa.gov/OWLS-Autonomy/OWLS-Autonomy',
      entry_points = {
              "console_scripts": [
                  "CSM_flight_pipeline = cli.CSM_flight_pipeline:main",
                  "HELM_flight_pipeline = cli.HELM_flight_pipeline:main",
                  "HELM_ground_pipeline = cli.HELM_ground_pipeline:main",
                  "HELM_analysis_pipeline = cli.HELM_analysis_pipeline:main",
                  "HELM_simulator = cli.HELM_simulator:main",
                  "FAME_flight_pipeline = cli.FAME_flight_pipeline:main",
                  "FAME_ground_pipeline = cli.FAME_ground_pipeline:main",
                  "FAME_analysis_pipeline = cli.FAME_analysis_pipeline:main",
                  "ACME_flight_pipeline = cli.ACME_flight_pipeline:main",
                  "ACME_ground_pipeline = cli.ACME_ground_pipeline:main",
                  "ACME_analysis_pipeline = cli.ACME_analysis_pipeline:main",
                  "HIRAILS_flight_pipeline = cli.HIRAILS_flight_pipeline:main",
                  "HIRAILS_ground_pipeline = cli.HIRAILS_ground_pipeline:main",
                  "HIRAILS_analysis_pipeline = cli.HIRAILS_analysis_pipeline:main",
                  "ACME_evaluation_strict = cli.ACME_evaluation_strict:main",
                  "ACME_simulator = cli.ACME_simulator:main",
                  "reconstruct_mugshot_video = cli.reconstruct_mugshot_video:main",
                  "TOGA_wrapper = cli.TOGA_wrapper:main",
                  "JEWEL = cli.JEWEL:main",
                  "JEWEL_plot_downlink = cli.JEWEL_plot_downlink:main",
                  "set_downlink_status = cli.set_downlink_status:main",
                  "simulate_downlink = cli.simulate_downlink:main",
                  "update_asdp_db = cli.update_asdp_db:main",
                  "set_priority_bin = cli.set_priority_bin:main",
              ]
          },
      package_dir={'': 'src'},
      packages=find_packages(),
      install_requires=required,
      python_requires='>=3.6',
     )
