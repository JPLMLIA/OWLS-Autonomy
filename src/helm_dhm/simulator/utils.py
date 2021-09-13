"""
Utility functions for HELM simulator
"""
import json
import logging
import os.path as op

import numpy as np
from scipy import stats

VALID_CONFIG_DISTS = ['truncnorm']
VALID_CONFIG_SHAPES = ['gaussian', 'airy_disk']


def create_dist_objs(**kwargs):
    """Create a scipy distribution object that we can sample"""
    dists = []

    # Truncated normal distribution
    if kwargs['distribution_name'] == 'truncnorm':

        # Confirm kwargs are in list format
        for kw in ['mean', 'std', 'min', 'max']:
            if not isinstance(kwargs[kw], list):
                kwargs[kw] = [kwargs[kw]]

        # Loop through specified velocity means and std devs
        for mu, sigma, min_val, max_val in zip(kwargs['mean'], kwargs['std'],
                                               kwargs['min'], kwargs['max']):
            dist = stats.truncnorm((min_val - mu) / sigma,
                                   (max_val - mu) / sigma,
                                   loc=mu,
                                   scale=sigma)
            dists.append(dist)
    else:
        raise ValueError(f'Distribution name ({kwargs["distribution_name"]}) not recognized.')

    return dists


def config_check(config_dict):
    """Validates a simulation configuration

    Parameters
    ----------
    config_dict: dict
        Dictionary containing loaded simulator config
    """

    ###################################
    # Image params
    logging.info('Checking simulation image parameters')
    ip = config_dict['image_params']
    n_chamber_dims = 3 if ip['chamber_depth'] else 2

    if len(ip['resolution']) != 2:
        raise ValueError('Image `resolution` must be 2 dimensional')
    if len(ip['buffer']) != 2:
        raise ValueError('Image `buffer` must be 2 dimensional')
    if ip['chamber_depth']:
        if not 0 <= ip['focus_plane'] <= ip['chamber_depth']:
            raise ValueError(f'`focus_plane` must be on interval [0, {ip["chamber_depth"]}]')
    distribution_check(ip['noise'], 1)

    ###################################
    # Experiment params
    logging.info('Checking simulation experiment parameters')
    distribution_check(config_dict['exp_params']['drift'], 2)

    ###################################
    # Particles

    # Movement, size, brightness distributions
    logging.info('Checking simulation particle parameters')
    particles = config_dict['non_motile']['particles'].copy()
    particles.update(config_dict['motile']['particles'])

    for val in particles.values():

        model_fpath = val['movement'].get('model_fpath')
        if model_fpath is not None:
            if not op.exists(model_fpath):
                raise ValueError(f'Can\'t find model for simulating track paths ({model_fpath}). Confirm it exist and check that relative file path is visible')
        else:
            distribution_check(val['movement'], n_chamber_dims)

    distribution_check(config_dict['non_motile']['size'], 1)
    distribution_check(config_dict['non_motile']['brightness'], 1)
    distribution_check(config_dict['motile']['size'], 1)
    distribution_check(config_dict['motile']['brightness'], 1)

    # Particle shapes
    for particle_shape in config_dict['non_motile']['shapes']:
        if particle_shape not in VALID_CONFIG_SHAPES:
            raise ValueError(f'Shape `{particle_shape}` not recognized.')
    for particle_shape in config_dict['motile']['shapes']:
        if particle_shape not in VALID_CONFIG_SHAPES:
            raise ValueError(f'Shape `{particle_shape}` not recognized.')


def distribution_check(dist_dict, n_noise_dims):
    """Validate a dictionary with keywords defining a scipy distribution"""

    if not dist_dict['distribution_name'] in VALID_CONFIG_DISTS:
        raise ValueError(f'Distribution {dist_dict["distribution_name"]} not recognized.')
    for key in ['mean', 'std', 'min', 'max']:
        if not key in dist_dict.keys():
            raise ValueError(f'Missing distribution key "{key}"')

        if not(isinstance(dist_dict[key], list)):
            dist_dict[key] = [dist_dict[key]]
        if len(dist_dict[key]) != n_noise_dims:
            raise ValueError('Length of each distribution parameter list incorrect.'
                             f' Got {len(dist_dict[key])}, expected {n_noise_dims}.')
    if not dist_dict['min'] <= dist_dict['mean'] <= dist_dict['max']:
        raise ValueError('Distribution must have min <= mean <= max')


def get_track_label_rows(track_fpath, rescale_factor=(0.5, 0.5)):
    """Load a track and get rows that can be saved to a CSV label file

    Parameters
    ----------
    track_fpath: str
        Path to track .json file containing a simulated track
    rescale_factor: tuple
        Proportion to multiply coordinates by (e.g., to convert labels from
        2048x2048 window to 1024x1024, use (0.5, 0.5)). Define as
        (row_rescale, col_rescale).

    Returns
    -------
    row_data: list of dict
        Dictionaries with each containing 1 row of information to be written to
        the labels CSV file.
    """

    # Load track json from file
    with open(track_fpath, 'r') as json_file:
        track_dict = json.load(json_file)

    # Determine static track properties
    track_num = track_dict['Track_ID']
    motility = 'motile' if track_dict['motility'] is True else 'non-motile'

    # Load position/time information
    frame_nums = track_dict['Times']
    row_vals, col_vals = [], []
    for pos in track_dict['Particles_Position']:
        row_vals.append(int(np.around(pos[0] * rescale_factor[0])))
        col_vals.append(int(np.around(pos[1] * rescale_factor[1])))

    # Save to a list of dict that can be written to CSV
    row_data = []
    for frame_num, row, col in zip(frame_nums, row_vals, col_vals):
        row_data.append({'track': track_num,
                         'X': col,
                         'Y': row,
                         'frame': frame_num,
                         'motility': motility})

    return row_data
