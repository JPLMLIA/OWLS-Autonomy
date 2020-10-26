'''
Autonomous Science Data Product (ASDP) generation library
'''

import os
import csv
import logging
import glob
import yaml
import os.path as op
from pathlib import Path

import numpy               as np
from PIL               import Image
from skimage.io        import imread

from utils.track_loaders import load_json


def get_track_classification_counts(track_fpaths):
    """Tally up number of motile, non-motile, and other tracks"""

    tracks = [load_json(fpath) for fpath in track_fpaths]

    n_motile = 0
    n_non_motile = 0
    other = 0

    # Calculate number of motile/non_motile tracks
    for track in tracks:
        if track['classification'] == "motile":
            n_motile += 1
        elif track['classification'] == "non-motile":
            n_non_motile += 1
        elif track['classification'] in ["ambiguous", "other"]:
            other += 1
        else:
            logging.warning('Motility type not understood. Got: "%s"',
                            track['classification'])

    return n_motile, n_non_motile, other


def mugshots(experiment, holograms, name, label_path, output_dir, config):

    features = config.get('features')
    mugshot_width = config.get('mugshot_width')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tracks = glob.glob(os.path.join(experiment, config.get("experiment_dirs").get('predict_dir'), config.get('detects_str')))

    for track in tracks:

        d = load_json(track)

        classification = d["classification"]
        positions = d["Particles_Estimated_Position"]
        times = d["Times"]
        numSamples = len(times)
        for x in range(0, numSamples):

            t = times[x]
            pos = positions[x]
            row = int(round(pos[0]))
            col = int(round(pos[1]))

            _ = imread(holograms[t])
            dimension = list(_.shape)
            dimension[0] = int(dimension[0] / config.get('resize_factor'))
            dimension[1] = int(dimension[1] / config.get('resize_factor'))
            dim = tuple(dimension)
            img = np.array(Image.fromarray(_).resize((dim[0], dim[1])))

            row_min = row - mugshot_width
            row_max = row + mugshot_width
            col_min = col - mugshot_width
            col_max = col + mugshot_width

            if row_min < 0:
                row_min = 0
            if col_min < 0:
                col_min = 0

            snapshot = img[row_min:row_max, col_min:col_max]
            fr, fc = snapshot.shape

            if fr >= 1 and fc >= 1:
                outImg = Image.fromarray(snapshot, 'L')
                outImg.save(os.path.join(output_dir,'{row}_{col}_{t}_{l}_{r}_{width}.png'.format(row=dimension[0], col=dimension[1], t=t, l=row_min, r=col_min, width=mugshot_width)))


def generate_SUEs(experiment_dir, asdp_dir, track_fpaths, sue_config):
    """Create and save a science utility for a HELM experiment

    Parameters
    ----------
    experiment_dir: str
        Path to experiment directory
    asdp_dir: str
        Path to ASDP directory where outputs will be saved
    track_fpaths: list
        List of an experiment's track files to compute SUE for
    sue_config: dict
        Subset of HELM config parameters relevent to the SUE calculation. Used
        to pull the desired weights and extrema.

    Returns
    -------
    sue: float
        Science Utility Estimate
    """

    n_motile, _, _ = get_track_classification_counts(track_fpaths)

    # Generate SUE vector and pull weights from config
    sue_vec = np.array([n_motile / sue_config['extrema']['n_motile']])
    sue_weights = np.array([sue_config['weights']['n_motile']])

    if np.sum(sue_weights) != 1.:
        logging.warning('Sum of SUE weights != 1. May lead to SUE that does not lie on interval [0, 1]')

    # Clip SUE vector between 0 and 1, and compute weighted average
    sue_clipped = np.clip(sue_vec, 0, 1)
    sue = np.round(np.sum(np.multiply(sue_clipped, sue_weights)), 3)

    # Write SUE to CSV and return value
    exp_name = Path(experiment_dir).name
    sue_csv_fpath = op.join(asdp_dir, f'{exp_name}_sue.csv')

    with open(sue_csv_fpath, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['SUE'])
        writer.writeheader()
        writer.writerow({'SUE': sue})

    return sue


def generate_DDs(experiment_dir, asdp_dir, track_fpaths, dd_config):
    """Create and save a diversity descriptor for a HELM experiment

    Parameters
    ----------
    experiment_dir: str
        Path to experiment directory
    asdp_dir: str
        Path to ASDP directory where outputs will be saved
    track_fpaths: list
        List of an experiment's track files to compute DD for
    dd_config: dict
        Subset of HELM config parameters relevent to the DD calculation. Used
        to pull the desired weights and extrema.

    Returns
    -------
    dd: float
        Diversity Descriptor
    """

    # Get raw inputs to Diversity Descriptor
    n_motile, n_non_motile, n_other = get_track_classification_counts(track_fpaths)

    raw_dd = {'n_motile': n_motile,
              'n_non_motile': n_non_motile,
              'n_other': n_other}
    dd_vals = {}

    weight_sum = 0  # Keep track of total weighting
    for key in raw_dd:
        # Compute DD on [0, 1] interval
        clipped_dd_val = np.clip((raw_dd[key] / dd_config['extrema'][key]), 0, 1)
        # Weight DD and store
        dd_vals[key] = np.round(clipped_dd_val * dd_config['weights'][key], 3)

        # Sum each weight component
        weight_sum += dd_config['weights'][key]

    if weight_sum != 1.:
        logging.warning('Sum of DD weights != 1. May lead to DD that does not lie on interval [0, 1]')

    # Write DD to CSV and return value
    exp_name = Path(experiment_dir).name
    dd_csv_fpath = op.join(asdp_dir, f'{exp_name}_dd.csv')

    with open(dd_csv_fpath, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dd_vals.keys())
        writer.writeheader()
        writer.writerow(dd_vals)

    return dd_vals
