"""
Calculates features from either hand labeled or generated track data
"""
# External dependencies
import json
import logging
import os
import os.path as op

from ast     import literal_eval
from csv     import DictReader, DictWriter
from glob    import glob
from inspect import getmembers, isfunction
from pathlib import Path

# Internal dependencies
from helm_dhm.features import absolute_features
from helm_dhm.features import relative_features

from utils.dir_helper import get_exp_subdir

def save_features(experiment, tracks, config):
    """
    Writes track features to a CSV file

    Parameters
    ----------
    experiment: str
        The path to the experiment directory
    tracks: list of OrderedDict
        The tracks to write
    config: dict
        OWLS configuration dictionary
    """
    path = get_exp_subdir('features_dir', experiment, config, rm_existing=True)
    feat_file = op.join(path, config['features']['output'])

    logging.info(f'Writing track features to {op.join(*Path(feat_file).parts[-2:])}')
    with open(feat_file, 'w') as f:
        # Only write to file if there are tracks to write
        # Otherwise, overwrite blank for downstream task error handling
        if tracks:
            writer = DictWriter(f, fieldnames=tracks[0].keys())

            writer.writeheader()
            for track in tracks:
                writer.writerow(track)

def read_labels(label_file):
    """
    Reads a labels.csv file into memory and converts columns to their
    appropriate dtype.

    Parameters
    ----------
    label_file: str
        Path to labels.csv

    Returns
    -------
    data: list of OrderedDict
        A list of the rows of the csv, where each row is an OrderedDict with the
        keys being the header row of the csv or the fieldnames param
    """
    # Read label CSV data
    with open(label_file) as csv:
        data = list(DictReader(csv))

    # Verify header
    expected_header = set(['frame', 'track', 'X', 'Y', 'motility'])
    if set(data[0].keys()) != expected_header:
        logging.error(f"Incorrect label keys [{data[0].keys()}] in {label_file}")
        return None

    # Determine the dtype of each column; Note, only care about non-string columns (float, int)
    types = {}
    for key, value in data[0].items():
        try:
            types[key] = type(literal_eval(value))
        except:
            pass

    # Now convert everything to its appropriate dtype
    for row in data:
        for key, dtype in types.items():
            row[key] = dtype(row[key])

    # Reformat the rows into track dicts
    lists = set(['X', 'Y', 'frame'])
    tracks = {row['track']: {} for row in data}
    for row in data:
        track = tracks[row['track']]
        for key, value in row.items():
            # If the variable is a list type, append it, otherwise save single instance
            if key in lists:
                if key not in track:
                    track[key] = []
                track[key].append(value)
            else:
                track[key] = value

    # Drop the keys from the dict and convert to a list
    tracks = list(tracks.values())

    # Add the span feature
    for track in tracks:
        track['span'] = len(track['frame'])

    return tracks

def read_tracks(dirpath, rename=None, labels=None, labeled=False):
    """
    Reads a directory of track.json files and formats the data into a list of
    dictionaries, where each dictionary is the data for a single track

    Parameters
    ----------
    dirpath: str
        Path to directory containing track.jsons
    rename: dict
        Dictionary of {old_key: new_key} to rename variables into expected
        key names
    labels: list of OrderedDict
        The list of labeled data loaded from some csv
    labeled: bool
        Only retrieves tracks that match to a hand label

    Returns
    -------
    tracks: list of dict
        A list of track data
    """
    if rename is None:
        rename = {}

    files = glob(op.join(dirpath, '*.json'))

    tracks = []
    for t_file in files:
        with open(t_file, 'r') as f:
            data = json.load(f)

        # Only append if...
        if labeled and data.get('Track_Match_ID'):
            # When labeled = True, only if Track_Match_ID has a value 
            tracks.append(data)
        elif not labeled:
            # When labeled = False, always append
            tracks.append(data)


    # Retrieve labeled information if available
    if labels:
        # Reorganize labels into a dict for easier matching
        labels = {item['track']: item for item in labels}
        match  = lambda id, key: labels[id][key] if key in labels[id] else ''
        for track in tracks:
            # Check if there's a match, if not fill empty
            if track.get('Track_Match_ID'):
                track['motility'] = match(track['Track_Match_ID'], 'motility')
            else:
                track['motility'] = ''

    # Split the positions list into X and Y lists; add `span` to the track data
    for track in tracks:
        track['X'] = []
        track['Y'] = []
        for x, y in track['Particles_Estimated_Position']:
            track['X'].append(x)
            track['Y'].append(y)
        track['span'] = len(track['Times'])

    # Drop keys that will be unused
    retain = set(['X', 'Y', 'span', 'motility', 'Times', 'Track_ID'])
    tracks = [{key: value for key, value in track.items() if key in retain} for track in tracks]

    # Rename keys to expected output format
    for track in tracks:
        for old, new in rename.items():
            track[new] = track.pop(old)

    return tracks

def get_features(experiment, config, save=False, labeled=False):
    """
    Extracts features from labeled data, a collection of track data, or both.
    See notes for different usecases.

    Parameters
    ----------
    experiment: str
        Path to the experiment directory
    config: dict
        OWLS configuration dictionary
    save: bool
        Saves the feature data to a CSV file. Defaults to False
    labeled: bool
        Limits retrieval of tracks to only those that match to some labeled
        data

    Returns
    -------
    tracks: list of dict
        A list of dictionaries that are the extracted feature data for each
        track

    Notes
    -----
    There are four use cases that can occur depending on a combination of
    flags. See the table for each use case. True/False for
    labels.csv/tracks.json means those exist for the use case.

    +------------+-------------+---------------+
    | labels.csv | tracks.json | labeled param | Use case
    +------------+-------------+---------------+
    | True       | False       | True/False    | Extracts features from labeled data only
    | True       | True        | False         | Accept all tracks, match where available, features extracted from track data
    | True       | True        | True          | Accept only labeled tracks, match everything, features extracted from track data
    | False      | True        | True          | Error
    | False      | True        | False         | Accept all tracks, no matching, features extracted from track data
    | False      | False       | True/False    | Error
    +------------+-------------+---------------+


    The config dictionary can expect the following keys. * means it is a
    required key.
    config = {
        *features: {
            *labels: {
                lists: list of str
            },
            tracks: {
                rename: dict
            },
            output: str
            whitelist: list of str
            blacklist: list of str
            drop: list of str
        }
    }
    """
    label_dir  = get_exp_subdir('label_dir', experiment, config)
    track_dir = get_exp_subdir('track_dir', experiment, config)
    
    label_files = glob(op.join(label_dir, '*_labels.csv'))
    if len(label_files) > 1:
        # multiple label files exist, use first one
        logging.warning("Multiple label CSVs found")
        label_file = label_files[0]
        logging.info(f"Using label {Path(label_file).name}")
    elif len(label_files) == 1:
        # exactly one label file exists, use it
        label_file = label_files[0]
        logging.info(f"Using label {Path(label_file).name}")
    else:
        label_file = None
        logging.info("No labels found.")
        logging.info("\nNOTE: If you were expecting a label, you may need to convert to the\nnew _labels.csv format with src/tools/labelbox/label_swap_util.py")

    # Retrieve labeled data
    labels = None
    if label_file:
        logging.info('Reading label data')
        labels = read_labels(label_file)

    # Verify that labels exist if --train_feats was set
    if labeled and labels is None:
        logging.error('--train_feats was specified, but labels do not exist.')
        logging.error('Unable to extract features for model training.')
        if save:
            save_features(experiment, None, config)
        return None

    # Retrieve track data
    if op.exists(track_dir) and os.listdir(track_dir):
        logging.info('Reading track data')
        tracks = read_tracks(
            track_dir,
            rename  = config['features'].get('tracks', {}).get('rename', {}),
            labels  = labels,
            labeled = labeled
        )
    else:
        tracks = labels

    if tracks is None:
        logging.error('There were no label.csv or tracks.jsons files to extract features from')
        if save:
            save_features(experiment, None, config)
        return None

    # Check for white/black lists of feature functions
    whitelist = config['features'].get('whitelist')
    blacklist = config['features'].get('blacklist')

    # Calculate absolute features per track
    logging.info('Calculating absolute features')
    funcs = getmembers(absolute_features, isfunction)
    if whitelist:
        funcs = [(feature, func) for feature, func in funcs if feature in whitelist]
    if blacklist:
        funcs = [(feature, func) for feature, func in funcs if feature not in blacklist]

    for track in tracks:
        for feature, func in funcs:
            try:
                features = func(track)
                track.update(features)
            except:
                logging.exception(f"Absolute feature function {feature}() failed to complete for track {track['track']}")

    # Calculate relative (movie-wide) features
    logging.info('Calculating relative features')
    funcs = getmembers(relative_features, isfunction)
    if whitelist:
        funcs = [(feature, func) for feature, func in funcs if feature in whitelist]
    if blacklist:
        funcs = [(feature, func) for feature, func in funcs if feature not in blacklist]

    for feature, func in funcs:
        try:
            features = func(tracks)
            for track in tracks:
                if track['track'] in features:
                    track.update(features[track['track']])
        except:
            logging.exception(f'Relative feature function {feature}() failed to complete')

    # Add the experiment name to the track data
    dataset = op.basename(experiment)
    for track in tracks:
        track['dataset_name'] = dataset

    # Drop features that are not desired
    if config['features'].get('drop'):
        drop = config['features']['drop']
        for track in tracks:
            for feature in drop:
                track.pop(feature)

    # Sort by track
    tracks.sort(key=lambda track: track['track'])

    # Write to CSV
    if save:
        save_features(experiment, tracks, config)

    return tracks
