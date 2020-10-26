import csv
import json
import logging

import numpy as np


def load_track_csv(csv_filepath, time_key='Frame #', x_key='X Coordinate',
                   y_key='Y Coordinate', track_num_key='Track #'):
    """Load a set of tracks from a single csv file"""

    track_times = []
    track_x_vals = []
    track_y_vals = []
    track_nums = []

    # Load all data points as list (since we don't know how long it'll be)
    with open(csv_filepath) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            track_x_vals.append(row[x_key])
            track_y_vals.append(row[y_key])
            track_times.append(row[time_key])
            track_nums.append(row[track_num_key])

    # Convert to array and return
    points = np.column_stack((np.asarray(track_x_vals, dtype=float),
                              np.asarray(track_y_vals, dtype=float),
                              np.asarray(track_times, dtype=float),
                              np.asarray(track_nums, dtype=float)))

    logging.info('CSV tracks loaded.  '
                 f'X vals range from {np.min(points[:, 0]):0.2f} to {np.max(points[:, 0]):0.2f}. '
                 f'Y vals range from {np.min(points[:, 1]):0.2f} to {np.max(points[:, 1]):0.2f}')

    return points

def transpose_xy_rowcol(points_arr):
    """Convert an array of points from XY point coords to matrix coords or vice versa

    Parameters
    ----------
    points_arr: np.ndarray
        Array with shape (n_points, 2) containing point data to be converted.
        Origin must be in the top left and coordinates should follow either
            * matrix format (0th coord increases downward, 1st coord increases
              to the right)
            * XY format (0th coord increases rightward, 1st coord increases
              downward)

    Returns
    -------
    converted_points_arr: np.ndarray
        Array with shape (n_points, 2) containing points converted to the other
        coordinate frame. Origin remains in top left.
    """
    return np.column_stack((points_arr[:, 1], points_arr[:, 0]))


def load_json(track_fpath):
    """Load a single json as a dictionary"""

    with open(track_fpath) as json_f:
        track_dict = json.load(json_f)

    return track_dict


def load_track_batch(track_fpaths, time_key='Times',
                     point_key='Particles_Position'):
    """Load a batch of .track json files and return as array

    Parameters
    ----------
    track_fpaths: list of str
    time_key: str
    point_key: str

    Returns
    -------
    tracker_arr: np.ndarray
        Numpy array with times and locations of particles
    """
    if not track_fpaths:
        return np.empty((0, 4))

    track_times = []
    track_points = []
    track_ids = []

    # Store each track in a list (since we don't know how much data there will be)
    for ti, track_fpath in enumerate(track_fpaths):
        track_dict = load_json(track_fpath)

        track_times.extend(track_dict[time_key])
        track_points.extend(track_dict[point_key])
        track_ids.extend([ti] * len(track_dict[time_key]))

    # If any time points don't have a value, make sure it's 2D for array conversion
    for ind, point in enumerate(track_points):
        if point is None:
            track_points[ind] = [None, None]

    # Convert to numpy. Set dtype so `None` becomes `np.nan`
    track_arr = np.empty((len(track_times), 4))
    track_arr[:, :2] = np.asarray(track_points, dtype=float)
    track_arr[:, 2] = np.asarray(track_times, dtype=float)
    track_arr[:, 3] = np.asarray(track_ids)

    logging.info('Track files loaded. '
                 f'X vals range from {np.nanmin(track_arr[:, 0]):0.2f} to {np.nanmax(track_arr[:, 0]):0.2f}. '
                 f'Y vals range from {np.nanmin(track_arr[:, 1]):0.2f} to {np.nanmax(track_arr[:, 1]):0.2f}')

    return track_arr


def finite_filter(arr):
    """Return array with all non-finite (NaNs or inf) rows removed"""

    good_mask = np.all(np.isfinite(arr), axis=1)
    return  arr[good_mask]