import csv
import json
import logging

import numpy as np


def load_track_csv(csv_filepath, time_key='frame', x_key='X',
                   y_key='Y', track_num_key='track'):
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

    logging.info(f'Loaded {len(set(track_nums))} CSV tracks.')
    if len(set(track_nums)) > 0:
        logging.info(f'X vals range [{np.min(points[:, 0]):0.2f}, {np.max(points[:, 0]):0.2f}]')
        logging.info(f'Y vals range [{np.min(points[:, 1]):0.2f}, {np.max(points[:, 1]):0.2f}]')

    return points


def convert_labels_to_track_dicts(csv_label_fpath):
    """Convert a hand-label file to a list of track dicts and classes"""

    with open(csv_label_fpath) as csv_file:
        reader = csv.DictReader(csv_file)
        converted_track_dicts = []
        track_labels = []

        track_nums = []
        track_times = []
        track_x_vals = []
        track_y_vals = []
        motility = []

        for row in reader:
            track_nums.append(row['track'])
            track_times.append(row['frame'])
            track_x_vals.append(row['X'])
            track_y_vals.append(row['Y'])
            motility.append(row['motility'])

        # Convert to array and return
        points = np.column_stack((np.asarray(track_nums, dtype=float),
                                  np.asarray(track_times, dtype=float),
                                  np.asarray(track_x_vals, dtype=float),
                                  np.asarray(track_y_vals, dtype=float)))

        n_tracks = np.unique(track_nums)

        # Convert tracks to dicts
        for track_num in n_tracks:
            track_inds = np.nonzero(np.asarray(track_nums) == track_num)[0]

            # Pull individual track from master list of points
            track_sub_arr = np.take(points, track_inds, 0)
            sorted_inds = np.argsort(track_sub_arr[:, 1])
            sorted_track_sub_arr = track_sub_arr[sorted_inds]
            estimated_velocity = np.diff(sorted_track_sub_arr[:, 2:4], axis=0)
            estimated_acceleration = np.diff(estimated_velocity, axis=0)


            # Save as dictionary
            track_dict = {"Times": sorted_track_sub_arr[:, 1].astype(int).tolist(),
                          "Particles_Estimated_Position": sorted_track_sub_arr[:, 2:4].tolist(),
                          "Particles_Estimated_Velocity": estimated_velocity.tolist(),
                          "Particles_Estimated_Acceleration": estimated_acceleration.tolist(),
                          "Track_ID": int(track_num)}

            # Save to dictionary
            converted_track_dicts.append(track_dict)
            track_labels.append(motility[track_inds[0]])  # Assume track motility is consistent. Grab first value

    return converted_track_dicts, track_labels


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

    logging.info(f'Loaded {len(set(track_ids))} tracker tracks.')
    logging.info(f'X vals range [{np.nanmin(track_arr[:, 0]):0.2f}, {np.nanmax(track_arr[:, 0]):0.2f}]')
    logging.info(f'Y vals range [{np.nanmin(track_arr[:, 1]):0.2f}, {np.nanmax(track_arr[:, 1]):0.2f}]')

    return track_arr


def finite_filter(arr):
    """Return array with all non-finite (NaNs or inf) rows removed"""

    good_mask = np.all(np.isfinite(arr), axis=1)
    return  arr[good_mask]