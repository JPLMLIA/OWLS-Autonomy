"""
Functionality to simulate a track
"""
import json
import csv
import os.path as op
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from helm_dhm.tracker.tracker import plot_tracks
from helm_dhm.simulator.utils import create_dist_objs, get_track_label_rows


def run_track_sim(config, exp_dir):
    """Simulate a batch of tracks from a config file"""

    ###################################
    # Generate drift distributions and sample once for experiment
    dconf = config['exp_params']['drift']
    drift_dists = create_dist_objs(**dconf)
    drift = [dist.rvs() for dist in drift_dists]

    # If there is a z-dimension in chamber, add 0 drift in the z-dimension
    if config['image_params']['chamber_depth'] and len(drift) == 2:
        drift.append(0)
    drift = np.array(drift)

    ###################################
    # Figure out total number of tracks to simulate, and motile/non-motile designation
    is_motile_list = [True] * config['exp_params']['n_motile'] + \
        [False] * config['exp_params']['n_non_motile']

    # Pull out some hologram image parameters to be used later
    iconf = config['image_params']
    row_val_bounds = [0, iconf['resolution'][0]]
    col_val_bounds = [0, iconf['resolution'][1]]

    ###################################
    # Simulate motile particles and then non-motile particles
    track_fpaths = []
    for track_motility in tqdm(is_motile_list, total=len(is_motile_list),
                               desc='Simulating particle tracks'):
        start_pos = get_random_start_pos(iconf['resolution'], iconf['buffer'],
                                         iconf['chamber_depth'])

        # Choose random particle configuration from list
        if track_motility:
            pconf = random.choice(list(config['motile']['particles'].items()))[1]
            sim_shape = random.choice(config['motile']['shapes'])
            sim_size = create_dist_objs(**config['motile']['size'])[0].rvs()
            sim_brightness = create_dist_objs(**config['motile']['brightness'])[0].rvs()

        else:
            pconf = random.choice(list(config['non_motile']['particles'].items()))[1]
            sim_shape = random.choice(config['non_motile']['shapes'])
            sim_size = create_dist_objs(**config['non_motile']['size'])[0].rvs()
            sim_brightness = create_dist_objs(**config['non_motile']['brightness'])[0].rvs()

        track = TrackGenerator(start_pos,
                               pconf['movement'],
                               pconf['momentum'],
                               is_motile=track_motility,
                               drift=drift)

        track.time_steps(n_steps=config['exp_params']['n_frames'])
        pos_arr = np.array(track.pos)

        # Find time indices where particle was in the sample image
        start_i, end_i = get_valid_window_inds(row_val_bounds, pos_arr[:, 0],
                                               col_val_bounds, pos_arr[:, 1])
        # Skip if there wasn't a valid time index
        # TODO: handle more gracefully
        if start_i is None or end_i is None:
            continue

        track.clip_to_inds(start_i, end_i)

        if track.times:  # Ensure we still have some points
            track_json = track.get_track_json()
            track_json['Track_ID'] = len(track_fpaths)
            track_json['Particle_Shape'] = sim_shape
            track_json['Particle_Size'] = sim_size
            track_json['Particle_Brightness'] = sim_brightness

            track_fpath = op.join(exp_dir, config['sim_track_dir'],
                                  f'{len(track_fpaths):05}{config["track_ext"]}')
            with open(track_fpath, 'w') as json_file:
                json.dump(track_json, json_file, indent=2)

            track_fpaths.append(track_fpath)

    if track_fpaths:
        # Generate plot showing all tracks in experiment
        save_dir = op.join(exp_dir, config['sim_track_dir'])
        plot_tracks(track_fpaths, Path(exp_dir).name, save_dir,
                    config['image_params']['resolution'])

        # Get track motility info for writing
        rescale_factor = [label_size / holo_res for (label_size, holo_res) in
                          zip(config['labels']['label_window_size'],
                              iconf['resolution'])]
        track_data = []
        for track_fpath in track_fpaths:
            track_data.extend(get_track_label_rows(track_fpath, rescale_factor))

        # Save track CSV file containing motility labels
        exp_name = Path(exp_dir).name
        labels_dir = Path(op.join(exp_dir, 'labels'))
        Path.mkdir(labels_dir, exist_ok=True)

        fieldnames = ['Track #', 'X Coordinate', 'Y Coordinate', 'Frame #',
                      'Species', 'Movement type', 'Size']
        with open(op.join(labels_dir, f'verbose_{exp_name}.csv'), 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames)
            writer.writeheader()
            writer.writerows(track_data)


class TrackGenerator:
    """TrackGenerator helps create, store, and manipulate one track of spatiotemporal data"""

    def __init__(self, start_pos, movement_dist_dict, momentum, is_motile,
                 start_time=0, drift=None):
        """Create a track generator object for creating simulated particle motion

        Parameters
        ----------
        start_pos: iterable
            Row, Column (and optionally Depth) position
        movement_dist_dict: dict
            Dict containing distribution name and parameters for sampling motion
        momentum: iterable
            Row, Column (and optionally Depth) momentum used when computing new
            velocities. All values should be on interval [0, 1]
        is_motile: bool
            Whether the particle is motile or not. (Only used when exporting
            track json).
        start_time: int
            Index of first time point
        drift: 2-iterable or None
            Velocity of background particle motion to incorporate into particle
            movement.

        Returns
        -------
        track: TrackGenerator
            Track generator for simulating motion and exporting track files.
        """

        self.momentum = momentum
        self.is_motile = is_motile
        self.times = [start_time]
        self.pos = [np.asarray(start_pos)]
        self.vel = []

        # Check if `drift` was set. Otherwise, set to zeroes
        if drift is not None:
            self.drift = drift
        else:
            self.drift = np.zeros_like(momentum)

        # Initialize velocity distributions
        self.vel_dists = create_dist_objs(**movement_dist_dict)

        # Initialize velocity at first timepoint
        self.get_set_new_velocity(use_momentum=False)

    def time_steps(self, n_steps=1):
        """Iterate for some number of timesteps (updating position/velocity)"""

        for _ in range(n_steps):
            self.times.append(self.times[-1] + 1)
            self.get_set_new_velocity()
            self.get_set_new_position()

    def get_set_new_velocity(self, use_momentum=True):
        """Get the next spatial position using last velocity known and position"""
        sampled_vel = np.array([dist.rvs() for dist in self.vel_dists])

        # Compute new velocity as a combo of old and new velocities
        if use_momentum:
            inertial_contr = (self.vel[-1] - self.drift) * self.momentum
            new_contr = sampled_vel * (np.ones_like(self.momentum) - self.momentum)

        # Compute new velocity as independent sample
        else:
            inertial_contr = np.zeros_like(self.momentum)
            new_contr = sampled_vel

        new_vel = inertial_contr + new_contr + self.drift
        self.vel.append(new_vel)

    def get_set_new_position(self):
        """Calculate/set new spatial position given last position and velocity"""
        new_pos = self.pos[-1] + self.vel[-1]

        # TODO: Check particle limits if needed

        self.pos.append(new_pos)

    def clip_to_inds(self, start_ind, end_ind):
        """Trim track to some set of indices"""
        self.times = self.times[start_ind:end_ind]
        self.pos = self.pos[start_ind:end_ind]
        self.vel = self.vel[start_ind:end_ind]

    def get_track_json(self):
        """Compile info to dict (for eventual json export)"""

        particle_dict = {'Times': self.times,
                         'Particles_Position': [temp_arr[:2].tolist() for temp_arr in self.pos],
                         'Particles_Velocity': [temp_arr[:2].tolist() for temp_arr in self.vel],
                         'Motile': self.is_motile}

        if np.array(self.pos).ndim == 3:
            particle_dict.update({'Particles_Z_Position': [temp_arr[2] for temp_arr in self.pos],
                                  'Particles_Z_Velocity': [temp_arr[2].tolist() for temp_arr in self.vel]})

        return particle_dict

def get_random_start_pos(img_res, img_buffer=None, chamber_depth=None):
    """Generate a random starting position within the sample chamber + buffer

    Parameters
    ----------
    img_res: 2-iterable
        Number of rows and cols in image
    img_buffer: 2-iterable
        Buffer pixels beyond chamber in row and column directions
    chamber_depth
        Depth of chamber in Z-direction (for 3D tracks)

    Returns
    -------
    pos: list
        List specifying randomly drawn position in 2 (or 3) dimensions
    """

    if img_buffer is None:
        img_buffer = [0, 0]

    # Get random position within some image plus a buffer
    row = np.random.uniform(0 - img_buffer[0], img_res[0] + img_buffer[0])
    col = np.random.uniform(0 - img_buffer[1], img_res[1] + img_buffer[1])
    pos = [row, col]

    # Add z position if required
    if chamber_depth:
        pos.append(np.random.uniform(0, chamber_depth))

    return pos


def get_valid_window_inds(row_bounds, row_vals, col_bounds, col_vals):
    """Finds row and column values that are within some range.

    Parameters
    ----------
    row_bounds: 2-iterable
        Min value and (non-inclusive) max value for row index
    row_vals: np.ndarray
        Row positions
    col_bounds: 2-iterable
        Min value and (non-inclusive) max value for col index
    col_vals: np.ndarray
        Col positions

    Returns
    -------
    val_inds: np.ndarray
        Start and end (+1) index of points of the first continuous sequence that
        meet both row and col bounds. Useful for clipping out a track within the
        chamber bounds.

    Note: Used in case particle exits sample chamber frame.
    TODO: should be extended to better handle the same track exiting and re-entering chamber
    """

    # Get all row and col values that fall within range
    valid_rows = np.logical_and(row_vals >= row_bounds[0], row_vals < row_bounds[1])
    valid_cols = np.logical_and(col_vals >= col_bounds[0], col_vals < col_bounds[1])

    val_inds = np.logical_and(valid_rows, valid_cols)

    # Quit if no indices are valid
    if not np.sum(val_inds):
        return None, None

    # Compute start timepoint of first continuous sequence
    start_ind = np.where(val_inds)[0][0]

    # Compute end timepoint of first continuous sequence
    # Works by finding first *invalid* time index and returning that position
    remaining_inds = val_inds[start_ind:]
    if np.any(np.invert(remaining_inds)):
        end_ind = np.where(val_inds[start_ind:] == False)[0][0]
    else:
        end_ind = len(row_vals - start_ind)

    return start_ind, start_ind + end_ind
