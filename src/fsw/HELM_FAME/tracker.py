'''
Identifies particle tracks in HELM image data
'''
import glob
import os
import shutil
import logging
import json
import yaml
import subprocess
import signal
import multiprocessing
import math

import os.path              as op
import numpy                as np
import matplotlib.pyplot    as plt
import matplotlib.patches   as patches

from pathlib           import Path
from collections       import defaultdict
from tqdm              import tqdm
from sklearn.cluster   import DBSCAN
from scipy.stats       import rankdata
from skimage.transform import resize
from scipy.ndimage     import gaussian_filter1d
from scipy.spatial     import distance_matrix
from scipy.stats       import norm, chi2
from scipy.interpolate import interp1d

from utils.dir_helper        import get_batch_subdir, get_exp_subdir
from utils.file_manipulation import read_image

# -------- Global Constants ----------------
VELOCITY = np.zeros(2)
ACCELERATION = np.zeros(2)
EPSILON = 1e-9


def run_tracker(exp_dir, holograms, originals, config, n_workers=1):
    """Execute the tracker code for an experiment

    Parameters
    ----------
    exp_dir: str
        Experiment directory path
    holograms: list
        Ordered list of filepaths to holograms
    originals: list
        Ordered list of filepaths to original holograms
    config: dict
        Loaded HELM configuration dictionary
    n_workers: int
        Number of workers to use for multiprocessed portions
    """

    exp_name = Path(exp_dir).name

    tracker_settings = config['tracker_settings']
    track_plot = tracker_settings['track_plot']
    debug_video = tracker_settings['debug_video']
    image_shape = tuple(config['preproc_resolution'])

    track_dir = get_exp_subdir('track_dir', exp_dir, config, rm_existing=True)
    plot_dir = get_exp_subdir('evaluation_dir', exp_dir, config, rm_existing=True)
    tracker_debug_dir = op.join(plot_dir, "tracker_debug")
    medsub_dir = get_exp_subdir('baseline_dir', exp_dir, config)

    # Track and plot directories if they don't exist yet
    Path(tracker_debug_dir).mkdir(parents=True, exist_ok=True)
    logging.info(f'Track files dir: {op.join(*Path(track_dir).parts[-2:])}')
    logging.info(f'Track plots dir: {op.join(*Path(plot_dir).parts[-2:])}')

    # Initialize the tracker model
    track_assignment_settings = tracker_settings['tracking']
    mad = track_assignment_settings['max_assignment_dist']
    mid = track_assignment_settings['max_init_dist']
    mto = track_assignment_settings['min_track_obs']
    mpf = track_assignment_settings['max_projected_frames']
    ua = track_assignment_settings['use_acceleration']
    particle_tracker = particle_tracker_func(max_assignment_dist=mad,
                                             max_init_dist=mid,
                                             min_track_obs=mto,
                                             max_projected_frames=mpf,
                                             use_acceleration=ua)

    metrics = {'tracks_started' : 0, 'detections' : 0, 'total_clusters' : 0, 'too_small_clusters' : 0}

    ###################################
    # Calculate median image of dataset

    median_dataset_image = read_image(op.join(get_exp_subdir('validate_dir', exp_dir, config), 
                                            f'{exp_name}_median_image.tif'),
                                      config['preproc_resolution'])
    median_dataset_image = median_dataset_image.astype(np.float)

    ###############################################################
    # Iterate over hologram, search for particles, assign to tracks
    for holo_ind, (img_fpath, orig_fpath) in tqdm(enumerate(zip(holograms, originals)),
                                                    desc='Running tracker',
                                                    total=len(holograms)):

        image = read_image(img_fpath, config['preproc_resolution'], flatten=True)
        original = read_image(orig_fpath, config['raw_hologram_resolution'])

        # NOTE: DEBUG plot initialization
        # ax00 is the input image
        # ax01 is the median subtracted image
        # ax10 is the diff thresholded image
        # ax11 is the tracker representation
        if debug_video:
            med_sub = None
            # if median subtracted images exist show them
            if len(os.listdir(medsub_dir)) == len(holograms):
                med_sub = read_image(op.join(medsub_dir, "{:04d}".format(holo_ind+1)+config['validate']['baseline_subtracted_ext']),
                                        config['preproc_resolution'])

            plt.style.use('dark_background')
            fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        else:
            ax11 = None

        # NOTE: DEBUG Resize medsub image for debug
        if debug_video:
            if med_sub is not None and med_sub.shape != image_shape:
                med_sub = resize(med_sub, image_shape, anti_aliasing=True)
                # Resize converts to float on interval [0, 1], get back to uint8
                med_sub = (med_sub * 255).astype(np.uint8)

        # Background subtraction with rolling median method
        diff = get_diff_static(image, median_dataset_image, tracker_settings['diff_comp'])

        # full range of values, take percentile transformation
        #diff = percentile_transformation(diff)

        # NOTE: DEBUG image plotting
        if debug_video:
            ax00.imshow(image, cmap="gray", vmin=0, vmax=255)
            ax00.set_title("Input Image")

            if med_sub is not None:
                ax01.imshow(med_sub)
                ax01.set_title("Median-sub Image")

            ax10.imshow(diff, cmap="gray", vmin=0, vmax=255)
            ax10.set_title("Diff Image")

            ax11.set_title("Tracker Output")
            ax11.set_xlim(ax00.get_xlim())
            ax11.set_ylim(ax00.get_ylim())
            ax11.set_aspect('equal')

            # NOTE: DEBUG old track plotting
            for track in particle_tracker['Current_Tracks']:
                track_points = np.array(track['Particles_Estimated_Position'])
                ax11.plot(track_points[:,1], track_points[:,0], color='green')

        # Don't run detections until after some number of startup frames
        skip_frames = tracker_settings['skip_frames']
        if holo_ind >= skip_frames:
            detections = get_particles(diff, original, tracker_settings['clustering'], ax11, metrics)
            metrics['detections'] += len(detections)
            islast = (holo_ind == len(holograms) - 1)
            tracks, particle_tracker = get_particle_tracks(particle_tracker, detections, holo_ind, islast, ax11, metrics)

            # Save finished tracks
            for track in tracks:
                # TODO: Eventually, add in position interpolation. This wasn't applied before
                #track = interp_track_positions(
                #    track, config['detection']['algorithm_settings']['track_smoothing_sigma'])
                save_fpath = op.join(track_dir, f'{track["Track_ID"]:05}.json')
                tracker_save_json(save_fpath, track)

        # NOTE: DEBUG save debug plot
        if debug_video:
            _, labels = ax11.get_legend_handles_labels()
            if len(labels):
                ax11.legend(bbox_to_anchor=(0, -0.06), loc='upper left')
            fig.savefig(op.join(tracker_debug_dir, "{:04d}.png".format(holo_ind)))
            plt.close()

    tracks = sorted(glob.glob(op.join(track_dir, '*.json')))
    logging.info(f'Number of detected tracks: {len(tracks)}')

    logging.debug("Other metrics:\n" + str(metrics))

    if track_plot:
        plot_tracks(tracks, Path(exp_dir).name, plot_dir, win_size=config['preproc_resolution'])
        plot_track_overlay(tracks, Path(exp_dir).name, plot_dir, win_size=config['preproc_resolution'])
        logging.info(f'Plotted tracks and overlays to {plot_dir}')

    if debug_video:
        # NOTE: DEBUG save video
        ffmpeg_input = op.join(tracker_debug_dir, "%4d.png")
        ffmpeg_output = op.join(plot_dir, "debug.mp4")
        ffmpeg_command = ['ffmpeg', '-framerate', '5', '-i', ffmpeg_input, '-y', '-vf', 'format=yuv420p', ffmpeg_output]

        cmd = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = cmd.communicate()
        cmd.send_signal(signal.SIGINT)
        cmd.wait()

    # NOTE: Revert plt style
    plt.style.use('classic')

    return tracks


def percentile_transformation(X):
    """
    Scaling values to between 0 and 255 (unless duplicate values) using percentile transformation
    """
    if X.size == 0:
        raise ValueError('Image is empty.')

    if not ((X >= 0).all() and (X <= 255).all()):
        raise ValueError('Invalid entries for conversion to uint8')

    shape = X.shape
    xx = X.ravel()
    xnz = xx[xx > 0]
    xx[xx > 0] = (255.0 * rankdata(xnz, method='average') / float(xnz.size)).astype(np.uint8)
    return xx.reshape(shape)


def get_diff_static(I, ds_median, config):
    """
    Computes a diff between current image I and the dataset median.

    Parameters
    ----------
    I: 2d array
        the current image frame
    ds_median: 2d array
        the dataset median
    config: dict
        configuration
    """

    diff = abs(I - ds_median)

    abs_threshold = config['absthresh'] 
    pc_threshold = config['pcthresh']
    # Threshold is the max value of the abs_threshold, and the value of diff at percentile pc_threshold
    threshold = max(abs_threshold, np.percentile(diff, pc_threshold))
    # Suppress values of rng_diff less than a threshold
    diff[diff < threshold] = 0

    return diff


def get_particles(range_diff, image, clustering_settings, debug_axis=None, metrics=None):
    """Get the detections using Gary's original method

    Returns a list of particles and their properties

    Parameters
    ----------
    range_diff:
        output from background subtraction
    image:
        original image frame for intensity calculation
        may have a different shape than range_diff
    cluster_settings:
        hyperparameters for the clustering algorithm
    debug_axis:
        plt ax for debug output. Defaults to None.
    metrics: dict
        output parameter for cluster filtering metrics

    Returns
    -------
    list of dicts with keys:
        pos:            (y, x) coordinates of particle
        n:              number of pixels in cluster
        bbox_tl:        bbox (top, left)
        bbox_hw:        bbox (height, width)
        max_intensity:  max intensity of pixels (list)
    """
    if not metrics:
        metrics = {}

    # select points above a threshold, and get their weights
    idx = (range_diff > 0)
    points = np.column_stack(np.nonzero(idx))
    weights = range_diff[idx].ravel().astype(float)

    # NOTE: DEBUG show thresholded points prior to DBSCAN
    if debug_axis is not None:
        debug_axis.scatter(points[:,1], points[:,0], marker='o', s=0.2, color='white', label='thresholded pixels')

    # empty list to store particles
    particles = []

    if len(points) > 0:
        # use DBSCAN to cluster the points
        dbscan = DBSCAN(eps=clustering_settings['dbscan']['epsilon_px'], 
                        min_samples=clustering_settings['dbscan']['min_weight'])
        labels = dbscan.fit_predict(points, sample_weight=weights)
        n_clusters = int(np.max(labels)) + 1

        # keep track of how many clusters were found
        if 'total_clusters' not in metrics:
            metrics['total_clusters'] = 0
        metrics['total_clusters'] += n_clusters

        for l in range(n_clusters):
            idx = (labels == l)
            # must have specified minimum number of points
            # keep track of clusters that fall below this thresh
            if np.sum(idx) < clustering_settings['filters']['min_px']:
                if 'too_small_clusters' not in metrics:
                    metrics['too_small_clusters'] = 0
                metrics['too_small_clusters'] += 1
                continue

            relevant = points[idx]
            
            # Build particle properties
            particle = {}
            # center of particle
            particle['pos'] = np.average(relevant, axis=0)
            # number of pixels in particle
            particle['size'] = int(np.sum(idx))

            # bounding box calculations
            bbox_y, bbox_x = int(np.min(relevant[:,0])), int(np.min(relevant[:,1]))
            bbox_h, bbox_w = int(np.max(relevant[:,0]) - np.min(relevant[:,0])), \
                             int(np.max(relevant[:,1]) - np.min(relevant[:,1]))

            particle['bbox_tl'] = (bbox_y, bbox_x)
            particle['bbox_hw'] = (bbox_h, bbox_w)

            # convert bounding box indices to original resolution
            yres_ratio = image.shape[0] / range_diff.shape[0]
            xres_ratio = image.shape[1] / range_diff.shape[1]
            bbox_y_ores = int(bbox_y * yres_ratio)
            bbox_h_ores = int(bbox_h * yres_ratio)
            bbox_x_ores = int(bbox_x * xres_ratio)
            bbox_w_ores = int(bbox_w * xres_ratio)

            # max intensity for each channel
            if len(image.shape) == 2:
                # grayscale original image, single channel
                particle['max_intensity'] = [int(np.max(image[bbox_y_ores:bbox_y_ores+bbox_h_ores+1, 
                                                              bbox_x_ores:bbox_x_ores+bbox_w_ores+1]))]
            else:
                # RGB original image, max per channel
                particle['max_intensity'] = np.amax(image[bbox_y_ores:bbox_y_ores+bbox_h_ores+1, 
                                                         bbox_x_ores:bbox_x_ores+bbox_w_ores+1],
                                                    axis=(0,1)).tolist()

            particles.append(particle)

    # NOTE: DEBUG show identified particles
    particle_pos = np.array([p['pos'] for p in particles])
    if particle_pos.size and debug_axis is not None:
        debug_axis.scatter(particle_pos[:,1], particle_pos[:,0], marker='o', s=2, color='fuchsia', label='proposed particles')
        for i in range(len(particles)):
            rect = patches.Rectangle(particles[i]['bbox_tl'][::-1],
                                     particles[i]['bbox_hw'][1],
                                     particles[i]['bbox_hw'][0],
                                     color='fuchsia',
                                     fill=False,
                                     linewidth=0.5)
            debug_axis.add_patch(rect)

    return particles


# TODO: Incorporate this function once we need interpolation. At time of
# refactor, interpolation was computed, but not saved within the track files
def interpolate_points(vals, times):
    """Return the interpolated("Particles_Estimated_Position") x positions,
    y positions, and times from a track dictionary

    Parameters
    ----------
    track: dictionary
        Dictionary of a track

    Returns
    -------
    x_interp: numpy.ndarray
        Interpolated x values
    y_interp: numpy.ndarray
        Interpolated y values
    times_interp: numpy.ndarray
        Full, filled in range of time values
    """

    # Create the linear interpolation function
    interp_func = interp1d(times, vals)

    # Generate full list of times and compute interpolated values
    times_interp = np.arange(times[0], times[-1] + 1)
    val_interp = interp_func(times_interp)

    return val_interp, times_interp


# TODO: Incorporate this function once we need interpolation. At time of
# refactor, interpolation was computed, but not saved within the track files
def interp_track_positions(track, sigma=None):
    """Interpolate track positions and apply (optional) gaussian smoothing

    Parameters
    ----------
    track: dictionary
        Dictionary of a track
    sigma: float or None
        Standard deviation for gaussian kernel. Larger values increase
        smoothing. If None (default), no smoothing is applied.

    Returns
    -------
    track: dict
        Track dictionary updated with interpolated and smoothed positions
    """

    # Get x and y positions interpolated at every time point
    particle_pos_arr = np.array(track['Particles_Position'])
    x_interp, times_interp = interpolate_points(particle_pos_arr[:, 0], track['Times'])
    y_interp, _ = interpolate_points(particle_pos_arr[:, 1], track['Times'])

    # Smooth positions if sigma was specified
    if sigma is not None:
        x_interp = gaussian_filter1d(x_interp, sigma)
        y_interp = gaussian_filter1d(y_interp, sigma)

    track['Times_Interpolated'] = times_interp.tolist()
    # TODO: Move to a more reasonable format instead of long list of 2D arrays
    track['Particles_Position_Interpolated'] = \
        [p for p in np.column_stack((x_interp, y_interp))]

    return track


def get_particle_tracks(_particle_tracker, particle_detects, time, is_last, debug_axis=None, metrics=None):
    """This module constructs the particle tracks.

    WARNING: No examples for Finishing. No error warning.

    Parameters
    ----------
    _particle_tracker: dict
        A particle tracker dictionary
    particle_detects: list
        A list of detected particle properties as dicts.
    particle_sizes: list
        A list of detected particle sizes. 
    time: int
        Current time.
    is_last: bool
        Last frame. Boolean.
    debug_axis: ax
        Matplotlib axis for debug plotting. Defaults to None.
    metrics: dict
        output parameter for track filtering metrics

    Returns
    -------
    (A list of track dictionaries, Updated particle tracker dictionary)
    """
    if not metrics:
        metrics = {}

    # Projecting the particle in the 'Current_Tracks' list.
    projection = np.zeros(2)
    projected_points = []

    for track in _particle_tracker['Current_Tracks']:
        projection = project(track['Particles_Estimated_Position'][-1],
                             track['Particles_Estimated_Velocity'][-1],
                             track['Particles_Estimated_Acceleration'][-1],
                             _particle_tracker['Use_Acceleration'],
                             delta_t=1.0)

        projection_copy = projection.copy()  # This prevents duplication in Projection_List
        projected_points.append(projection_copy)
    
    # NOTE: DEBUG plot projected particles
    if len(projected_points) and debug_axis is not None:
        projected_points = np.array(projected_points)
        debug_axis.scatter(projected_points[:,1], projected_points[:,0], marker='o', s=2, color='cyan', label='path projections')

    assignments = defaultdict(list)
    assignment_dists = defaultdict(list)
    unassigned = []

    # If no detection, set it to empty list
    if not particle_detects:
        particle_detects = []

    # Find the distances from the detections to projections and assign to most probable track
    if len(projected_points) == 0 or len(particle_detects) == 0:
        unassigned = particle_detects
    else:
        particle_locs = [p['pos'] for p in particle_detects]
        pairwise_distances = distance_matrix(particle_locs, projected_points)
        for index, particle in enumerate(particle_detects):
            # Get associated track index
            assign_index = np.argmin(pairwise_distances[index])
            # Get assignment distance
            min_assign_dist = np.min(pairwise_distances[index])
            # Get length of associated track
            curr_track_len = len(_particle_tracker['Current_Tracks'][assign_index]['Particles_Position'])

            if curr_track_len == 1 and min_assign_dist > _particle_tracker['Max_Init_Dist']:
                # Second point in a track, but too far from first point
                unassigned.append(particle)
            elif curr_track_len > 1 and min_assign_dist > _particle_tracker['Max_Assignment_Dist']:
                # Future points in a track, but too far from projection
                unassigned.append(particle)
            else:
                # Detection is close enough to be associated to track
                assignments[assign_index].append(particle)
                min_dist = pairwise_distances[index][assign_index]
                assignment_dists[assign_index].append(min_dist)

    finished_tracks = []
    current_tracks = []

    # Updating the tracks
    for i, _ in enumerate(_particle_tracker['Current_Tracks']):
        # If multiple particles got assigned to a track, pick the best one
        best_particle = None
        if len(assignments[i]) > 0:
            best_index = np.argmin(assignment_dists[i])
            best_particle = assignments[i][best_index]

            # Unchosen particles
            unchosen = [p for j, p in enumerate(assignments[i]) if j != best_index]
            for u in unchosen:
                unassigned.append(u)

        track = _particle_tracker['Current_Tracks'][i]
        finished = True
        if (best_particle is not None 
            or track['Projected_Frames'] < _particle_tracker['Max_Projected_Frames']):
            # New particle found or within project limit
            finished = False
            track = update(time, best_particle, track, _particle_tracker['Use_Acceleration'])
            current_tracks.append(track)
        if is_last or finished:
            # Track finished
            track_length = sum(p is not None for p in track['Particles_Position'])
            if track_length >= _particle_tracker['Min_Track_Obs']:
                finished_track = finish(track, _particle_tracker['Last_ID'] + 1)
                _particle_tracker['Last_ID'] = _particle_tracker['Last_ID'] + 1
                finished_tracks.append(finished_track)

    # Adding new tracks from the Unassigned list to the Current_Tracks
    for idx, particle in enumerate(unassigned):
        _particle_tracker['Temp_ID'] += 1
        track = particle_track(time,
                               particle,
                               VELOCITY,
                               ACCELERATION,
                               _particle_tracker['Temp_ID'])
        current_tracks.append(track)
        if 'tracks_started' not in metrics:
            metrics['tracks_started'] = 0
        metrics['tracks_started'] += 1

    _particle_tracker['Current_Tracks'] = current_tracks

    return finished_tracks, _particle_tracker


def particle_tracker_func(max_assignment_dist,
                          max_init_dist,
                          min_track_obs,
                          max_projected_frames,
                          use_acceleration,
                          velocity_prior=VELOCITY,
                          acceleration_prior=ACCELERATION):
    """
    This module is a wrapper around a dictionary that resembles Gary Doran's ParticleTracker class. It collects
    all current tracks. Here, Current_Tracks is a python list of ParticleTrack.

    :param max_assignment_dist: Used to set assignment criterion for association of projections and current particles.
    :param min_track_obs: The minimum number of observations for a track.
    :param velocity_prior: Prior velocity to start a track.
    :param acceleration_prior: Prior acceleration to start a track.
    :return: A particle tracker dictionary
    """
    _particle_tracker = {'Max_Assignment_Dist': max_assignment_dist,
                         'Max_Init_Dist': max_init_dist,
                         'Min_Track_Obs': min_track_obs,
                         'Max_Projected_Frames': max_projected_frames,
                         'Use_Acceleration': use_acceleration,
                         'Velocity_Prior': velocity_prior,
                         'Acceleration_Prior': acceleration_prior,
                         'Current_Tracks': [],
                         'Last_ID': -1,
                         'Temp_ID': -1}
    return _particle_tracker


def finish(_particle_track, track_id):
    """
    This module trims, records the average velocity and finilizes the Track_ID.

    :param _particle_track: A track dictionary.
    :return: The updated track dictionary.
    """
    while len(_particle_track['Particles_Position']) > 0 and _particle_track['Particles_Position'][-1] is None:
        _particle_track['Times'].pop(-1)
        _particle_track['Particles_Position'].pop(-1)
        _particle_track['Particles_Size'].pop(-1)
        _particle_track['Particles_Bbox'].pop(-1)
        _particle_track['Particles_Max_Intensity'].pop(-1)
        _particle_track['Particles_Estimated_Position'].pop(-1)
        _particle_track['Particles_Estimated_Velocity'].pop(-1)

    _particle_track['Track_ID'] = track_id

    return _particle_track

def tracker_save_json(track_file, _particle_track):
    """
    This module dumps a ParticleTrack into a JSON file.

    :param track_file: Name of file to write the track dictionary on.
    :param _particle_track: A track dictionary.
    :return: A JSON file written on disk.
    """
    # Converts the numpy arrays into lists so that json.dump can be used.
    # Hope you're familiar with conditional nested list comprehension -Jake
    interim_dictionary = \
        {'Times': [t for t in _particle_track['Times']],
         'Particles_Position': [[round(i, 1) for i in p.tolist()] if p is not None else None for p in _particle_track['Particles_Position']],
         'Particles_Size': _particle_track['Particles_Size'],
         'Particles_Bbox': _particle_track['Particles_Bbox'],
         'Particles_Max_Intensity': [i if i is not None else None for i in _particle_track['Particles_Max_Intensity']],
         'Particles_Estimated_Position': [[round(i, 1) for i in p.tolist()] for p in _particle_track['Particles_Estimated_Position']],
         'Particles_Estimated_Velocity': [[round(i, 1) for i in p.tolist()] for p in _particle_track['Particles_Estimated_Velocity']],
         'Particles_Estimated_Acceleration': [[round(i, 1) for i in p.tolist()] for p in _particle_track['Particles_Estimated_Acceleration']],
         'Track_ID': _particle_track['Track_ID'],
         'classification': None}

    with open(track_file, 'w') as f:
        json.dump(interim_dictionary, f, indent=2)  # Writing the JSON.

def particle_track(time_0, particle, velocity, acceleration, ID):
    """
    This module creates a ParticleTrack. This module is an equivalent of Gary Doran's ParticleTrack class.
    Remember that this module only creates a ParticleTrack. So, for the momemnt of creation, the Particles and
    its Estimated variables are known, and equal to each-other. However, in time, we might not be able to
    associate a particle to this track. So, the Particles_Position and Particles_Variance would be None. At the
    same time it is possible to update them using the Estimated variables.

    Parameters
    ----------
    time_0: int
        Frame index of track initiation
    particle: dict
        Particle properties in a dict
    velocity: array
        (2,), probably zeros at initialization
    acceleration: array
        (2,), probably zeros at initialization
    
    Returns
    -------
    A dictionary of a track.
    """
    _particle_track = {'Times': [time_0],
                       'Particles_Position': [particle['pos']],
                       'Particles_Size': [particle['size']], 
                       'Particles_Bbox': [(particle['bbox_tl'], particle['bbox_hw'])],
                       'Particles_Max_Intensity': [particle['max_intensity']],
                       'Particles_Estimated_Position': [particle['pos']],
                       'Particles_Estimated_Velocity': [velocity],
                       'Particles_Estimated_Acceleration': [acceleration],
                       'Projected_Frames': 0,
                       'Track_ID': ID}

    return _particle_track


def update(current_time, particle_new, _particle_track, use_acceleration):
    """
    This function updates a ParticleTrack. It can be updated in two ways.
    1- A new particle is associated with the track: In this case, the Particles_Position is the same
       random variable.
    2- No particle is associated with the track: In this case, the Particles_Position is None.

    Parameters
    ----------
    current_time: int
        Current frame time
    particle_new: dict
        A new particle to be associated with the current particle track. 
        It could be None or a dict.
    _particle_track: dict
        Dictionary of a track.

    Returns
    -------
    Updated dictionary of a track.
    """

    # The time difference between the detection and last recorded time.
    time_difference = current_time - _particle_track['Times'][-1]

    if particle_new is None:
        # No new particle found, projection only
        _particle_track['Projected_Frames'] += 1

        # Projection calc
        position_projected = project(
            _particle_track['Particles_Estimated_Position'][-1],
            _particle_track['Particles_Estimated_Velocity'][-1],
            _particle_track['Particles_Estimated_Acceleration'][-1],
            use_acceleration,
            time_difference)
        velocity_projected = ((position_projected - _particle_track['Particles_Estimated_Position'][-1]) /
                              time_difference)  # Explicit Euler method to update velocity.
        if len(_particle_track['Particles_Estimated_Velocity']) > 1: # Wait for first experimental value
            acceleration_projected = ((velocity_projected - _particle_track['Particles_Estimated_Velocity'][-1]) /
                                       time_difference)
        else:
            acceleration_projected = _particle_track['Particles_Estimated_Acceleration'][-1]
    
        # Updating the track
        _particle_track['Times'].append(current_time)
        _particle_track['Particles_Position'].append(None)
        _particle_track['Particles_Size'].append(None)
        _particle_track['Particles_Bbox'].append(None)
        _particle_track['Particles_Max_Intensity'].append(None)
        _particle_track['Particles_Estimated_Position'].append(position_projected)
        _particle_track['Particles_Estimated_Velocity'].append(velocity_projected)
        _particle_track['Particles_Estimated_Acceleration'].append(acceleration_projected)
    else:
        # New particle found, don't store projection
        _particle_track['Projected_Frames'] = 0

        # pos, vel, acc calc
        position_projected = particle_new['pos']
        velocity_projected = ((particle_new['pos'] - _particle_track['Particles_Estimated_Position'][-1]) /
                              time_difference)  # Explicit Euler method to update velocity.
        if len(_particle_track['Particles_Estimated_Velocity']) > 1: # Wait for first experimental value
            acceleration_projected = ((velocity_projected - _particle_track['Particles_Estimated_Velocity'][-1]) /
                                       time_difference)
        else:
            acceleration_projected = _particle_track['Particles_Estimated_Acceleration'][-1]
        
        # Updating the track
        _particle_track['Times'].append(current_time)
        _particle_track['Particles_Position'].append(particle_new['pos'])
        _particle_track['Particles_Size'].append(particle_new['size'])
        _particle_track['Particles_Bbox'].append((particle_new['bbox_tl'], particle_new['bbox_hw']))
        _particle_track['Particles_Max_Intensity'].append(particle_new['max_intensity'])
        _particle_track['Particles_Estimated_Position'].append(position_projected)
        _particle_track['Particles_Estimated_Velocity'].append(velocity_projected)
        _particle_track['Particles_Estimated_Acceleration'].append(acceleration_projected)

        
    return _particle_track


def project(position_current, velocity_current, acceleration_current, use_acceleration, delta_t):
    """
    This module estimates the projected position of a particle.

    :param position_current: The position of the particle at time "t". It must be a numpy array.
    :param velocity_current: The velocity of the particle at time "t". It must be a numpy array.
    :param delta_t: Time increment. It must be positive.
    :param use_acceleration: Ignore acceleration if false.
    :return:   (The projected position of the particle at time "t+1". It is a numpy array,
                The projected position variance of the particle at time "t+1". It is a numpy array.)
    """
    if delta_t <= 0:
        raise ValueError('Timestep must be positive')
    if not use_acceleration:
        acceleration_current = 0
    position_projected = (position_current
                          + delta_t * velocity_current
                          + 0.5 * delta_t * delta_t * acceleration_current)

    return position_projected


def plot_tracks(track_fpaths, exp_name, plot_output_directory,
                win_size=(1024, 1024)):
    """Plot traces for all tracks on a dark background

    Parameters
    ----------
    track_fpaths: list of str
        Full filepaths to each track to be plotted
    exp_name: str
        Experiment name
    plot_output_directory: str
        Directory for saving the track plot
    win_size: iterable
        Number of pixels in row and column dimensions, respectively.
    """

    # Create plot and use dark background
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 8))

    if not track_fpaths:
        logging.warning('No tracks were available to plot')

    # Plot each track
    for track_fpath in track_fpaths:
        # Get all the particle positions
        with open(track_fpath) as yaml_file:
            track_info = yaml.safe_load(yaml_file)

        # Skip None values when plotting. They'll be interpolated in the plot
        positions = [pos for pos in track_info['Particles_Position']
                     if pos is not None]
        pos_array = np.asarray(positions)

        # Get all the particle positions. Row/Col corresponds to Y/X
        ax.plot(pos_array[:, 1], pos_array[:, 0])

    # Set up title and axis labels
    ax.set_title('Particle tracks identified in experiment\n' + exp_name)
    ax.invert_yaxis()
    ax.axis('equal')  # Force a square axis
    ax.set_xlim(0, win_size[1])
    ax.set_ylim(win_size[0], 0)

    fig.savefig(op.join(plot_output_directory, exp_name + "_track_plots.png"),
                dpi=150)
    plt.close()


def plot_track_overlay(track_fpaths, exp_name, plot_output_directory,
                       win_size=(1024, 1024)):
    """Plot traces for all tracks on a transparent background

    Parameters
    ----------
    track_fpaths: list of str
        Full filepaths to each track to be plotted
    exp_name: str
        Experiment name
    plot_output_directory: str
        Directory for saving the track plot
    win_size: iterable
        Number of pixels in row and column dimensions, respectively.
    """

    # set dark_background, which uses plot colors better for track videos
    plt.style.use('dark_background')

    # calculate size for 2048x2048
    px = 1/128
    fig = plt.figure(frameon=False, dpi=128)
    fig.set_size_inches(2048*px, 2048*px)

    # turn off everything except figure itself
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    if not track_fpaths:
        logging.warning('No tracks were available to plot')

    # Plot each track
    for track_fpath in track_fpaths:
        # Get all the particle positions
        with open(track_fpath) as yaml_file:
            track_info = yaml.safe_load(yaml_file)

        # Skip None values when plotting. They'll be interpolated in the plot
        positions = [pos for pos in track_info['Particles_Position']
                     if pos is not None]
        pos_array = np.asarray(positions)

        # Get all the particle positions. Row/Col corresponds to Y/X
        ax.plot(pos_array[:, 1], pos_array[:, 0])

    # Set up title and axis labels
    ax.invert_yaxis()
    ax.axis('equal')  # Force a square axis
    ax.set_xlim(0, win_size[1])
    ax.set_ylim(win_size[0], 0)
    ax.axis('off')

    #fig.tight_layout()
    fig.savefig(op.join(plot_output_directory, exp_name + "_track_overlay.png"),
                dpi=128, transparent=True)
    plt.close()
