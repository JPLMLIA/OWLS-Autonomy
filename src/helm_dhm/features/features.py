'''
Extracts track features from either csv label or helm generated tracks
'''
import glob
import json
import os
import logging
import yaml
import matplotlib
import copy
import math
import statistics

import matplotlib.pyplot as plt
import pandas            as pd
import os.path           as op
import numpy             as np

from pathlib import Path
from typing  import Dict, Any

from utils.dir_helper import get_batch_subdir, get_exp_subdir

xmin = 0
xmax = 1024
ymin = 0
ymax = 1024
fig_width = 12
fig_height = 12
save_dpi = 200
font_size = 24
legend_font_size = 12
marker_size_motile = 160
marker_size_nonmotile = 160
marker_size_helm = 160
marker_motile = "s"
marker_nonmotile = "o"
marker_helm = "v"
alpha_motile = 0.5
alpha_nonmotile = 0.5
alpha_helm = 0.4

log_features = {"mean_velocity": {"plot_min": 10 ** -4, "plot_max": 10 ** 3},
                "mean_accel": {"plot_min": 10 ** -3, "plot_max": 10 ** 2},
                "rel_vel": {"plot_min": 10 ** -4, "plot_max": 10 ** 2}}

color_motile_ref = "#6227cf"
color_nonmotile_ref = "#96eb8d"  # green
color_motile = "#d9274e"  # red
color_nonmotile = "#fc9342"  # yellow
color_helm = "#f5462f"



def _clean_feature_dataframe(df_in):
    """
    Removes rows with missing values of autoCorr_vel_lag1 and autoCorr_stepAngle_lag1 and makes sure column
    'movement_type' values are uniform.
    Parameters
    ----------
    df_in : pandas.core.frame.DataFrame
        Dataframe where each row is a track in a movie and columns are metadata and features.
    Returns
    -------
    df_in : pandas.core.frame.DataFrame
        Cleaned dataframe.
    """
    # Assure format of 'movement_type' and 'size' in df:
    logging.debug('cleaning dataframe to work with graphing format')

    if 'movement_type' in df_in.keys():
        df_in['movement_type'].replace(['Non-motile'], 'non-motile', inplace=True)
        df_in['movement_type'].replace(['Motile'], 'motile', inplace=True)

    if 'size' in df_in.keys():
        df_in['size'].replace(['small'], 'Small', inplace=True)

    # Remove all rows with NaN or inf in certain column:
    check = df_in.autoCorr_vel_lag1.isin([np.nan, np.inf, -np.inf])
    df_in = df_in[~check]
    check = df_in.autoCorr_stepAngle_lag1.isin([np.nan, np.inf, -np.inf])
    df_in = df_in[~check]
    return df_in



def plot_feat_i_vs_j(exp_name, plot_dir, feat_x, feat_y, track_features, reference_csv):
    """
    Scatter plot of feat_x vs feat_y. Names include ["track_length", "max_velocity", "mean_velocity",
    "stdev_velocity", "autoCorr_vel_lag1", "autoCorr_vel_lag2", "max_stepAngle", "mean_stepAngle",
    "autoCorr_stepAngle_lag1", "autoCorr_stepAngle_lag2", "max_accel", "mean_accel", "stdev_accel",
    "autoCorr_accel_lag1", "autoCorr_accel_lag2", "ud_x", "ud_y", "theta_displacement", "rel_vel",
    "rel_theta_displacement", "rel_dir_dot"]
    Parameters
    ----------
    feat_x : str
        one of the feature names
    feat_y : str
        one of the feature names
    """

    # Generated features dataframe
    in_df = _clean_feature_dataframe(track_features.copy())
    df_motile = in_df.loc[in_df['movement_type'] == 'motile']
    df_nonmotile = in_df.loc[in_df['movement_type'] == 'non-motile']
    df_features = in_df

    # Previously generated features
    ref_feat = None
    if not os.path.exists(reference_csv):
        logging.warning(f'reference csv path does not exist: {reference_csv}')
    if os.path.exists(reference_csv):
        logging.info(f'including reference csv in plots')
        ref_feat = _clean_feature_dataframe(pd.read_csv(reference_csv))
        df_motile_ref = ref_feat.loc[ref_feat['movement_type'] == 'motile']
        df_nonmotile_ref = ref_feat.loc[ref_feat['movement_type'] == 'non-motile']
        ref_feat = ref_feat

    try:
        contains_non_motile = df_features["movement_type"].str.contains("non-motile").any()
        contains_motile = df_features["movement_type"].str.contains("motile").any()

        # check if this dataframe is already classified before feature extraction
        if contains_non_motile or contains_motile:
            data_type = 'hand-labeled'
        else:
            data_type = 'helm-tracks'
    except:
        data_type = 'helm-tracks'

    try:
        lower_contains_simulated_species = df_features["species"].str.contains("simulated").any()
        upper_contains_simulated_species = df_features["species"].str.contains("Simulated").any()

        # confirm that this dataframe was from simulated label
        if lower_contains_simulated_species or upper_contains_simulated_species:
            data_type += '_simulated'
    except:
        pass

    plt.style.use('dark_background')
    plt.figure(figsize=(fig_width, fig_height))

    if ref_feat is not None:
        plt.scatter(df_nonmotile_ref[feat_x],
                    df_nonmotile_ref[feat_y],
                    marker=marker_nonmotile,
                    c=color_nonmotile_ref,
                    label="Non-motile: hand-labeled ref",
                    alpha=alpha_nonmotile,
                    s=marker_size_nonmotile)

        plt.scatter(df_motile_ref[feat_x],
                    df_motile_ref[feat_y],
                    marker=marker_motile,
                    c=color_motile_ref,
                    label="Motile: hand-labeled ref",
                    alpha=alpha_motile,
                    s=marker_size_motile)

    _color_motile = color_motile_ref if ref_feat is None and data_type != 'helm' else color_motile
    _color_non_motile = color_nonmotile_ref if ref_feat is None and data_type != 'helm' else color_nonmotile

    if data_type == 'simulated':

        plt.scatter(df_nonmotile[feat_x],
                    df_nonmotile[feat_y],
                    marker=marker_motile,
                    c=_color_non_motile,
                    label="Non-motile: simulated",
                    alpha=alpha_motile,
                    s=marker_size_motile)

        plt.scatter(df_motile[feat_x],
                    df_motile[feat_y],
                    marker=marker_motile,
                    c=_color_motile,
                    label="Motile: labeled simulated",
                    alpha=alpha_motile,
                    s=marker_size_motile)

    elif data_type == "hand-labeled":

        plt.scatter(df_nonmotile[feat_x],
                    df_nonmotile[feat_y],
                    marker=marker_motile,
                    c=_color_non_motile,
                    label="Non-motile: hand-labeled",
                    alpha=alpha_motile,
                    s=marker_size_motile)

        plt.scatter(df_motile[feat_x],
                    df_motile[feat_y],
                    marker=marker_motile,
                    c=_color_motile,
                    label="Motile: hand-labeled",
                    alpha=alpha_motile,
                    s=marker_size_motile)

    else:
        plt.scatter(df_features[feat_x],
                    df_features[feat_y],
                    marker=marker_motile,
                    c=color_helm,
                    label="HELM",
                    alpha=alpha_motile,
                    s=marker_size_motile)

    plt.xlabel(feat_x, fontsize=font_size)
    plt.ylabel(feat_y, fontsize=font_size)
    plt.legend(fontsize=legend_font_size, loc='best')
    ax = plt.gca()

    if feat_x in log_features.keys():
        x_scale = 'log'
        x_plot_min = log_features[feat_x]["plot_min"]
        x_plot_max = log_features[feat_x]["plot_max"]
    else:
        x_scale = 'linear'
        x_plot_min = None
        x_plot_max = None

    if feat_y in log_features.keys():
        y_scale = 'log'
        y_plot_min = log_features[feat_y]["plot_min"]
        y_plot_max = log_features[feat_y]["plot_max"]
    else:
        y_scale = 'linear'
        y_plot_min = None
        y_plot_max = None

    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.set_xlim(left=x_plot_min, right=x_plot_max)
    ax.set_ylim(bottom=y_plot_min, top=y_plot_max)
    file_name = feat_x + "_vs_" + feat_y + ".png"
    plt.savefig(os.path.join(plot_dir, file_name), dpi=save_dpi)



def estimated_autocorrelation(x):
    """
    Computes statistical autocorrelation of a time-series/sequence

    Parameters
    ----------
    x : list
        1D sequence of int or floats

    Returns
    -------
    result : list
        Biased autocorrelation function.
    """
    x = np.array(x)
    lags = range(len(x))
    acf = []
    for lag in lags:
        # Get subseries given lag
        y1 = x[:(len(x) - lag)]
        y2 = x[lag:]
        try:
            # Calculate covariance
            sum_product = np.sum((y1 - np.mean(x)) * (y2 - np.mean(x)))
            # Normalize with sample variance
            acf.append(sum_product / (len(x) * np.var(x)))
        except TypeError:
            raise TypeError('Input array values must be only int or float type')
    return acf


def get_track_absolute_features(track, give_dictionary=False):
    """
    Given features given solely a track's own information (x,y,t). Feature values are np.inf where calculations are
    invalid.

    Parameters
    ----------
    give_dictionary
    track : object
        Object of class Track

    Returns
    -------
    featuresList : list
        List of 18 track features including (in order): ["track_length", "max_velocity", "mean_velocity",
        "stdev_velocity", "autoCorr_vel_lag1", "autoCorr_vel_lag2", "max_stepAngle", "mean_stepAngle",
        "autoCorr_stepAngle_lag1", "autoCorr_stepAngle_lag2", "max_accel", "mean_accel", "stdev_accel",
        "autoCorr_accel_lag1", "autoCorr_accel_lag2", "ud_x", "ud_y", "theta_displacement"]

    """
    x = track["x"]
    y = track["y"]
    t = track["time"]  # check time step = 1 frame
    track_span = track["span"]

    # 1. Velocity
    vel = []
    for i in range(1, track_span):
        # divide by 1 since velocity = change in position/change in frames
        velx = x[i] - x[i - 1]
        vely = y[i] - y[i - 1]
        velTotal = math.sqrt(velx ** 2 + vely ** 2)
        vel.append(velTotal)

    track_length = sum(vel)
    # velNonZero = [i for i in vel if i > 0]
    if len(vel) > 0:
        max_velocity = max(vel)
        mean_velocity = np.mean(vel)
        if len(vel) > 2:
            stdev_velocity = statistics.stdev(vel)
        else:
            stdev_velocity = np.inf
    else:
        logging.info(f"Track {track['track_id']} is not long enough to extract Velocity.")
        max_velocity = np.inf
        mean_velocity = np.inf
        stdev_velocity = np.inf
    # vel autocorrelation
    if len(vel) > 2:
        autoCor = estimated_autocorrelation(vel)
        autoCorr_vel_lag1 = autoCor[1]
        autoCorr_vel_lag2 = autoCor[2]
    else:
        autoCorr_vel_lag1 = np.inf
        autoCorr_vel_lag2 = np.inf

    # 2. Acceleration
    accel = []
    for i in range(1, len(vel)):
        accel.append(vel[i] - vel[i - 1])
    # accelNonZero = [i for i in accel if i > 0]
    if len(accel) > 0:
        max_accel = np.max(np.abs(accel))
        mean_accel = np.mean(np.abs(accel))
        if len(accel) > 2:
            stdev_accel = statistics.stdev(accel)
        else:
            stdev_accel = np.inf
    else:
        logging.info(f"Track {track['track_id']} is not long enough to extract Acceleration.")
        max_accel = np.inf
        mean_accel = np.inf
        stdev_accel = np.inf
    # accel autocorrelation
    if len(accel) > 2:
        autoCor = estimated_autocorrelation(accel)
        autoCorr_accel_lag1 = autoCor[1]
        autoCorr_accel_lag2 = autoCor[2]
    else:
        autoCorr_accel_lag1 = np.inf
        autoCorr_accel_lag2 = np.inf

    # Local angle
    stepAngle = []
    for i in range(1, track_span - 1):
        x_im1 = x[i - 1]
        y_im1 = y[i - 1]
        x_i = x[i]
        y_i = y[i]
        x_ip1 = x[i + 1]
        y_ip1 = y[i + 1]

        vector1x = x_im1 - x_i
        vector1y = y_im1 - y_i

        vector2x = x_ip1 - x_i
        vector2y = y_ip1 - y_i

        angle = math.atan2(vector2y, vector2x) - math.atan2(vector1y, vector1x)
        if (angle < 0):
            angle += 2 * math.pi
        stepAngle.append(angle)

    # stepAngleNonZero = [i for i in stepAngle if i > 0]
    if len(stepAngle) > 0:
        max_stepAngle = max(stepAngle)
        mean_stepAngle = np.mean(stepAngle)
    else:
        logging.info(f"Track {track['track_id']} is not long enough to extract step angle.")
        max_stepAngle = np.inf
        mean_stepAngle = np.inf

    # step angle autocorrelation
    if len(stepAngle) > 2:
        autoCor = estimated_autocorrelation(stepAngle)
        autoCorr_stepAngle_lag1 = autoCor[1]
        autoCorr_stepAngle_lag2 = autoCor[2]
    else:
        autoCorr_stepAngle_lag1 = np.inf
        autoCorr_stepAngle_lag2 = np.inf

    # Track displacement vector
    delxArray = np.zeros(track_span - 1)
    delyArray = np.zeros(track_span - 1)
    for i in range(1, track_span):
        # divide by 1 since velocity = change in position/change in frames
        delx = x[i] - x[i - 1]
        dely = y[i] - y[i - 1]
        delxArray[i - 1] = delx
        delyArray[i - 1] = dely
    ud_x = np.mean(delxArray)
    ud_y = np.mean(delyArray)

    udMag = math.sqrt(ud_x ** 2 + ud_y ** 2)
    if udMag > 0:
        ud_x = ud_x / udMag
        ud_y = ud_y / udMag
        theta_displacement = math.atan2(ud_y, ud_x)
        if (theta_displacement < 0):
            theta_displacement += 2 * math.pi
    elif udMag == 0:
        theta_displacement = 0
    else:
        theta_displacement = np.inf

    featuresList = list(
        [track_length, max_velocity, mean_velocity, stdev_velocity, autoCorr_vel_lag1, autoCorr_vel_lag2, max_stepAngle,
         mean_stepAngle, autoCorr_stepAngle_lag1, autoCorr_stepAngle_lag2, max_accel, mean_accel, stdev_accel,
         autoCorr_accel_lag1, autoCorr_accel_lag2, ud_x, ud_y, theta_displacement])
    if give_dictionary:
        return {"track_length": track_length,
                "max_velocity": max_velocity,
                "mean_velocity": mean_velocity,
                "stdev_velocity": stdev_velocity,
                "autoCorr_vel_lag1": autoCorr_vel_lag1,
                "autoCorr_vel_lag2": autoCorr_vel_lag2,
                "max_stepAngle": max_stepAngle,
                "mean_stepAngle": mean_stepAngle,
                "autoCorr_stepAngle_lag1": autoCorr_stepAngle_lag1,
                "autoCorr_stepAngle_lag2": autoCorr_stepAngle_lag2,
                "max_accel": max_accel,
                "mean_accel": mean_accel,
                "stdev_accel": stdev_accel,
                "autoCorr_accel_lag1": autoCorr_accel_lag1,
                "autoCorr_accel_lag2": autoCorr_accel_lag2,
                "ud_x": ud_x,
                "ud_y": ud_y,
                "theta_displacement": theta_displacement}

    return featuresList


def get_track_relative_features(df):
    """

    Given a dataframe of absolute (track-specific) track features of a dataset, calculate the relative (movie-wide)
    features. "rel_vel" is the ratio of a track's mean velocity to the average mean velocity of all tracks in the movie.
    "rel_theta_displacment" is the directional difference in radians between the vector forming a track's mean direction
    and the average of the mean direction of all tracks in a movie. "rel_dir_dot" is the dot product of the two vectors
    used to calculate "rel_theta_displacment".

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataframe where rows are tracks and columns are absolute track features

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Dataframe where rows are tracks and columns are absolute track features, and 3 relative track features:
        "rel_vel", "rel_theta_displacement", and "rel_dir_dot"

    """

    df["rel_vel"] = float("NaN")
    df["rel_theta_displacement"] = float("NaN")
    df["rel_dir_dot"] = float("NaN")

    for t in range(df.shape[0]):
        dataset_name = df.loc[t, 'dataset_name']
        valid_mean_velocities = df.mean_velocity[df.dataset_name == dataset_name].tolist()
        avg_track_vel_per_movie = np.mean([i for i in valid_mean_velocities if np.isfinite(i)])
        track_mean_velocity = df.loc[t, 'mean_velocity']
        if np.isfinite(track_mean_velocity) and avg_track_vel_per_movie > 0:
            df.loc[t, 'rel_vel'] = track_mean_velocity / avg_track_vel_per_movie

        ud_x = df.loc[t, 'ud_x']
        ud_y = df.loc[t, 'ud_y']
        theta_displacement = df.loc[t, 'theta_displacement']
        if ud_x < np.inf and ud_y < np.inf:
            theta_displacement_movie = df.theta_displacement[df.dataset_name == dataset_name].tolist()
            theta_displacement_movie_avg = np.mean([i for i in theta_displacement_movie if np.isfinite(i)])
            df.loc[t, 'rel_theta_displacement'] = theta_displacement - theta_displacement_movie_avg

            ud_x_mean = np.mean(df.ud_x[df.dataset_name == dataset_name])
            ud_y_mean = np.mean(df.ud_y[df.dataset_name == dataset_name])
            df.loc[t, 'rel_dir_dot'] = np.abs(np.dot([ud_x, ud_y], [ud_x_mean, ud_y_mean]))

    return df


def parse_yaml_file(features_file):
    features = []
    with open(features_file, 'r') as f:
        features_dict = yaml.safe_load(f)

    features_dict = features_dict.get('features')
    for key, value in features_dict.items():
        if value == 1:
            features.append(key)

    return features


def _extract_features_from_dataset(df_dataset, dataset_name, absolute_features):
    """
    Extracts absolute (track-specific) and relative (movie-wide) features given a dataframe of tracks.
    Parameters
    ----------
    df_dataset : pandas.core.frame.DataFrame
        Dataframe with track information. Both SNR and non-SNR versions are acceptable.
    dataset_name : str
        Name of dataset in standard format.
    Returns
    -------
    df_dataset_features : pandas.core.frame.DataFrame
        Dataframe where each row is a track in a movie and columns are metadata and features for the given dataset.
    """
    if df_dataset.shape[1] == 10:
        df_dataset.columns = ['track_id', 'x', 'y', 'frame', 'species', 'movement_type', 'size',
                              'snr_std', 'snr_90', 'snr_max']
    else:
        df_dataset.columns = ['track_id', 'x', 'y', 'frame', 'species', 'movement_type', 'size']

    track_ids = df_dataset.track_id.unique()
    num_tracks_in_dataset = len(track_ids)
    df_columns = ['dataset_name', 'track_id', 'species', 'movement_type', 'size', 'track_span']
    df_columns.extend(absolute_features)
    df_dataset_features = pd.DataFrame(columns=df_columns)
    track_index = 0
    for t in range(num_tracks_in_dataset):
        track_id = track_ids[t]
        track_data = df_dataset.loc[df_dataset.track_id == track_id]
        species = track_data['species'].iloc[0]
        movement_type = track_data['movement_type'].iloc[0]
        size = track_data['size'].iloc[0]
        x = list(track_data.x)
        y = list(track_data.y)
        time = list(track_data.frame)
        track_span = len(time)
        track_absolute_features = get_track_absolute_features({"x":x, "y":y, "time":time, "span":track_span, "dataset_name":dataset_name, "track_id":track_id, "species":species, "movement_type":movement_type, "size":size})
        data_row = [dataset_name, track_id, species, movement_type, size, track_span]
        data_row.extend(track_absolute_features)
        df_dataset_features.loc[track_index] = data_row
        track_index = track_index + 1

    # Get relative (movie-wide) track features
    df_dataset_features = get_track_relative_features(df_dataset_features)
    return df_dataset_features


# XXX: A refactored/improved version of this function exists in
# helm_dhm.tracker.tracker
def plot_movie_tracks(df_dataset, dataset_name, plot_output_directory):
    """
    Generate plot of all tracks in XY space

    Parameters
    ----------
    df_dataset : pandas.core.frame.DataFrame
        Dataframe of dataset tracks including columns for X, Y, and frame #
    dataset_name : str
        Name of dataset in standard format.

    """
    xmin = 0
    xmax = 1024
    ymin = 0
    ymax = 1024
    track_ids = df_dataset.track_id.unique()
    num_tracks_in_dataset = len(track_ids)
    plt.style.use('dark_background')
    plt.figure(figsize=(8, 8))
    for t in range(num_tracks_in_dataset):
        track_id = track_ids[t]
        track_data = df_dataset.loc[df_dataset.track_id == track_id]
        x = list(track_data.x)
        y = list(track_data.y)
        plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax = plt.gca()
    plt.title("DATASET: \n" + dataset_name)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(plot_output_directory,"xy_all_tracks_dataset_" + dataset_name + ".png"), dpi=300)

def update_track_files(dataset_features, dataset_name, features_list):
    """
    Append track features from dataset_features to .track files

    # json load in this track
    # resolve things that apply to one track and append it to 0000.track that matches id
    # apply global movie features to this track as well
    # json write out this track to same location
    # add a flag to write out to a temporary location for debugging

    Parameters
    ----------
    dataset_features : pandas.core.frame.DataFrame
        Dataframe where each row is a track in a movie and columns are metadata and features for the given dataset.
    dataset_name : str
        Name of dataset in standard format.

    """
    logging.info('updating existing track files to store extracted features')
    tracks = glob.glob(os.path.join("output", 'tracks', dataset_name, '*.track'))
    for track in tracks:
        # Read track JSON file as dictionary
        with open(track, 'r') as f:
            track_data = json.load(f)
        # Add extracted features to track
        track_id = track_data["Track_ID"]
        _f = dataset_features.loc[dataset_features['track_id'] == track_id, features_list]
        _f = _f.to_dict(orient='list')
        for key in _f:
            _f.update(dict([(key, _f[key][0])]))
        track_data_with_features = dict(track_data, **_f)

        # Export JSON as track file
        with open(track, 'w') as json_file:
            logging.debug(f'writing track file to {track}')
            json.dump(track_data_with_features, json_file, indent=4)

def get_track_features(experiment, plot_dir, config, allow_plotting=False, train=False):
    """
    Main method to call which takes a dataset and extracts path features to a
    CSV. Also generates a single plot of all the tracks in the dataset in xy space.

    Parameters
    ----------
    experiment: str
        Path to the experiment directory
    plot_dir: str
        Path where the output plot should be saved
    config: dict
        Config read in from YAML
    allow_plotting: bool
        Whether save the plot. Defaults to False.
    train: bool
        Whether to only extract features from tracks with hand labels.
        Defaults to False.

    Returns
    -------
    Dataframe with extracted features
    """
    exp_name = op.basename(experiment)
    labels_dir = get_exp_subdir('label_dir', experiment, config)
    label_path = op.join(labels_dir, f'verbose_{exp_name}.csv')
    if not op.exists(label_path):
        return None
    track_dir = get_exp_subdir('track_dir', experiment, config)

    # Get list of features available
    features_list = config['features']['mask'].keys()

    # Initialize dataframe to store features
    data_track_features = pd.DataFrame(columns=['dataset_name', 'handtrack_id', 'autotrack_id', 'species', 'movement_type', 'size', 'track_span'].extend(features_list))
    absolute_features = config.get('absolute_features')

    # Make output directory
    plot_dir = op.join(get_exp_subdir('output_plot_dir', experiment, config))
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # Get tracks
    logging.info(f'Creating dataframe to calculate features from {exp_name}')

    # Check which files and track directories exist and run appropriate version

    if train:
        # We're extracting features to train on from the tracker and hand label
        # TODO: Change this once we're confident in the tracker output, it should
        # only read hand labeled data at some point
        if op.isdir(track_dir) and op.isfile(label_path):
            # both tracker directory and hand labels exist
            df_dataset = get_tracks_df(config, label_path=label_path,
                                        track_path=track_dir, train=True)
        elif not op.isdir(track_dir) and not op.isfile(label_path):
            logging.error("Labels enforced but no tracker or labels found for experiment {}".format(exp_name))
            return data_track_features
        elif not op.isdir(track_dir):
            logging.error("Labels enforced but no tracker output found for experiment {}".format(exp_name))
            return data_track_features
        elif not op.isfile(label_path):
            logging.error("Labels enforced but no labels found for experiment {}".format(exp_name))
            return data_track_features
        # TODO: To only extract from the hand labels for training, uncomment
        # the lines below
        #if op.isfile(label_path):
        #    df_dataset = get_tracks_df(config, label_path=label_path)
        #else:
        #    logger.error("Labels enforced but no labels found for experiment {}".format(exp_name))
        #    return data_track_features
    else:
        # We're extracting features to test on from the tracker. We won't enforce
        # hand labels existing, but if they do exist we'll provide it.
        if op.isfile(label_path) and op.isdir(track_dir):
            # hand labels do exist, provide both
            df_dataset = get_tracks_df(config, label_path=label_path,
                                        track_path=track_dir, train=False)
        elif op.isdir(track_dir):
            # only track dir exists
            df_dataset = get_tracks_df(config, track_path=track_dir)
        else:
            logging.error("Tracker output not found for experiment {}".format(exp_name))
            return data_track_features

    logging.info(f'extracting features from {exp_name}')
    df_dataset_features = _extract_features_from_dataset(df_dataset, exp_name, absolute_features)

    logging.info(f'append {exp_name} to run total run history')
    data_track_features = data_track_features.append(df_dataset_features, ignore_index=True)

    if allow_plotting:
        plot_movie_tracks(df_dataset, exp_name, plot_dir)

    return data_track_features


def _mil_format_to_csv(track_data: dict, scale_factor: float, flip_axis: bool):
    """
    Converts a dictionary of the json in a .track file to a dataframe that has the
    same format as the output of the FIJI Manual Tracker (aka raw hand label file).

    07/29/2020 Marking this function as internal so we don't accidentally use it
    for another package. There are plenty of better JSON-to-CSV converters in
    this project, so I'm specializing this one for the feature step. Jake Lee.

    Specifically, the reader handles the Track_Match_ID key needed to sync
    hand-labeled labels with the associated autotrack.

    Parameters
    ----------
    track_data : dict
        A dictionary of the json in a .track file
    scale_factor : float
        The ratio that the (x,y) coordinates from the .track file will be scaled before being
        put into the returning dataframe
    flip_axis : bool
        If True, the (x,y) coordinates from the .track file will be flipped before being
        put into the returning dataframe.

    Returns
    -------
        Dataframe with the following columns:
        "Track #"
        "Match #"
        "X Coordinate"
        "Y Coordinate"
        "Frame #"
        "Species"
        "Movement type"
        "Size"
    """
    data = track_data

    xy = data['Particles_Estimated_Position']
    n = len(xy)
    x = []
    y = []
    for i in range(n):
        if flip_axis:
            x.append(xy[i][1] * scale_factor)
            y.append(xy[i][0] * scale_factor)
        else:
            x.append(xy[i][0] * scale_factor)
            y.append(xy[i][1] * scale_factor)

    frameNums = list(data['Times'])
    frameNums = [int(x) for x in frameNums]
    trackID = [data['Track_ID']] * len(frameNums)

    if 'Movement_Type' in data.keys():
        movement_type = [data['Movement_Type']] * len(frameNums)
    else:
        arr = np.zeros(len(frameNums))
        arr[arr == 0] = np.nan
        movement_type = arr.tolist()

    if 'Species' in data.keys():
        species = [data['Species']] * len(frameNums)
    else:
        arr = np.zeros(len(frameNums))
        arr[arr == 0] = np.nan
        species = arr.tolist()

    if 'Size' in data.keys():
        size = [data['Size']] * len(frameNums)
    else:
        arr = np.zeros(len(frameNums))
        arr[arr == 0] = np.nan
        size = arr.tolist()

    if 'Track_Match_ID' in data.keys() and data['Track_Match_ID'] != "":
        match = [data['Track_Match_ID']] * len(frameNums)
    else:
        arr = np.zeros(len(frameNums))
        arr[arr == 0] = np.nan
        match = arr.tolist()

    df_track = pd.DataFrame(
        {'Track #': trackID, 'Match #': match, 'X Coordinate': x, 'Y Coordinate': y, 'Frame #': frameNums, 'Species': species,
         'Movement type': movement_type, 'Size': size})

    return df_track


def _convert_features(input_data_dir, config, enforce_match=False):
    """Converts autotracks as JSONs into hand labeled format

    07/29/2020 Marking this function as internal so we don't accidentally use it
    for another package. There are plenty of better JSON-to-CSV converters in
    this project, so I'm specializing this one for the feature step. Jake Lee

    Specifically, the reader handles the Track_Match_ID key needed to sync
    hand-labeled labels with the associated autotrack.

    Parameters
    ----------
    input_data_dir: str
        Path to directory containing JSON tracks
    config: dict
        Configuration read from YAML
    enforce_match: bool
        If True, only converts tracks that have matched hand labels under
        "Track_Match_ID" of the JSON. The "HandTrack #" column is updated
        accordingly.

    Returns
    -------
        Dataframe with the following columns:
        "Track #"
        "Match #"
        "X Coordinate"
        "Y Coordinate"
        "Frame #"
        "Species"
        "Movement type"
        "Size"

    """

    # List of data directories in track_input_directory
    tracks = sorted(glob.glob(os.path.join(input_data_dir, '*'+config['track_ext'])))

    # Initialize dataframe columns
    df_converted = pd.DataFrame({'Track #': [], 'Match #': [],'X Coordinate': [], 'Y Coordinate': [], 'Frame #': [], 'Species': [], 'Movement type': [], 'Size': []})

    for track in tracks:

        # Read and convert to dataframe
        with open(track, 'r') as f:
            data = json.load(f)
        df_track = _mil_format_to_csv(track_data=data, scale_factor=config.get('resize_factor'), flip_axis=True)

        # Check matched track enforcement
        if enforce_match:
            # If this track doesn't have a matched hand track, then don't include
            if not np.isnan(df_track.iloc[0]['Match #']):
                df_converted = df_converted.append(df_track, ignore_index=True)
        else:
            df_converted = df_converted.append(df_track, ignore_index=True)

    # Enforce integer datatype for indices
    df_converted['Frame #'] = df_converted['Frame #'].astype('int64')
    df_converted['Track #'] = df_converted['Track #'].astype('int64')
    if enforce_match:
        df_converted['Match #'] = df_converted['Match #'].astype('int64')

    return df_converted

def get_tracks_df(config, label_path="", track_path="", train=False):
    """ Get dataframe from tracks for get_track_features()

    This function has different behavior depending on which kwargs are provided.

    Both provided:
        Only the tracks with matched hand labeled tracks are returned. Labels
        are moved from the hand labeled tracks to the auto-tracks.
    Only track_path:
        A standard conversion of the JSON paths to CSV format is returned. There
        are no labels.
    Only label_path:
        A standard read of the hand labeled CSV is returned. There are labels.
    Neither:
        This throws an exception, don't do this.

    Parameters
    ----------
    track_path: str
        Path to directory containing JSON tracks. Defaults to empty str.
    label_path: str
        Filepath to hand-labeled CSV. Defaults to empty str.
    config: dict
        Configuration read from YAML

    Returns
        Dataframe containing required track information
    """

    if train:
        # This is for training. We need the tracks from the tracker but the
        # motility labels from the hand labels. We load both.
        converted_tracker = _convert_features(track_path, config, enforce_match=True)
        hand_labels = pd.read_csv(label_path, converters = {
            "Track #": int,
            "X Coordinate": int,
            "Y Coordinate": int,
            "Frame #": int,
            "Species": str,
            "Movement type": str,
            "Size": str
        })

        # For each row in the converted tracker output,
        # 1. Check the "Match #" column
        # 2. Find the referenced track in the hand labels
        # 3. Get the referenced track's label
        # 4. Put the label in the converted tracker output's
        # This is surprisingly complicated due to dataframes

        # 1. BUILD A HANDTRACK -> MOTILITY LABEL DICTIONARY
        handtrack_label_lookup = {}
        for row in hand_labels.itertuples(index=False):
            handtrack_label_lookup[int(row._0)] = row._5

        # 2. BUILD NEW MOVEMENT TYPE COLUMN
        new_label_column = []
        for ind, row in converted_tracker.iterrows():
            # !!! If you're getting a NaN index error here you set enforce_match
            #     to False when you called _convert_features()
            # !!! If you're getting an index does not exist error here than the
            #     track matching algorithm messed up and matched the autotrack
            #     to a hand track that doesn't exist
            # TODO: See if we ever hit these errors and handle them accordingly
            new_label = handtrack_label_lookup[int(row['Match #'])]
            new_label_column.append(new_label)

        # 3. REPLACE "Movement type" WITH NEW LIST
        converted_tracker['Movement type'] = new_label_column

        # 4. DELETE "Match #" Column before returning
        final_df = converted_tracker.drop('Match #', 1)

    elif track_path and label_path:
        # This is for testing, but we still have tracker output and hand labels
        # for metrics generation. This section is slightly modified from the
        # above section.

        # Note that enforce_match is False here, so some rows will have matched
        # tracks and some will not.
        converted_tracker = _convert_features(track_path, config, enforce_match=False)
        hand_labels = pd.read_csv(label_path, converters = {
            "Track #": int,
            "X Coordinate": int,
            "Y Coordinate": int,
            "Frame #": int,
            "Species": str,
            "Movement type": str,
            "Size": str
        })

        # For each row in the converted tracker output,
        # 1. Check the "Match #" column
        # 2. Find the referenced track in the hand labels
        # 3. Get the referenced track's label
        # 4. Put the label in the converted tracker output's
        # This is surprisingly complicated due to dataframes

        # 1. BUILD A HANDTRACK -> MOTILITY LABEL DICTIONARY
        handtrack_label_lookup = {}
        for row in hand_labels.itertuples(index=False):
            handtrack_label_lookup[int(row._0)] = row._5

        # 2. BUILD NEW MOVEMENT TYPE COLUMN
        new_label_column = []
        for ind, row in converted_tracker.iterrows():
            # This section now tolerates missing Match #s. If there is no
            # Match # for a track, it just appends a blank label.
            if np.isnan(row['Match #']):
                new_label_column.append("")
            else:
                new_label = handtrack_label_lookup[int(row['Match #'])]
                new_label_column.append(new_label)

        # 3. REPLACE "Movement type" WITH NEW LIST
        converted_tracker['Movement type'] = new_label_column

        # 4. DELETE "Match #" Column before returning
        final_df = converted_tracker.drop('Match #', 1)

    elif track_path:
        # This is for testing. Only the tracker output exists, there is no hand
        # label. This is fine, the 'Movement type' column will be NaN.
        converted_tracker = _convert_features(track_path, config, enforce_match=False)

        # Delete the "Match #" column, it's empty.
        final_df = converted_tracker.drop('Match #', 1)

    elif label_path:
        # This is for extracting features from hand labeled tracks directly. Once
        # The tracker has been trained, we can switch to this pipeline.
        final_df = pd.read_csv(label_path)

    else:
        # This is invalid, but don't throw an exception, just return None
        logging.error("get_tracks_df() received two empty filepaths, returning None")
        final_df = None

    return final_df


def output_features(experiment, data_track_features, config):
    """
     Write the features dataframe to CSV (saved as <outfile>.csv) and create feature vs feature scatter plots
    (optional).
    Returns
    -------

    """
    logging.info('writing feature extraction to csv')

    feature_file_name = op.join(get_exp_subdir('features_dir', experiment, config),
                                config['features']['feature_file'])
    allow_plotting = config.get('allow_plotting')
    plot_reference_track_features = config.get('plot_reference_data')
    reference_track_feature_file = config.get('reference_track_feature_file')

    logging.info("Writing features to {file}".format(file=feature_file_name))
    data_track_features.to_csv(feature_file_name, index=False)

    exp_name = Path(experiment).name

    if allow_plotting:
        logging.info('making features plots')

        if plot_reference_track_features:
            logging.info('Including reference features file')

            if os.path.exists(reference_track_feature_file):
                plot_dir = get_exp_subdir('feature_plot_dir', experiment, config)
                Path(plot_dir).mkdir(parents=True, exist_ok=True)
                plot_feat_i_vs_j(exp_name, plot_dir, "mean_velocity", "mean_accel", data_track_features, feature_file_name)
                plot_feat_i_vs_j(exp_name, plot_dir, "mean_velocity", "autoCorr_vel_lag1", data_track_features, feature_file_name)
                plot_feat_i_vs_j(exp_name, plot_dir, "mean_velocity", "rel_dir_dot", data_track_features, feature_file_name)
                plot_feat_i_vs_j(exp_name, plot_dir, "mean_velocity", "mean_stepAngle", data_track_features, feature_file_name)

                plot_feat_i_vs_j(exp_name, plot_dir, "autoCorr_vel_lag1", "mean_stepAngle", data_track_features, feature_file_name)
                plot_feat_i_vs_j(exp_name, plot_dir, "autoCorr_vel_lag1", "autoCorr_vel_lag2", data_track_features, feature_file_name)
                plot_feat_i_vs_j(exp_name, plot_dir, "autoCorr_vel_lag1", "autoCorr_stepAngle_lag1", data_track_features, feature_file_name)
                plot_feat_i_vs_j(exp_name, plot_dir, "autoCorr_vel_lag1", "autoCorr_accel_lag1", data_track_features, feature_file_name)

                plot_feat_i_vs_j(exp_name, plot_dir, "autoCorr_stepAngle_lag1", "autoCorr_stepAngle_lag2", data_track_features, feature_file_name)
                plot_feat_i_vs_j(exp_name, plot_dir, "autoCorr_stepAngle_lag1", "autoCorr_accel_lag1", data_track_features, feature_file_name)

                plot_feat_i_vs_j(exp_name, plot_dir, "rel_vel", "mean_stepAngle", data_track_features, feature_file_name)

            else:
                logging.warning(f'Invalid file path to {reference_track_feature_file}')

    else:
        logging.warning('Plotting for feature extraction has been disabled by configuration')

