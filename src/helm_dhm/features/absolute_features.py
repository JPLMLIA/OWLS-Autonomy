"""
Functions to calculate track-related metrics
"""
import logging
import numpy as np
from scipy.stats import linregress

from helm_dhm.features import feature_utils as utils


def speed_and_acceleration(points, times=None):
    """
    Calculates speed and acceleration features. First calculates speed,
    then leverages that information to calculate acceleration.

    By doing so as nested functions, memory can be reused while separating the
    logic for organization.

    Parameters
    ----------
    points: np.ndarray
        Array of track points
    times: np.ndarray
        Array of times corresponding to each track point. If provided, speed and
        acceleration features will be in terms of these temporal units.

    Returns
    -------
    features: dict
        Dictionary of features to add into the track's feature dict. Includes track
        length and mean/max/stdev of both speed and acceleration.
    """
    def speed(points, times):
        """
        Calculates speed features

        Parameters
        ----------
        points: np.ndarray
            Spatial coordinates of particle
        times: np.ndarray
            Time cooresponding to each spatial coordinate

        Returns
        -------
        features: dict
            Dictionary of features to add into the track's feature dict
        speeds: np.ndarray
            Speed for each interval
        time_diffs: np.ndarray
            Time difference between each interval
        """
        if times is None:
            times = np.arange(len(points))

        point_diffs = np.diff(points, axis=0)
        dists = np.linalg.norm(point_diffs, axis=1)

        time_diffs = np.diff(times)
        speeds = dists / time_diffs

        if speeds.size > 0:
            dists_sum  = np.sum(dists)
            vmax  = np.max(speeds)
            vmean = np.mean(speeds)

            if len(speeds) > 2:
                vstd = np.std(speeds)
            else:
                vstd = np.inf
        else:
            logging.info(f"Track is not long enough to extract speed features from")
            dists_sum  = 0
            vmax  = np.inf
            vmean = np.inf
            vstd  = np.inf

        # Autocorrelation
        ac_lag1, ac_lag2 = utils.normed_autocorr(speeds, lags=[15, 30],
                                                 time_intervals=time_diffs)

        # Save the features
        features = {
            'track_length': dists_sum,
            'speed_mean': vmean,
            'speed_max': vmax,
            'speed_stdev': vstd,
            'speed_autoCorr_lag1': ac_lag1,
            'speed_autoCorr_lag2': ac_lag2,
        }

        # Return the speeds list to be used in the acceleration function
        return features, speeds, time_diffs

    def acceleration(speeds, time_diffs):
        """
        Calculates acceleration features

        Parameters
        ----------
        speeds: np.ndarray
            Array of speeds returned from speed()
        times_diffs: np.ndarray
            Array of time intervals that correspond to each element in `speeds`

        Returns
        -------
        features: dict
            Dictionary of acceleration features to add into the track's feature
            dict. Note that the absolute value of all accelerations are used.
        """
        # Calculate the accelerations
        accs = np.diff(speeds) / time_diffs

        if accs.size > 0:
            aabs  = np.abs(accs)
            amax  = np.max(aabs)
            amean = np.mean(aabs)

            if len(accs) > 2:
                astd = np.std(accs)
            else:
                astd = np.inf
        else:
            logging.warning("Track is not long enough to extract acceleration features.")
            amax  = np.inf
            amean = np.inf
            astd  = np.inf

        # Autocorrelation
        ac_lag1, ac_lag2 = utils.normed_autocorr(accs, lags=[15, 30],
                                                 time_intervals=time_diffs)

        # Save the features
        features = {
            'accel_mean': amean,
            'accel_max': amax,
            'accel_stdev': astd,
            'accel_autoCorr_lag1': ac_lag1,
            'accel_autoCorr_lag2': ac_lag2,
        }

        return features

    # Run speed() first to get the speeds, then calculate acceleration()
    speed_feats, speeds, time_diffs = speed(points, times)

    # Only pass times from index 1 on since speeds comes from a diff calculation
    accel_feats = acceleration(speeds, time_diffs[1:])

    # Combine the features together to one dict and return
    features = dict(**speed_feats, **accel_feats)

    return features


def local_angle_2D(points, times=None):
    """
    Calculates 2-dimensional absolute step angles in radians where step angle
    is deviation from a straight path when going from one time point to the next

    Parameters
    ----------
    points: np.ndarray
        Array of track points in matrix coordinates (origin in TL, 0th coord
        increases going downward, 1st coord increases going rightward)
    times: iterable
        Iterable containing time when each point was measured. If not None, used
        to check for valid time intervals in autocorrelation calculation.

    Returns
    -------
    features: dict
        Dictionary of features to add into the track's feature dict

    Includes:
        step_angle_mean:   Mean angle deviation from straight (in radians) of each step
        step_angle_max:    Maximum ""
        step_angle_stdev:  Standard deviation of ""
        step_angle_autoCorr_lag1: Autocorrelation at 15 frame of ""
        step_angle_autoCorr_lag2: Autocorrelation at 30 frame of ""
    """
    point_diffs = np.diff(points, axis=0)

    # Convert diffs to Cartesian X,Y coords (X increases going rightward, Y increases upwards)
    diff_x = point_diffs[:, 1]
    diff_y = -1 * point_diffs[:, 0]

    # Get angle of each particle step
    vec_angles = np.arctan2(diff_y, diff_x)

    # Convert from [-π, π] to [0, 2π] so diff calc works properly
    vec_angles = np.mod(vec_angles, 2 * np.pi)
    delta_angles = np.diff(vec_angles)

    # Convert back to [-π, π] and get absolute difference
    # Using [-π, π] interval ensures we pick the smaller of the two angle differences (i.e., diff will not be > π)
    delta_angles = np.abs(np.mod(delta_angles + np.pi, 2 * np.pi) - np.pi)

    if len(delta_angles) > 0:
        max_ang = np.max(delta_angles)
        mean_ang = np.mean(delta_angles)
        stdev_ang = np.std(delta_angles)
    else:
        logging.warning("Track is not long enough to extract step angle features.")
        max_ang = np.inf
        mean_ang = np.inf
        stdev_ang = np.inf

    # Autocorrelation
    time_diffs = None
    if times is not None:
        time_diffs = np.diff(times)
    ac_lag1, ac_lag2 = utils.normed_autocorr(delta_angles, lags=[15, 30],
                                             time_intervals=time_diffs)

    # Save the features
    features = {
        'step_angle_mean'         : mean_ang,
        'step_angle_max'          : max_ang,
        'step_angle_stdev'        : stdev_ang,
        'step_angle_autoCorr_lag1': ac_lag1,
        'step_angle_autoCorr_lag2': ac_lag2
    }

    return features


def displacement(points):
    """
    Calculates the displacement-related features between a series of spatial coords

    Parameters
    ----------
    points: np.ndarray
        Array of track points in matrix coordinates. Origin in TL, positive
        direction of 0th coord is downward, positive direction of 1st coord is
        rightward.

    Returns
    -------
    features: dict
        Dictionary of features to add into the track's feature dict. Includes:

        track_length:   Total distance traveled (sum of inter-frame movements)
        track_lifetime: Total number of frames track was present
        disp_mean_h:    Mean horizontal displacement
        disp_mean_v:    Mean vertical displacement (in matrix coords, positive downward)
        disp_e2e_h:     End-to-end horizontal displacement
        disp_e2e_v:     End-to-end vertical displacement (in matrix coords, positive downward)
        disp_e2e_norm:  Distance of end-to-end displacement
        disp_angle_e2e: Angle (in radians) of total movement vector (0 is rightward, increaseses CC-wise)
        sinuosity:      Movement efficiency calculated as (total_cumulative_path_length/e2e_distance)
    """
    point_diffs = np.diff(points, axis=0)
    step_dists = np.linalg.norm(point_diffs, axis=1)

    # Mean per-frame vertical and horizontal displacement
    disp_mean_v = np.mean(point_diffs[:, 0])
    disp_mean_h = np.mean(point_diffs[:, 1])

    # End to end horizontal/vertical displacement
    disp_e2e_v = points[-1, 0] - points[0, 0]
    disp_e2e_h = points[-1, 1] - points[0, 1]

    # Generate same displacement vectors but in X,Y coord frame
    # Gives angle with 0 pointing rightward, increasing CC-wise
    disp_e2e_y = -1 * disp_e2e_v
    disp_e2e_x = disp_e2e_h

    disp_angle_e2e = np.arctan2(disp_e2e_y, disp_e2e_x)

    # Sinuosity (ratio of path length to E2E euclidean distance)
    track_length = np.sum(step_dists)
    disp_e2e_norm = np.linalg.norm([disp_e2e_h, disp_e2e_v])

    if disp_e2e_norm == 0:
        sinuosity = np.inf
    else:
        sinuosity = track_length / disp_e2e_norm

    # Save the features
    features = {
        'track_length'   : track_length,
        'track_lifetime' : len(points),
        'disp_mean_h'    : disp_mean_h,
        'disp_mean_v'    : disp_mean_v,
        'disp_e2e_h'     : disp_e2e_h,
        'disp_e2e_v'     : disp_e2e_v,
        'disp_e2e_norm'  : disp_e2e_norm,
        'disp_angle_e2e' : disp_angle_e2e,
        'sinuosity'      : sinuosity
    }

    return features


def bbox_size(bbox_dims):
    """Calculates statistics of the particle bounding box size

    Parameters
    ----------
    bbox_dims: 2D np.ndarray
        Iterable with bounding box (width, height) for each point

    Returns
    -------
    features: dict
        Dictionary of size features to add into the track's feature dict
    """

    # Multiply all heights by all widths of each bbox to get area
    areas = bbox_dims[:, 0] * bbox_dims[:, 1]

    features = {'bbox_area_mean': np.mean(areas),
                'bbox_area_median': np.median(areas),
                'bbox_area_max': np.max(areas),
                'bbox_area_min': np.min(areas)}

    return features


def msd_slope(points, tau_interval=1., flow_offset=None):
    """Generate the Mean Squared Displacement slope

    Parameters
    ----------
    points: np.ndarray
        Spatial position of particle over time
    tau_interval: float
        Amount of time between each point measurement
    flow_offset: iterable
        Flow vector. Spatial dimensions should match `points`

    Returns
    -------
    features: dict
        Dictionary containing MSD slope feature to add into the track's feature dict
    """
    if flow_offset is None:
        flow_offset = np.zeros_like(points[0, :])

    # Compute MSD and get slope
    lag_times, msds, _ = utils.compute_msd(points, tau_interval, flow_offset)
    slope = np.inf

    # Error checks: make sure we have at least 2 points and value lies in a reasonable range
    if len(lag_times) > 1:
        slope = linregress(lag_times, msds).slope

    if not np.isfinite(slope) or slope < 0:
        slope = np.inf

    features = {'msd_slope': slope}
    return features