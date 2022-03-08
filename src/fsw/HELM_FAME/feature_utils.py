"""
Utility functions for absolute_features.py and relative_features.py
"""
import logging

import numpy as np


def apply_white_black_lists(features, white_list=None, black_list=None):
    """Helper to select features based on white and/or black list

    Parameters
    ----------
    features: dict
        Dictionary of track feature metrics to be filtered
    white_list: list or None
        List of feature names to use. If specified features not in white_list
        will be thrown away. Do not specify if specifying `black_list`.
    black_list: list or None
        List of features names to exclude. If specified, all features that do
        not appear in this list will be used. Do not specify if specifying
        `white_list`.

    Returns
    -------
    filtered_features: dict
        Dictionary of same feature values but reduced depending on specification
        of `white_list` or `black_list`
    """

    # Throw warning if we see both lists
    if white_list and black_list:
        logging.warning('Feature white_list and black_list were specified. Using only white_list.')

    if white_list:
        features_filtered = {feat:val for feat, val in features.items()
                             if feat in white_list}
        return features_filtered

    if black_list:
        features_filtered = {feat:val for feat, val in features.items()
                             if feat not in black_list}
        return features_filtered

    # If neither existed, return features
    return features


def substitute_feats(feature_dict, sub_dict):
    '''Update a track dictionary using a substitution dict.

    Parameters
    ----------
    feature_dict: dict
        Dictionary of features to update.
    sub_dict: dict
        Dictionary of keys to be replaced. Format for each key/val pair is:
        `{key_to_check: (value_to_replace, replacement_value)}`

    Returns
    -------
    updated_dict: dict
    '''
    # Copy dict since it's mutable
    updated_dict = feature_dict.copy()

    for key, sub_tup in sub_dict.items():
        if key in updated_dict.keys():             # Check if key exists
            if updated_dict[key] == sub_tup[0]:    # If key exist, check for value match
               updated_dict[key] = sub_tup[1]      # Replace value

    return updated_dict


def normed_autocorr(seq, lags=None, time_intervals=None):
    """
    Computes autocorrelation of a time-series/sequence and normalizes it to the
    value at lag 0

    Parameters
    ----------
    seq: iterable of int or float
        1D time series with at least 2 points
    lags: list of int
        List of lag indices to return. If None, returns autocorr value for all.
    time_intervals: 1D iterable
        Iterable containing the amount of time from one point to the next. If not
        None, will check that the intervals are consistent so autocorrelation is
        valid.

    Returns
    -------
    normed_autocorr: list
        Normalized autocorrelation values at specified lags.
    """
    # Make sure we only use time lags that are less than the sequence length
    n_invalid_lags = 0
    if lags is not None:
        valid_lags = [lag_val for lag_val in lags if lag_val < len(seq)]
        n_invalid_lags = len(lags) - len(valid_lags)
        '''
        # If desired, print autocorrelation warnings for every invalid lag.
        if n_invalid_lags:
            logging.info('Dropping some requested autocorrelation lags as they were longer than the time series.')
        '''
        lags = valid_lags

    # Error check that we have at least two points to compute correlation
    if len(seq) < 2:
        logging.warning('Fewer than two points in sequence. Can\'t compute autocorrelation.')
        return [np.inf] * (n_invalid_lags + len(lags))

    # Convert to array and zero-center
    orig_vals = np.asarray(seq)
    centered_vals = orig_vals - np.mean(orig_vals)

    # Error check that our array is 1-dimensional
    if centered_vals.ndim != 1:
        logging.error('Can\'t compute autocorrelation for sequence that\'s not 1-dimensional.')
        return [np.inf] * (n_invalid_lags + len(lags))

    # Error check that the time sampling is consistent
    if time_intervals is not None:
        time_interval_diffs = np.diff(time_intervals)
        is_close = np.allclose(time_interval_diffs, 0)
        if not is_close:
            logging.warning('Time intervals in autocorrelation are not consistent and will produce misleading results.')

    # Mode must be 'full' to get all lags
    autocorr_vals = np.correlate(centered_vals, centered_vals, mode='full')
    # Only get second half of symmetric autocorrelation sequence
    autocorr_vals = autocorr_vals[len(autocorr_vals) // 2:]

    norm_val = autocorr_vals[0]

    if lags is not None:
        autocorr_vals = [autocorr_vals[i] for i in lags]
    if n_invalid_lags:
        autocorr_vals = autocorr_vals + [np.inf] * n_invalid_lags

    # Make sure we have a valid correlation value
    if np.isfinite(norm_val) and norm_val > 0:
        return autocorr_vals / norm_val  # Normalize by 0th (largest) value

    return [np.inf] * len(autocorr_vals)


# TODO: Explore FFT version to be more efficient
def compute_msd(positions, tau, flow_offset, n_points=20):
    """Compute mean squared displacement in non-FFT approach

    Parameters
    ----------
    positions: list or np.ndarray
        Spatial points for particle of interest
    tau: float
        Amount of time between each float measurement
    flow_offset: tuple or np.ndarray
        Vector representing flow field direction per `tau` time interval
    n_points: int
        Total number of time lags to estimate MSD for. Function will linearly
        sample from from lag 1 to `n_points`. Use a smaller number for faster
        computation.

    Returns
    -------
    lags: list of float
        Time lags used in calculation
    msds: list of float
        Mean squared distance for each lag
    msds_stdev: list of float
        Standard deviation of squared distances for each lag

    Notes
    -----
    Users should ensure that the time intervals between each data point are
    uniform as this is necessary for the MSD calculation.
    """

    # Ensure we have numpy arrays
    positions = np.asarray(positions)
    flow_offset = np.asarray(flow_offset)

    # Generate some time lags to sample. Avoid using all possible lags for efficiency's sake
    lag_inds = np.linspace(1, len(positions), endpoint=False, num=n_points,
                           dtype=int)
    lag_inds = np.unique(lag_inds)  # Makes sure we don't recompute identical lags

    msds = np.zeros(len(lag_inds))
    msds_stdev = np.zeros(len(lag_inds))

    # Loop over each time lag (or "shift")
    for lii, lag_ind in enumerate(lag_inds):
        # Compute displacement and subtract off contribution of background flow
        diffs = (positions[:-lag_ind] - positions[lag_ind:]) - (flow_offset * lag_ind)
        dist = np.sum(np.square(diffs), axis=1)

        # Store displacement and stdev of displacement
        msds[lii] = np.mean(dist)
        msds_stdev[lii] = np.std(dist)

    return lag_inds * tau, msds, msds_stdev