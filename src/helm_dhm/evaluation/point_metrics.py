'''
Quantitative measures for evaluating tracker performance.
'''
import os.path as op
from pathlib import Path
import logging

import numpy as np
from scipy.spatial.distance import cdist

from utils.track_loaders import (load_track_csv, load_track_batch,
                                 transpose_xy_rowcol)

from helm_dhm.evaluation.reporting import point_score_report, extended_point_report


def run_point_evaluation(label_csv_fpath, track_fpaths, score_report_fpath,
                         extended_report_fpath, config):
    """Evaluate proposed points (from .track files) against a CSV of true points

    Parameters
    ----------
    label_csv_fpath:  str
        Filepath of CSV data containing label data
    track_fpaths: list: list of str
        Filepaths of all track files
    score_report_fpath: str
        Filepath to save score report to (containing precision, recall, etc.)
    config: dict
        Dictionary containing HELM_pipeline configuration

    Returns
    -------
    scores : dict
        the compared metrics
    """

    # Load CSV values, convert points to matrix coordinate system
    true_points = load_track_csv(label_csv_fpath)
    track_numbers = true_points[:, 3]
    true_points[:, :2] = transpose_xy_rowcol(true_points[:, :2])

    # Load track files
    pred_points = load_track_batch(track_fpaths)

    # Calculate point matches
    dist_thresh = float(config['evaluation']['points']['point_eval_dist_threshold'])
    matches = detect_point_matches(true_points[:, 0:3], pred_points[:, 0:3],
                                   dist_thresh)

    # Handle particle points that were not in both sets of the data
    y_true, y_pred = convert_point_matches(matches)

    # Write out per track recall
    extended_point_report(y_pred, track_numbers, extended_report_fpath)
    logging.info(f'Saved ext. point eval: {op.join(*Path(extended_report_fpath).parts[-2:])}')

    logging.info(f'Saved point eval: {op.join(*Path(score_report_fpath).parts[-2:])}')
    # Save out the point evaluation report
    return point_score_report(y_true, y_pred, score_report_fpath)


def detect_point_matches(true_points, pred_points, dist_threshold=5.):
    """Detect if a two series of spatial points overlap within some tolerance.

    Useful for finding spatial overlaps between points on a predicted track and
    a set of ground truth tracks. Option to look for matches just in space or in
    space and time.

    Parameters
    ----------
    true_points: numpy.ndarray
        Array containing ground truth points (e.g., from a set of tracks).
        Shape should be Nx3 with each row containing values as (x, y, time).
    pred_points: numpy.ndarray
        Array containing ground truth points (e.g., from a set of tracks).
        Shape should be Nx3 with each row containing values as (x, y, time). It
        need not match the length of or time span of `true_points`.
    dist_threshold: float
        Maximum allowed Euclidean distance between predicted and true points.

    Returns
    -------
    matches: numpy.ndarray
        2D boolean array (with shape len(`true_points`) x len(`pred_points`))
        specifying all points matching in time and within `dist_threshold`.
    """

    # Matrix to track all point-wise matches
    matches = np.zeros((len(true_points), len(pred_points)), dtype=np.bool)

    # Loop over all ground truth points
    for tp_i, tp in enumerate(true_points):

        # First, find indices in predicted points that match in time
        time_match_inds = np.argwhere(tp[2] == pred_points[:, 2])

        # For matches in time, calculate spatial distance
        dists = cdist(np.atleast_2d(tp[:2]),
                      np.atleast_2d(pred_points[time_match_inds.squeeze(), :2]))

        # If time match also meets distance threshold, record in match matrix
        for dist, tm_i in zip(np.atleast_1d(dists.squeeze()),
                              np.atleast_1d(time_match_inds.squeeze())):
            if dist <= dist_threshold:
                matches[tp_i, tm_i] = True

    return matches


def convert_point_matches(matches):
    """Convert a matrix of matches to binary lists for metrics calculation

    Parameters
    ----------
    matches: np.ndarray
        2D binary array containing pointwise matches between true positives
        (along axis 0) and predictions (along axis 1).

    Returns
    -------
    y_true: np.ndarray
        Bool 1D array representing ground truth. This contains `True` values
        for points that truly exist and `False` values for points proposed in
        the prediction that are false positives.
    y_pred: np.ndarray
        Bool 1D array representing predictions. This contains `True` values
        for ground truth points that were correctly identified and false
        positives and `False` values for false negatives.
    """

    # Construct array for all ground values
    y_true = np.ones(matches.shape[0])

    # Get columns (inds of predictions) with no match to ground truth points
    false_positives = np.invert(np.any(matches, axis=0))
    n_false_positives = np.sum(false_positives)
    # Add these false positives to the ground truth array
    y_true = np.concatenate((y_true, np.zeros(n_false_positives)))

    # Get all points that matched a ground truth point
    y_pred = np.any(matches, axis=1)
    # Add on the number of false positives
    y_pred = np.concatenate((y_pred, np.ones(n_false_positives)))

    return y_true.astype(np.bool), y_pred.astype(np.bool)
