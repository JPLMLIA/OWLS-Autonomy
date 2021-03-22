"""
Evaluation metrics quantifying how well proposed tracks match label tracks
"""
import os.path as op
import logging
import json
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from utils.track_loaders import (load_track_csv, load_track_batch,
                                 transpose_xy_rowcol, finite_filter)
from helm_dhm.evaluation.reporting import track_score_report, plot_labels


def run_track_evaluation(label_csv_fpath, track_fpaths, eval_subdir,
                         n_frames, experiment_name, config):
    """Evaluate proposed tracks (from .track files) against a CSV of true tracks

    Parameters
    ----------
    label_csv_fpath:  str
        Filepath of CSV data containing label data
    track_fpaths: list: list of str
        Filepaths of all track files
    eval_subdir: str
        Directory path to track evaluation output
    n_frames: int
        Number of frames in this exepriment
    experiment_name: str
        Name of experiment
    config: dict
        Dictionary containing HELM_pipeline configuration
    """
    
    score_report_fpath = op.join(eval_subdir,
                                 experiment_name + '_track_evaluation_report.json')
    # Load CSV values, convert (x, y) points to matrix coordinate system (row, col)
    true_points = load_track_csv(label_csv_fpath)
    true_points[:, :2] = transpose_xy_rowcol(true_points[:, :2])

    # Load track files
    if config['evaluation']['use_interpolated_tracks']:
        raise NotImplementedError()
        #TODO: need to confirm estimated times/positions are generated
        pred_points = load_track_batch(track_fpaths,
                                       time_key='Estimated_Times',
                                       point_key='Particles_Estimated_Position')
    else:
        pred_points = finite_filter(load_track_batch(track_fpaths))

    true_track_list = unique_track_list(true_points[:, :3], true_points[:, 3])

    plot_labels(true_track_list, experiment_name, eval_subdir)

    # If there are no tracks loaded, return appropriately
    if len(pred_points) == 0:
        n_true_tracks = len(true_track_list)
        return track_score_report({}, true_track_list, [], 0, 0, score_report_fpath, n_frames)

    # Convert raw track arrays into list of individual track arrays
    pred_track_list = unique_track_list(pred_points[:, :3], pred_points[:, 3])

    n_true_tracks = len(true_track_list)
    n_pred_tracks = len(pred_track_list)

    # Apply desired matching algorithm
    track_match_algo = config['evaluation']['tracks']['track_matcher']
    if track_match_algo == 'simple_spatial_match':
        matcher_config = config['evaluation']['tracks']['simple_spatial_match_args']
        coverage_thresh = matcher_config['track_association_overlap_threshold']
        matches = simple_spatial_match(true_track_list, pred_track_list,
                                         matcher_config)
    else:
        raise NotImplementedError(f'Track matcher {track_match_algo} not implemented')

    match_arr = matches.keys()

    if len(match_arr) != 0:
        for ti, track_fpath in enumerate(track_fpaths):
            match_row = [m[0] for m in match_arr if m[1] == ti]
            match_val = int(match_row[0]) if len(match_row) > 0 else ""

            with open(track_fpath, 'r') as json_file:
                track_dict = json.load(json_file)
            with open(track_fpath, 'w') as json_file:
                track_dict['Track_Match_ID'] = match_val
                json.dump(track_dict, json_file, indent=2)

    # Save out/return the track evaluation report
    logging.info(f'Saved track eval: {op.join(*Path(score_report_fpath).parts[-2:])}')
    return track_score_report(matches, true_track_list, pred_track_list, 
                              n_pred_tracks, coverage_thresh, score_report_fpath,
                              n_frames)


def unique_track_list(track_points, track_ids):
    """Convert array of full track set into list of arrays (each containing x,y,t points)

    Parameters
    ----------
    coords: numpy.ndarray
        Numpy array containing points coordinates. Each coordinate must be a new
        row with value in each dimension along the columns axis.
    track_ids: numpy.ndarray
        Array with same length as `coords` containing the track ID number.

    Returns
    -------
    track_list: list of  numpy.ndarray
        List of arrays with each array containing (x, y, t) points for a single
        track.
    """

    if len(track_points) != len(track_ids):
        raise ValueError('Length of track point coordinates and track ID arrays must match')

    # Find number of unique tracks based on track number
    unique_track_ids = sorted(np.unique(track_ids))
    if len(unique_track_ids) - 1 != np.max(unique_track_ids):
        raise ValueError('Number of track IDs not equal to max track ID number')

    # For each ID, get all point coords and store in array
    track_list = []
    for track_id in unique_track_ids:
        track_rows = track_ids == track_id
        track_list.append(track_points[track_rows])

    return track_list


def simple_spatial_match(true_point_lists, pred_point_lists, matcher_config):
    """Match tracks based on spatial overlap between sets of track points

    Parameters
    ----------
    true_points: list
        List of arrays where each array contains one ground truth track's
        (X, Y, T) points
    pred_points: list
        List of arrays where each array contains one proposed track's (X, Y, T)
        points
    matcher_config: dict
        Configuration parameters (`track_eval_dist_threshold` and
        `track_association_overlap_threshold`)

    Returns
    -------
    matches: dict
        Keys represent matches of the form (true index, predicted index)
        Values represent the number of matched points
    """

    # Generate KD trees from points and get number of tracks
    true_kdtrees = [KDTree(point_list) for point_list in true_point_lists]
    pred_kdtrees = [KDTree(point_list) for point_list in pred_point_lists]

    # Get number of points per track
    true_track_lengths = [tree.n for tree in true_kdtrees]
    pred_track_lengths = [tree.n for tree in pred_kdtrees]

    # Calculate track match neighbors (i.e., number of matching points)
    dist_thresh = float(matcher_config['track_eval_dist_threshold'])
    neighbors_arr = calc_kdtree_neighbors(true_kdtrees, pred_kdtrees,
                                          dist_thresh)

    # From track point neighbors, associate predicted tracks to labeled ones
    assoc_thresh = matcher_config['track_association_overlap_threshold']
    match_arr = calc_pred_track_matches(neighbors_arr, pred_track_lengths,
                                        assoc_thresh)

    # Map match indices to number of matched points
    matches = {}
    for match in match_arr:
        true_index = match[0]
        pred_index = match[1]
        matches[(true_index, pred_index)] = neighbors_arr[true_index, pred_index]

    return matches

def calc_kdtree_neighbors(true_kdtrees, pred_kdtrees, dist_thresh):
    """Find number of points for each predicted track that met the distance
    threshold to a labeled track

    Parameters
    ----------
    true_kdtrees: list of scipy.spatial.KDTree
        KDTrees containing points of labeled tracks.
    pred_kdtrees: list of scipy.spatial.KDTree
        KDTrees containing points of predicted tracks.
    dist_thresh: float
        Spatiotemporal distance threshold for matching points.

    Returns
    -------
    neighbors_arr: numpy.ndarray
        Number of points in each predicted track that met the distance threshold
        with at least 1 point in the label track. Shape is
        n_true_trees x n_pred_trees
    """
    if not dist_thresh > 0:
        raise ValueError('`dist_thresh` must be positive.')
    if isinstance(true_kdtrees, KDTree):
        true_kdtrees = [true_kdtrees]
    if isinstance(pred_kdtrees, KDTree):
        true_kdtrees = [pred_kdtrees]

    # Initialize neighbors array
    neighbors_arr = np.zeros((len(true_kdtrees), len(pred_kdtrees)))

    # Iterate over all pairs of trees
    for ti, true_tree in enumerate(true_kdtrees):
        for pi, pred_tree in enumerate(pred_kdtrees):
            # Find number of points in each pred tree that match with each true tree
            neighbors_arr[ti, pi] = unique_overlapping_pts(pred_tree, true_tree,
                                                           dist=dist_thresh)

    return neighbors_arr


def calc_pred_track_matches(neighbors_arr, pred_track_lengths,
                            proportion_overlap):
    """Convert track overlap metrics to a binary list for performance calculations

    Parameters
    ----------
    neighbors_arr: numpy.ndarray
        Array of shape n_true_tracks x n_pred_tracks specifying how many unique
        points were matched between each pair of tracks.
    pred_track_lengths: list of int
        Length of each predicted track in terms of number of points
    proportion_overlap: float
        Float between 0 and 1 specifying proportion of the predicted track (in
        terms of number of points) that must be matched with a ground truth
        track.

    Returns
    -------
    matches: numpy.ndarray
        Array containing all track matches between the predicted and label track
        sets. Shape is n_matches x 2. The two column indicies correspond to the
        row/col in `neighbors_arr`, which should means the 0th column should
        contain the true track index and the 1st column should contain the
        predicted track index.
    """

    if not isinstance(pred_track_lengths, list):
        pred_track_lengths = [pred_track_lengths]
    if not 0 < proportion_overlap <= 1.:
        raise ValueError('The `proportion_overlap` must like on interval (0, 1].'
                         f' Got {proportion_overlap}')
    if len(pred_track_lengths) != neighbors_arr.shape[1]:
        raise ValueError(f'Number of predicted tracks ({len(pred_track_lengths)})'
                         f' must match 2nd dim of `neighbors_arr` ({neighbors_arr.shape[1]})')

    matches = []
    # Find best candidate overlap
    prop_overlap = neighbors_arr / np.atleast_2d(pred_track_lengths)
    candidate_match_inds = np.argmax(prop_overlap, axis=0)

    # Make sure overlap is greater than config threshold
    # Each match in `matches` is (true_track_idx, pred_track_idx)
    for pi, ti in enumerate(candidate_match_inds):
        if prop_overlap[ti, pi] >= proportion_overlap:
            matches.append((ti, pi))

    return np.array(matches)


def unique_overlapping_pts(tree1, tree2, dist):
    """Calculate number of unique points overlapping between two KDTrees

    Note: order of trees matters. For track datasets, it's unlikely that
    `unique_overlapping_pts(tree_a, tree_b, dist)` will equal
    `unique_overlapping_pts(tree_b, tree_a, dist)`

    Parameters
    ----------
    tree1: scipy.spatial.KDTree
        KDTree containing set of points. Overlapping points will be calculated
        in refernece to this tree.
    tree2: scipy.spatial.KDTree
        KDTree containing set of points that serves as the query tree.
    dist: float
        Maximum Euclidean distance between points to designate them as a match.

    Returns
    -------
    n_overlap_pts: int
        Number of points in tree1 that had at least one point in tree2 within
        distance `dist`.
    """

    # Get list of points in tree2 that overlap with each point in tree1
    overlap = tree1.query_ball_tree(tree2, dist)

    # Return number of points in tree1 that overlap with at least 1 point in tree2
    return np.sum([pt_list != [] for pt_list in overlap])
