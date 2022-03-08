
"""
Unit tests for reporting.py
"""
import os
import os.path as op

import pytest
import numpy as np
from numpy.testing import assert_allclose

from helm_dhm.evaluation.reporting import point_score_report, track_score_report

y_true = np.array([1, 1, 1, 1, 1])
y_pred = np.array([1, 1, 1, 0, 0])


def test_point_dict_report():
    """Test if precision, recall, etc. are calculated correctly."""

    report = point_score_report(y_true, y_pred)
    print(report)

    assert_allclose(report['precision'], 1.0)
    assert_allclose(report['recall'], 0.6)
    assert_allclose(report['f_1'], 0.75)
    assert_allclose(report['f_0.5'], 0.8823529411764706)
    assert_allclose(report['f_0.25'], 0.9622641509433962)


def test_track_dict_report():
    """Test if precision, recall, etc. are calculated correctly."""

    match_dict = {(0, 0): 5,
                  (0, 1): 2,
                  (2, 3): 3}

    # Dummy set of 3 true tracks and 4 predicted tracks
    true_track_points = [np.array([[0, 0, 0], [0, 1, 1], [0, 2, 2], [0, 3, 3]]),
                         np.array([[1, 0, 0], [2, 0, 1], [3, 0, 2]]),
                         np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])]
    pred_track_points = [np.array([[0, 0, 0], [0, 1, 1]]),  # Overlaps 0th true track
                         np.array([[0, 2, 2], [0, 3, 3]]),  # Overlaps 0th true track
                         np.array([[0, 0, 5], [1, 1, 6], [2, 2, 7]]),  # Dummy track that doesn't match
                         np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])]  # Overlaps 2nd true track

    report = track_score_report(match_dict, true_track_points, pred_track_points,
                                0.75, None, 8)

    # Check precision/recall
    assert_allclose(report['prop_true_tracks_matched'], 2/3)
    assert_allclose(report['prop_pred_tracks_matched'], 0.75)

    # Check metrics that report errors
    report['n_false_positives'] = 1
    report['n_unmatched_points'] = 3
    report['false_track_points_per_frame'] = 3/8

    # Check simple ratios
    report['pred_over_true_ratio'] = 4/3
    report['true_over_pred_ratio'] = 3/4

    # Test if track_score_report() works with empty match array
    report = track_score_report({}, true_track_points, pred_track_points,
                                0.75, None, 8)
    assert_allclose(report['prop_true_tracks_matched'], 0)
    assert_allclose(report['prop_pred_tracks_matched'], 0)

    # Test if track_score_report() can handle both empty match array and zero tracks
    # TODO: Do we want these to be zeros or np.inf (to indicate they should be handled in a special way)?
    report = track_score_report({}, [], [], 0.75, None, 8)
    assert_allclose(report['prop_true_tracks_matched'], 0)
    assert_allclose(report['prop_pred_tracks_matched'], 0)
