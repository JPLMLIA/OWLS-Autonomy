
"""
Unit tests for reporting.py
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose
import os.path as op
import os

from helm_dhm.evaluation.reporting import point_score_report, track_score_report
from helm_dhm.evaluation.reporting import plot_metrics_hist


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

    match_arr = np.array([[0, 0],
                          [0, 1],
                          [2, 3]])
    n_true_tracks = 3
    n_pred_tracks = 4

    report = track_score_report(match_arr, n_true_tracks, n_pred_tracks)
    print(report)

    assert_allclose(report['prop_true_tracks_matched'], 2/3)
    assert_allclose(report['prop_pred_tracks_matched'], 0.75)

    # Test if track_score_report() works with empty match array
    report = track_score_report(np.array([]), n_true_tracks, n_pred_tracks)
    assert_allclose(report['prop_true_tracks_matched'], 0)
    assert_allclose(report['prop_pred_tracks_matched'], 0)

    # Test if track_score_report() can handle zero true/pred tracks
    report = track_score_report(match_arr, 0, 0)
    assert_allclose(report['prop_true_tracks_matched'], np.inf)
    assert_allclose(report['prop_pred_tracks_matched'], np.inf)

    # Test if track_score_report() can handle both empty match array and zero tracks
    report = track_score_report(np.array([]), 0, 0)
    assert_allclose(report['prop_true_tracks_matched'], np.inf)
    assert_allclose(report['prop_pred_tracks_matched'], np.inf)

@pytest.mark.skip
def test_plot_metrics():
    '''Test batch plot creation'''

    tm = 'test_metric'
    data = [{tm : i*i/100.0} for i in range(100)]
    out_dir = op.join(os.getcwd(), "test_out")
    out_path = op.join(out_dir, tm + ".png")
    if op.exists(out_path):
        os.remove(out_path)
    plot_metrics_hist(data, [tm], 20, out_dir)
    assert(op.exists(out_path))
