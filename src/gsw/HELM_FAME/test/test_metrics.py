"""
Unit tests for metrics.py
"""
from pytest import raises
import numpy as np
from numpy.testing import assert_array_equal
from scipy.spatial import KDTree

from helm_dhm.evaluation.track_metrics import (calc_pred_track_matches,
                                               calc_kdtree_neighbors,
                                               unique_overlapping_pts)
from helm_dhm.evaluation.point_metrics import (detect_point_matches,
                                               convert_point_matches)

# Test arrays specifying points w/ (coord_1, coord_2, time)
series_a = np.array([[0., 0., 0.],
                     [1., 0., 1.],
                     [2., 0., 2.]])
series_b = np.array([[0., 0., 0.],
                     [1., 0., 0.],
                     [3., 0., 1.],
                     [8., 0., 3.],
                     [0., 0., 3.]])

class TestPointMetrics:
    def test_point_matrix(self):
        """Test `detect_point_matches` with true and proposed points"""

        matches = detect_point_matches(series_a, series_b)

        correct_matches = np.array([[True, True, False, False, False],
                                    [False, False, True, False, False],
                                    [False, False, False, False, False]])
        assert_array_equal(matches, correct_matches)


    def test_point_match_conversion(self):
        """Test conversion of point matches for metric calcs"""

        matches = detect_point_matches(series_a, series_b)
        y_true, y_pred = convert_point_matches(matches)

        assert_array_equal(y_true, np.array([True, True, True, False, False]))
        assert_array_equal(y_pred, np.array([True, True, False, True, True]))


class TestTrackMetrics:

    def test_overlapping_points(self):
        """Test ability to calculate number of point neighbors"""

        tree_a = KDTree(series_a)
        tree_b = KDTree(series_b)

        assert unique_overlapping_pts(tree_a, tree_b, 0.01) == 1
        assert unique_overlapping_pts(tree_a, tree_b, 1) == 2
        assert unique_overlapping_pts(tree_a, tree_b, 2) == 3
        assert unique_overlapping_pts(tree_b, tree_a, 2) == 3

    def test_neighbor_calc(self):
        """Test ability to create KDTrees and find point neighbors"""
        tree_a = KDTree(series_a)
        tree_b = KDTree(series_b)

        with raises(ValueError):
            calc_kdtree_neighbors(tree_a, tree_b, -1)

        # Test large distance threshold
        neighbors_arr = calc_kdtree_neighbors([tree_a, tree_b], [tree_a, tree_b], 10)
        assert_array_equal(neighbors_arr, np.array([[3, 5],
                                                    [3, 5]]))

        # Test tight distance threshold
        neighbors_arr = calc_kdtree_neighbors([tree_a, tree_b], [tree_a, tree_b], 2)
        assert_array_equal(neighbors_arr, np.array([[3, 3],
                                                    [3, 5]]))


    def test_track_metrics(self):
        """Test ability to create KDTrees and find point neighbors"""
        tree_a = KDTree(series_a)
        tree_b = KDTree(series_b)

        neighbors_arr = calc_kdtree_neighbors([tree_a, tree_b], [tree_a, tree_b], 1)
        matches = calc_pred_track_matches(neighbors_arr, [tree_a.n, tree_b.n], 0.8)
        assert_array_equal(matches, np.array([[0, 0],
                                              [1, 1]]))

        neighbors_arr = calc_kdtree_neighbors([tree_a, tree_b], [tree_a, tree_b], 10)
        print(neighbors_arr)
        matches = calc_pred_track_matches(neighbors_arr, [tree_a.n, tree_b.n], 0.8)
        assert_array_equal(matches, np.array([[0, 0],
                                              [0, 1]]))
