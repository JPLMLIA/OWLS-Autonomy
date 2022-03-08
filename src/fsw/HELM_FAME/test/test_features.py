"""
Unit tests for features.py
"""
from pytest import raises

import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_array_almost_equal

from helm_dhm.features.features import SUBSTITUTE_FEAT_DICT
from helm_dhm.features.absolute_features import (speed_and_acceleration,
                                                 local_angle_2D, displacement,
                                                 msd_slope)
from helm_dhm.features.relative_features import (relative_speeds,
                                                 relative_direction_feats)
from helm_dhm.features.feature_utils import (normed_autocorr,
                                             compute_msd,
                                             substitute_feats)


# Test tracks for angle calcs (in matrix coords here)
track_a = np.array([[0, 0], [1, 1], [0, 2]])  # Test wrapping around 0 radians. One 90 degree turn
track_b = np.array([[0, 0], [-1, -1], [0, -2]])  # Test wrapping around π radians One 90 degree turn
track_c = np.array([[0, 0], [0, 0], [1, 1], [0, 2], [0, 3]])
track_d = np.array([[0, 0], [1, 1], [1, 2], [2, 3]])
track_stationary = np.array([[1, 1], [1, 1], [1, 1]])  # Particle that sits still
track_circular = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])  # Particle following a square path; no net movement

class TestAbsoluteFeatureCalcs:

    def test_speed_acc(self):
        """Test basic speed and acceleration calculations"""
        sa_feats = speed_and_acceleration(track_stationary)
        assert_array_equal(sa_feats['track_length'], 0)
        assert_array_equal(sa_feats['speed_mean'], 0)
        assert_array_equal(sa_feats['accel_mean'], 0)
        assert_array_equal(sa_feats['accel_max'], 0)

        sa_feats = speed_and_acceleration(track_c, times=[0, 0.5, 1, 1.5, 2])
        assert_array_equal(sa_feats['track_length'], 1+2*np.sqrt(2))
        assert_array_equal(sa_feats['speed_mean'], ((1+2*np.sqrt(2))/0.5)/4)
        assert_array_equal(sa_feats['accel_mean'], 4*(2*np.sqrt(2) - 1)/3)
        assert_array_equal(sa_feats['accel_max'], 4*np.sqrt(2))

    def test_angle_wrap(self):
        """Make sure that step angles are correctly determined (including wrapping)"""

        feats = local_angle_2D(track_a)
        assert_array_almost_equal(feats['step_angle_mean'], np.pi / 2)

        feats = local_angle_2D(track_b)
        assert_array_almost_equal(feats['step_angle_mean'], np.pi / 2)

        feats = local_angle_2D(track_c)
        assert_array_almost_equal(feats['step_angle_mean'], np.pi/3)
        assert_equal(feats['step_angle_max'], np.pi/2)
        assert_array_almost_equal(feats['step_angle_stdev'], 0.3702402448465305)

        feats = local_angle_2D(track_stationary)
        assert_equal(feats['step_angle_mean'], 0)

    def test_sinuosity(self):
        assert_equal(displacement(track_circular)['sinuosity'], np.inf)
        assert_equal(displacement(track_stationary)['sinuosity'], np.inf)
        assert_equal(displacement(track_b)['sinuosity'], (2 * np.sqrt(2)) / 2)

    def test_basic_displacement(self):
        disp_feats = displacement(track_stationary)  # Stationary track
        assert_equal(disp_feats['disp_mean_h'], 0)
        assert_equal(disp_feats['disp_e2e_h'], 0)
        assert_equal(disp_feats['disp_mean_v'], 0)
        assert_equal(disp_feats['disp_e2e_v'], 0)
        assert_equal(disp_feats['disp_angle_e2e'], np.arctan2(0, 0))

        disp_feats = displacement(track_d)  #
        assert_equal(disp_feats['disp_mean_h'], 1)
        assert_equal(disp_feats['disp_e2e_h'], 3)
        assert_equal(disp_feats['disp_mean_v'], 2/3)
        assert_equal(disp_feats['disp_e2e_v'], 2)
        assert_equal(disp_feats['disp_angle_e2e'], np.arctan2(-2, 3))

    def test_msd_slope(self):
        assert_equal(msd_slope(track_stationary)['msd_slope'], 0)

        slope = msd_slope(track_stationary, tau_interval=0.5, flow_offset=(1, 1))['msd_slope']
        assert_equal(slope, 12)


autocorr_a = [0, -1, 2, -1, 0]
autocorr_b = [1, 1, 1, 1, 1]

class TestFeatureUtils:
    def test_autocorr(self):
        acf_a = normed_autocorr(autocorr_a)
        assert_array_equal(acf_a, [1, -2/3, 1/6, 0, 0])

        acf_b = normed_autocorr(autocorr_b, [0, 1])
        assert_array_equal(acf_b, [np.inf, np.inf])

    def test_msd(self):
        lags, msds, msds_stdevs = compute_msd(track_stationary, tau=1., flow_offset=(0, 0))
        assert_array_equal(lags, [1, 2])
        assert_array_equal(msds, [0, 0])
        assert_array_equal(msds_stdevs, [0, 0])

        lags, msds, msds_stdevs = compute_msd(track_d, tau=1., flow_offset=(0, 0))
        assert_array_equal(lags, [1, 2, 3])
        assert_array_equal(msds, [5/3, 5, 13])
        assert_array_almost_equal(msds_stdevs, [0.4714045207910317, 0, 0])

        lags, msds, msds_stdevs = compute_msd(track_d, tau=0.5, flow_offset=(0, 0))
        assert_array_equal(lags, [0.5, 1., 1.5])
        assert_array_equal(msds, [5/3, 5, 13])
        assert_array_almost_equal(msds_stdevs, [0.4714045207910317, 0, 0])

        # Test substitutions with a vector that has no displacement
        track_feats = relative_direction_feats(test_dirs_b)[0]
        assert_array_equal(track_feats['rel_step_angle'], np.inf)  # Unsubstituted data
        new_track_feats = substitute_feats(track_feats, SUBSTITUTE_FEAT_DICT)
        assert_array_equal(new_track_feats['rel_step_angle'], 0)  # Substituted data


test_speeds_a = [1, 2, 3, 4]
test_speeds_b = [1, 0]

test_dirs_a = [(1, 1), (1, 1), (1, 1)]
test_dirs_b = [(1, 1), (0, 0)]
test_dirs_c = [(1, 1), (0, 1), (1, 0)]

class TestRelativeFeats:

    def test_relative_speed(self):

        # Test some reasonable speeds
        feats = relative_speeds(test_speeds_a)
        feat_vals = [d['rel_speed'] for d in feats]
        assert_array_equal(feat_vals, [1/(9/3), 2/(8/3), 3/(7/3), 4/(6/3)])  # Speed / mean of other speeds

        # Test that relative calculations can handle 0 mean speeds
        feats = relative_speeds(test_speeds_b)
        feat_vals = [d['rel_speed'] for d in feats]
        assert_array_equal(feat_vals, [np.inf, 0])

    def test_relative_direction(self):

        # Test 3 displacement vectors going same direction
        feats = relative_direction_feats(test_dirs_a)
        cosine_similarities = [track['rel_disp_cosine_similarity'] for track in feats]
        rel_angles = [track['rel_step_angle'] for track in feats]
        assert_array_almost_equal(cosine_similarities, [1, 1, 1])
        assert_array_almost_equal(rel_angles, [0, 0, 0])

        # Test with one vector that has no displacement
        feats = relative_direction_feats(test_dirs_b)
        cosine_similarities = [track['rel_disp_cosine_similarity'] for track in feats]
        rel_angles = [track['rel_step_angle'] for track in feats]
        assert_array_equal(rel_angles, [np.inf, np.inf])

        # Test with 3 reasonable displacement vectors
        feats = relative_direction_feats(test_dirs_c)
        cosine_similarities = [track['rel_disp_cosine_similarity'] for track in feats]
        rel_angles = np.array([track['rel_step_angle'] for track in feats])
        assert_array_almost_equal(rel_angles, [0,
                                               np.pi/2 - np.arctan2(0.5, 1),
                                               np.arctan2(1, 0.5)])
        assert_array_almost_equal(cosine_similarities, [1, 0.5/np.sqrt(1.25), 0.5/np.sqrt(1.25)])