"""
Tests for the functionality that simulates tracks.
"""
import numpy as np
from numpy.testing import assert_array_equal

from helm_dhm.simulator.sim_tracks import (TrackGeneratorFromDists, TrackGeneratorFromVAR, get_random_start_pos,
                                           get_valid_window_inds)

# Test Track creation and core functionality
class TestTrackSim:
    def test_dist_based_track_generator(self):
        """Test addition of data during new time steps"""
        start_pos = [100, 100]
        momentum = [0, 0]
        movement_dist_dict = dict(distribution_name='truncnorm',
                                  mean=5, std=1, min=0, max=10)

        track_gen = TrackGeneratorFromDists(start_pos, movement_dist_dict, momentum, True)
        track_gen.time_steps()
        assert len(track_gen.times) == 2
        assert len(track_gen.pos) == 2
        assert len(track_gen.vel) == 1

        track_gen.time_steps(8)
        assert len(track_gen.times) == 10
        assert len(track_gen.pos) == 10
        assert len(track_gen.vel) == 9

    def test_var_based_track_generator(self):
        """Test addition of data during new time steps"""
        start_pos = [100, 100]
        model_fpath = "../var_models/nov_2019_chlamy_motile.pickle"

        track_gen = TrackGeneratorFromVAR(start_pos, model_fpath, True, 0)
        track_gen.time_steps()
        assert len(track_gen.times) == 2
        assert len(track_gen.pos) == 2
        assert len(track_gen.vel) == 1

        track_gen.time_steps(8)
        assert len(track_gen.times) == 10
        assert len(track_gen.pos) == 10
        assert len(track_gen.vel) == 9

    # Test valid spatial positions
    def test_valid_bounds(self):

        pos = np.array([[1, 1], [1, 3], [5, 5], [2, 2]])
        valid_row_bounds = [0, 4]
        valid_col_bounds = [0, 4]

        valid_inds = get_valid_window_inds(valid_row_bounds, pos[:, 0],
                                           valid_col_bounds, pos[:, 1])
        print(valid_inds)
        assert_array_equal(valid_inds,
                           np.array([0, 2]))

    # Test start position dimensions
    def test_random_start_position(self):
        """Make sure we get 2 and 3 dimensional start positions as expected"""

        pos_2D = get_random_start_pos(img_res=[2, 2], img_buffer=[2, 2])
        pos_3D = get_random_start_pos(img_res=[2, 2, 2], img_buffer=[2, 2],
                                      chamber_depth=2)
        assert len(pos_2D) == 2
        assert len(pos_3D) == 3
