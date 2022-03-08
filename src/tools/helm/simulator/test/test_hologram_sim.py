"""
Tests for the functionality that simulates holograms.
"""
from copy import copy

import pytest

from helm_dhm.simulator.utils import (create_dist_objs, distribution_check,
                                      VALID_CONFIG_DISTS, VALID_CONFIG_SHAPES)
from helm_dhm.simulator.sim_holograms import get_kernel


class TestHologramSim:

    def test_get_kernel(self):
        """Test that we can get all particle kernels"""
        for kernel_name in VALID_CONFIG_SHAPES:
            kernel = get_kernel(kernel_name, 2)
            assert kernel is not None

    def test_distributions(self):
        """Test that we can create distribution objects (and catch bad configs)"""

        good_dist_dict = dict(mean=0, std=1, min=-1, max=1)
        bad_dist_dict = dict(mean=0, std=1, min=-1, distribution_name='invalid')

        # Check we can generate valid distribution objects for sampling
        for dist in VALID_CONFIG_DISTS:
            full_good_dist_dict = copy(good_dist_dict)
            full_good_dist_dict.update({'distribution_name': dist})
            distribution_check(full_good_dist_dict, 1)

            dist_obj = create_dist_objs(**full_good_dist_dict)
            assert dist_obj[0].rvs() is not None

        # Check for bad distribution name
        with pytest.raises(ValueError):
            distribution_check(bad_dist_dict, 1)
        # Check for bad distribution settings (min > max)
        with pytest.raises(ValueError):
            bad_dist_dict['distribution_name'] = VALID_CONFIG_DISTS[0]
            bad_dist_dict['max'] = -5
            distribution_check(bad_dist_dict, 1)
