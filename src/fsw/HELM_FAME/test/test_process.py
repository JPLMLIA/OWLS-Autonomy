import os
import os.path as op
from numpy.testing._private.utils import assert_array_almost_equal
import pytest
import glob
import tempfile

import numpy as np
from numpy.testing import assert_array_equal

from helm_dhm.validate import process, preproc

class TestImageChecks:
    """Check if bad images are correctly filtered"""

    def test_valid(self):
        target_res = (2048, 2048)
        image = op.join(op.dirname(op.realpath(__file__)), 'data', '00001_holo.tif')
        assert process.is_valid_image(image, target_res)

    def test_invalid(self):
        target_res = (2048, 2048)
        other_res = (123, 123)
        image = op.join(op.dirname(op.realpath(__file__)), 'data', '00001_holo.tif')
        bad_image = op.join(op.dirname(op.realpath(__file__)), 'data', 'invalid_tif.tif')
        assert process.is_valid_image(image, other_res) == False
        assert process.is_valid_image(bad_image, target_res) == False

class TestGetFiles:
    '''Check that the correct set of holograms is found'''

    def test(self):
        experiment = op.dirname(op.realpath(__file__))
        config = {'hologram_file_extensions': ['.tif'],
                  'min_holograms': 5,
                  'raw_hologram_resolution': [2048, 2048],
                  'experiment_dirs':{'hologram_dir': 'data'}}
        assert(len(process.get_files(experiment, config)) == 10)

class TestGetExperiments:
    """Check if experiments are properly filtered"""

    def test_valid(self):
        experiment = op.dirname(op.realpath(__file__))
        config = {'hologram_file_extensions': ['.tif'],
                  'min_holograms': 5,
                  'raw_hologram_resolution': [2048, 2048],
                  'experiment_dirs':{'hologram_dir': 'data'},
                  'validate':{'min_holograms': 5}}
        assert(process.get_experiments([experiment], config))

    def test_not_enough_hols(self):
        experiment = op.dirname(op.realpath(__file__))
        config = {'hologram_dir': 'data',
                  'hologram_file_extensions': ['.tif'],
                  'min_holograms': 200,
                  'raw_hologram_resolution': [2048, 2048],
                  'experiment_dirs':{'hologram_dir': 'data'},
                  'validate':{'min_holograms': 15}}
        assert(not process.get_experiments([experiment], config))

    def test_bad_resolution(self):
        experiment = op.dirname(op.realpath(__file__))
        config = {'hologram_dir': 'data',
                  'hologram_file_extensions': ['.tif'],
                  'min_holograms': 3,
                  'raw_hologram_resolution': [12345, 100],
                  'experiment_dirs':{'hologram_dir': 'data'},
                  'validate':{'min_holograms': 5}}
        assert(not process.get_experiments([experiment], config))

    def test_multi(self):
        experiment = op.dirname(op.realpath(__file__))
        config = {'hologram_dir': 'data',
                  'hologram_file_extensions': ['.tif'],
                  'min_holograms': 5,
                  'raw_hologram_resolution': [2048, 2048],
                  'experiment_dirs':{'hologram_dir': 'data'},
                  'validate':{'min_holograms': 5}}
        experiments = ['wont/match/anything',
                       experiment,
                       experiment]
        assert(len(process.get_experiments(experiments, config)) == 1)


class TestPreprocCalcs:

    def test_median_calculation(self):
        all_images = glob.glob(op.join(op.dirname(op.realpath(__file__)),
                                       'data', '*_holo.tif'))

        with tempfile.TemporaryDirectory() as temp_dir:

            # Resize from 2048x2048 as is done in the main pipeline
            preproc.resize_holograms(all_images, temp_dir, (1024, 1024))

            # Get resized images and compute median image
            resized_images = glob.glob(op.join(temp_dir, '*_holo.tif'))
            med_image = process.calc_median_image({'file_batch': resized_images})

            # Check against ground truth
            gt_med_image = np.load(op.join(op.dirname(op.realpath(__file__)),
                                           'data', 'test', 'gt_med_image.npy'))
            assert_array_almost_equal(med_image, gt_med_image)
