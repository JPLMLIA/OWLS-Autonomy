import os
import os.path as op
import pytest

from numpy.testing import assert_array_equal

from helm_dhm.validate import process

class TestImageChecks:
    """Check if bad images are correctly filtered"""
    
    @pytest.mark.skip
    def test_valid(self):
        target_res = (2048, 2048)
        image = op.join(os.getcwd(), "data/00001_holo.tif")
        assert process.is_valid_image(image, target_res)
        
    def test_invalid(self):
        target_res = (2048, 2048)
        other_res = (123, 123)
        image = op.join(os.getcwd(), "data/00001_holo.tif")
        bad_image = op.join(os.getcwd(), "data/invalid_tif.tif")
        assert process.is_valid_image(image, other_res) == False
        assert process.is_valid_image(bad_image, target_res) == False

class TestGetFiles:
    '''Check that the correct set of holograms is found'''

    @pytest.mark.skip
    def test(self):
        experiment = os.getcwd()
        config = {'hologram_dir': 'data',
                  'hologram_file_extensions': ['.tif'],
                  'min_holograms': 5,
                  'raw_hologram_resolution': [2048, 2048]}
        assert(len(process.get_files(experiment, config)) == 10)

class TestGetExperiments:
    """Check if experiments are properly filtered"""
    @pytest.mark.skip
    def test_valid(self):
        experiment = os.getcwd()
        config = {'hologram_dir': 'data',
                  'hologram_file_extensions': ['.tif'],
                  'min_holograms': 5,
                  'raw_hologram_resolution': [2048, 2048]}
        assert(process.get_experiments([experiment], config))
    
    @pytest.mark.skip
    def test_not_enough_hols(self):
        experiment = os.getcwd()
        config = {'hologram_dir': 'data',
                  'hologram_file_extensions': ['.tif'],
                  'min_holograms': 200,
                  'raw_hologram_resolution': [2048, 2048]}
        assert(not process.get_experiments([experiment], config))
        
    @pytest.mark.skip
    def test_bad_resolution(self):
        experiment = os.getcwd()
        config = {'hologram_dir': 'data',
                  'hologram_file_extensions': ['.tif'],
                  'min_holograms': 3,
                  'raw_hologram_resolution': [12345, 100]}
        assert(not process.get_experiments([experiment], config))
        
    @pytest.mark.skip
    def test_hologram_dir(self):
        experiment = os.getcwd()
        config = {'hologram_dir': 'does-not-exist',
                  'hologram_file_extensions': ['.tif'],
                  'min_holograms': 3,
                  'raw_hologram_resolution': [2048, 2048]}
        assert(not process.get_experiments([experiment], config))
        
    @pytest.mark.skip
    def test_multi(self):
        experiment = os.getcwd()
        config = {'hologram_dir': 'data',
                  'hologram_file_extensions': ['.tif'],
                  'min_holograms': 5,
                  'raw_hologram_resolution': [2048, 2048]}
        experiments = ['wont/match/anything',
                       experiment, 
                       experiment]
        assert(len(process.get_experiments(experiments, config)) == 1)

def test_work_dir_setup():
	with pytest.raises(Exception):
		setup.work_dir_setup(basename=None, files=None, clean_dir=None, reprocess=None)

	with pytest.raises(Exception):
		setup.work_dir_setup(basename=None, files= ["HelloWorld"], clean_dir=None, reprocess=None)
