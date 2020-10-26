import os
import tempfile

import numpy as np

from utils.hdf5_helper import h5_util


def test_save_h5(h5_util_test_dicts):
    # Set up temporary testing directory
    output_dir = tempfile.mkdtemp()

    test_case_1, test_case_2 = h5_util_test_dicts

    # First test case (dictionary of long doubles)
    h5_util.save_h5(os.path.join(output_dir, 'test_case_1.hdf5'), test_case_1)
    assert (os.path.exists(os.path.join(output_dir, 'test_case_1.hdf5')))

    # Second test case (dictionary of dictionaries)
    h5_util.save_h5(os.path.join(output_dir, 'test_case_2.hdf5'), test_case_2)
    assert (os.path.exists(os.path.join(output_dir, 'test_case_2.hdf5')))

    # Third test case (empty dict)
    test_case = {}
    h5_util.save_h5(os.path.join(output_dir, 'test_case_3.hdf5'), test_case)
    assert (os.path.exists(os.path.join(output_dir, 'test_case_3.hdf5')))


def test_load_h5(h5_util_test_dicts):
    expected_dict_1, expected_dict_2 = h5_util_test_dicts

    # First test case (dictionary of long doubles)
    test_case = os.path.join(os.path.dirname(__file__), 'data',
                             'load_h5_test_case_1.hdf5')
    loaded_dict = h5_util.load_h5(test_case)
    np.testing.assert_equal(loaded_dict, expected_dict_1)

    # Second test case (dictionary of dictionaries)
    test_case = os.path.join(os.path.dirname(__file__), 'data',
                             'load_h5_test_case_2.hdf5')
    loaded_dict = h5_util.load_h5(test_case)
    np.testing.assert_equal(loaded_dict, expected_dict_2)
