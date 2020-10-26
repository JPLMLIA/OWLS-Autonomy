import pytest
import numpy as np


@pytest.fixture
def h5_util_test_dicts():
    # First test case (dictionary of long doubles)
    test_dict_1 = {
        'Particles_Estimated_Position': np.array(
            [[698.0, 744.6],
             [708.0909090909091, 748.6363636363636],
             [721.6666666666667, 739.9166666666666],
             [726.7692307692307, 733.1538461538462],
             [731.8717948717947, 726.3910256410257],
             [736.9743589743587, 719.6282051282053],
             [742.0769230769226, 712.8653846153849],
             [734.0, 722.6]]),
        'Times': np.array([15, 16, 17, 18, 19, 20, 21, 22]),
        'Track_ID': np.array(0)
    }

    # Second test case (dictionary of dictionaries)
    test_dict_2 = {
        '1_1': {'2_1': np.array([1, 2]),
                '2_2': np.array([1, 2, 3]),
                '2_3': np.array([1, 2])},
        '1_2': {'2_1': np.array([135.32, 1423.64]),
                '2_2': np.array([345.9248, 42.42221, 123.5345]),
                '2_3': np.array([12.442, 1324.132])}
    }

    return test_dict_1, test_dict_2
