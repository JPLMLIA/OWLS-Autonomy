import pytest
import glob
import tempfile
import pytest

from pandas.testing import assert_series_equal
from pandas.testing import assert_frame_equal

import pandas            as pd
import matplotlib.pyplot as plt
import numpy             as np

from helm_dhm.features.features import *


def test_plot_movie_tracks(hl_test_case):

    hl_df, dataset_name = hl_test_case
    output_dir = tempfile.mkdtemp()
    plot_movie_tracks(hl_df, dataset_name, output_dir)
    assert(len(glob.glob(os.path.join(output_dir, '*.png'))) == 1)


CLEAN_FEATURE_DF_PARAMS = [(pd.DataFrame({'movement_type': [],
                                          'size': [],
                                          'autoCorr_vel_lag1': [],
                                          'autoCorr_stepAngle_lag1': []}),
                            pd.DataFrame({'movement_type': [],
                                          'size': [],
                                          'autoCorr_vel_lag1': [],
                                          'autoCorr_stepAngle_lag1': []})),
                           (pd.DataFrame({'movement_type':           ['Motile', 'Motile', 'Non-motile', 'Non-motile'],
                                          'size':                    ['small', 'small', 'small', 'small'],
                                          'autoCorr_vel_lag1':       [1.0 , np.nan, np.nan, 1.0],
                                          'autoCorr_stepAngle_lag1': [1.0, np.nan, np.nan, 1.0]}),
                            pd.DataFrame({'movement_type': ['motile', 'non-motile'],
                                          'size': ['Small', 'Small'],
                                          'autoCorr_vel_lag1': [1.0, 1.0],
                                          'autoCorr_stepAngle_lag1': [1.0, 1.0]}, index=[0, 3]))]


@pytest.mark.skip
# skipping because plotter.get_type() is partially commented out so it won't function properly.
def test_Plotter(data_track_features):
    plotter = Plotter(data_track_features)
    assert(plotter.get_type() == 'simulated')
    plotter.plot_by_type('autoCorr_vel_lag1', 'mean_stepAngle')
    plt.close()

    plotter.plot_dir = tempfile.mkdtemp()
    plotter.plot_feat_i_vs_j('autoCorr_vel_lag1', 'mean_stepAngle')
    assert(len(glob.glob(os.path.join(plotter.plot_dir, '*.png'))) == 1)


def test_estimated_autocorrelation():
    x = [1, 2, 3, 4, 5]
    assert(estimated_autocorrelation(x) == [1.0, 0.4, -0.1, -0.4, -0.4])

    x = [2, 3, 'NaN', 3, 3]
    with pytest.raises(TypeError) as excinfo:
        estimated_autocorrelation(x)
    assert(str(excinfo.value) == 'Input array values must be only int or float type')


GET_TRACK_ABS_FEATURES_PARAMS = [([1, 2, 3, -1], [2, 3, 7, 10], [0, 1, 2, 3], 4,
                                  'test_movie', 1,
                                  {'track_length': 10.537319187990756, 'max_velocity': 5.0,
                                   'mean_velocity': 3.512439729330252, 'stdev_velocity': 1.8692647543174938,
                                   'autoCorr_vel_lag1': -0.05336244292053344, 'autoCorr_vel_lag2': -0.4466375570794665,
                                   'max_stepAngle': 4.31386653471827, 'mean_stepAngle': 3.9979393442893234,
                                   'autoCorr_stepAngle_lag1': np.inf, 'autoCorr_stepAngle_lag2': np.inf,
                                   'max_accel': 2.7088920632445657, 'mean_accel': 1.7928932188134525,
                                   'stdev_accel': np.inf, 'autoCorr_accel_lag1': np.inf, 'autoCorr_accel_lag2': np.inf,
                                   'ud_x': -0.24253562503633297, 'ud_y': 0.9701425001453319,
                                   'theta_displacement': 1.8157749899217608},
                                  [10.537319187990756, 5.0, 3.512439729330252, 1.8692647543174938, -0.05336244292053344,
                                   -0.4466375570794665, 4.31386653471827, 3.9979393442893234, np.inf, np.inf,
                                   2.7088920632445657, 1.7928932188134525, np.inf, np.inf, np.inf,
                                   -0.24253562503633297, 0.9701425001453319, 1.8157749899217608])]


def test_get_track_relative_features(data_track_features_df):

    test_case = data_track_features_df.copy()
    test_case = test_case.drop(['rel_vel', 'rel_theta_displacement', 'rel_dir_dot'], axis=1)
    test_case = get_track_relative_features(test_case)
    assert_series_equal(data_track_features_df['rel_vel'], test_case['rel_vel'])
    assert_series_equal(data_track_features_df['rel_theta_displacement'], test_case['rel_theta_displacement'])
    assert_series_equal(data_track_features_df['rel_dir_dot'], test_case['rel_dir_dot'])
