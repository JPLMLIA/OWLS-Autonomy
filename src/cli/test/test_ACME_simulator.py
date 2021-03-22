import pytest
import numpy as np
import os

from cli.ACME_simulator import *

@pytest.mark.skip
def test_plot_exp():
    exp = np.zeros((11,11))
    exp[5,5] = 1
    save = True
    save_path = 'data/test_plot_for_plot_exp'
    plot_exp(exp,save=save, save_path=save_path)
    #check if plot exists
    assert os.path.isfile('data/test_plot_for_plot_exp.png')


def test_add_peak():
    exp = np.zeros((11,11))
    peaks = {'time_idx':[5], 'mass_idx': [6], 'mass_width_idx': [5],'time_width_idx': [5],'height': [3]}
    peaks = pd.DataFrame(data=peaks)
    for peak in peaks.itertuples():
        exp, volume = add_peak(exp, peak)
    # volume should be approx. 2*pi*peak.height*sigma_x*sigma_y = 13.1041 (https://en.wikipedia.org/wiki/Gaussian_function)
    assert volume < 13.5
    assert volume > 12.5

    sum_exp = np.sum(exp)
    assert sum_exp < 13.5
    assert sum_exp > 12.5

    slice = np.round(exp[6,:],3)
    expected_slice = np.array([0., 0., 0.005, 0.168, 1.46 , 3., 1.46 , 0.168, 0.005, 0., 0.])
    np.testing.assert_array_equal(slice, expected_slice)

    slice = np.round(exp[:,5],3)
    expected_slice = np.array([0., 0., 0., 0.005, 0.168, 1.46, 3., 1.46, 0.168,0.005, 0.])
    np.testing.assert_array_equal(slice, expected_slice)


    exp = np.zeros((11,11))
    peaks = {'time_idx':[5], 'mass_idx': [6], 'mass_width_idx': [5],'time_width_idx': [3],'height': [3]}
    peaks = pd.DataFrame(data=peaks)
    for peak in peaks.itertuples():
        exp, volume = add_peak(exp, peak) # volume should be approx 7.862

    assert volume > 7.5
    assert volume < 8.5

    assert np.max(exp) == 3

    exp = np.ones((11,11))
    peaks = {'time_idx':[5], 'mass_idx': [6], 'mass_width_idx': [5],'time_width_idx': [3],'height': [3]}
    peaks = pd.DataFrame(data=peaks)
    for peak in peaks.itertuples():
        exp, volume = add_peak(exp, peak)

    assert volume > 7.5 # volume should be approx 7.862
    assert volume < 8.5

    sum_exp = np.sum(exp) #should equal volume + 11*11

    assert sum_exp > 128.5
    assert sum_exp < 129.5

@pytest.mark.skip
def test_add_stripe():
    exp = np.zeros((11,101))
    stripes = {'stripe_noise': [10], 'stripe_offset': [100], 'stripe_width': [3],'stripe_mass_idx': [5]}
    stripes = pd.DataFrame(data=stripes)
    cliffs = np.array([0,30,60,101])
    for stripe in stripes.itertuples():
        exp = add_stripe(exp, stripe, cliffs)

    max_exp = np.max(exp)

    assert max_exp < 150
    assert max_exp > 100

    min_exp = np.min(exp)
    assert min_exp <= 0


def test_add_background_offset():
    exp = np.zeros((11,101))
    background_offsets = np.array([5,50,5])
    cliffs = np.array([0,30,60,101])
    exp = add_background_offset(exp, background_offsets, cliffs)

    slice = exp[:5,50]
    expected_slice = np.array([50., 50., 50., 50., 50.])

    np.testing.assert_array_equal(slice, expected_slice)

    slice = exp[:5,80]
    expected_slice = np.array([5., 5., 5., 5., 5.])

    np.testing.assert_array_equal(slice, expected_slice)

#TODO add test for main program of simulator