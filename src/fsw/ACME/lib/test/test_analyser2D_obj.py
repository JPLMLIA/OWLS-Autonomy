# unit testing of analyser2D_obj_vX.py via pytest
#
# Steffen Mauceri
# May 2020
# updated: Aug 2020

import pytest

from acme_cems.lib.analyzer import *
from cli.ACME_flight_pipeline import *

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# test for functions
def test_make_crop():
    peak = np.array([10, 10])
    roi = np.eye(100)
    window_x = 9
    window_y = 3
    center_x = 3
    crop_center, crop_l, crop_r = make_crop(peak,roi,window_x,window_y,center_x)

    expected_crop_center = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    np.testing.assert_array_equal(crop_center, expected_crop_center)
    assert np.mean(crop_l) == 0
    assert np.shape(crop_l) == (3, 3)

    #TODO: add test to get error for even window_x

@pytest.mark.skip
def test_diff_gauss():
    sigma = 2
    ratio = 2.2
    g = diff_gauss(sigma, ratio)
    assert np.shape(g) == (13, 13)
    assert np.sum(g) == 0.9999999999999991
    assert g[0, 0] == g[-1, -1]


def test_filter_by_SNR():
    peaks = np.array([[20,20],[10, 10]])
    data = np.zeros(100) + 10 + np.random.randn(100,100) * 10
    data[20,20] = 100
    data[10,10] = 10
    window_x = 9
    window_y = 3
    center_x = 3
    threshold = 5

    filtered_peak = filter_by_SNR(peaks, data, threshold, window_x, window_y, center_x)
    expected_filtered_peak = np.array([[20, 20]])
    np.testing.assert_array_equal(filtered_peak, expected_filtered_peak)

    threshold = 50
    filtered_peak = filter_by_SNR(peaks, data, threshold, window_x, window_y, center_x)
    assert filtered_peak.size == 0


def test_get_peak_center():
    roi_peaks = np.array([[5,7]])
    exp = np.zeros((10,10))
    exp[5,5] = 1
    w_y = 5
    c = 5
    peak_center = get_peak_center(roi_peaks,exp, w_y,c)
    expected_peak_center = np.array([[5, 5]])
    np.testing.assert_array_equal(peak_center, expected_peak_center)

    roi_peaks = np.array([[4,5]])
    peak_center = get_peak_center(roi_peaks,exp, w_y,c)
    expected_peak_center = np.array([[5, 5]])
    np.testing.assert_array_equal(peak_center, expected_peak_center)

    exp[6,6] = 10
    peak_center = get_peak_center(roi_peaks,exp, w_y,c)
    expected_peak_center = np.array([[6, 6]])
    np.testing.assert_array_equal(peak_center, expected_peak_center)

# test for complete analyzer 2D object
@pytest.mark.skip
def test_analyzer():
    args = Namespace(data = 'acme_cems/lib/test/data/test/500.pickle',
                     masses = 'cli/configs/compounds.yml', 
                     params = 'cli/configs/acme_config.yml',
                     outdir = 'acme_cems/lib/test/data/test/',
                     sue_weights = 'cli/configs/acme_sue_weights.yml',
                     dd_weights = 'cli/configs/acme_dd_weights.yml',
                     noplots = False,
                     noexcel = False,
                     debug_plots = True,
                     cores = None,
                     saveheatmapdata = False,
                     knowntraces = False,
                     reprocess_version = None,
                     space_mode = False,
                     skip_existing = False
    )

    analyse_all_data(vars(args))

    #check output
    output = pd.read_csv('acme_cems/lib/test/data/test/500/Unknown_Masses/500_UM_peaks.csv')
    assert len(output) == 1
    assert abs(output['Peak Amplitude (Counts)'][0] - 500) < 10
    assert abs(output['Mass (idx)'][0] - 500) < 10
    assert output.background_abs.max() < 15
    assert output.background_abs.min() > 5
    assert output.background_std.max() < 10
    assert output.background_std.min() > 1


    # test --knowntraces
    args = Namespace(data = 'acme_cems/lib/test/data/test/500.pickle',
                     masses = 'cli/configs/compounds.yml', 
                     params = 'cli/configs/acme_config.yml',
                     outdir = 'acme_cems/lib/test/data/test/',
                     sue_weights = 'cli/configs/acme_sue_weights.yml',
                     dd_weights = 'cli/configs/acme_dd_weights.yml',
                     noplots = False,
                     noexcel = False,
                     debug_plots = True,
                     cores = None,
                     saveheatmapdata = False,
                     knowntraces = True,
                     reprocess_version = None,
                     space_mode = False,
                     skip_existing = False
    )
    analyse_all_data(vars(args))

    # test --space_mode
    args = Namespace(data = 'acme_cems/lib/test/data/test/500.pickle',
                     masses = 'cli/configs/compounds.yml', 
                     params = 'cli/configs/acme_config.yml',
                     outdir = 'acme_cems/lib/test/data/test/',
                     sue_weights = 'cli/configs/acme_sue_weights.yml',
                     dd_weights = 'cli/configs/acme_dd_weights.yml',
                     noplots = False,
                     noexcel = False,
                     debug_plots = True,
                     cores = None,
                     saveheatmapdata = False,
                     knowntraces = False,
                     reprocess_version = None,
                     space_mode = True,
                     skip_existing = False
    )
    analyse_all_data(vars(args))
