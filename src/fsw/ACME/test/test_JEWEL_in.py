# unit testing of JEWEL_in via pytest
#
# Steffen Mauceri
# Oct 2020

import pytest

from acme_cems.lib.JEWEL_in import *

# test for functions
def test_calc_SUE():
    peak_properties = pd.DataFrame(data={'mass_idx': [10,20,30], 'time_idx': [4,5,6], 'zscore': [10,20,30]})
    sue_weights_path = 'cli/configs/acme_sue_weights.yml'
    compounds = {10: 'test1', 20.5: 'test2', 104: 'test104'}
    mass_axis = np.arange(0,1000)
    masses_dist_max = 1
    savepath = 'acme_cems/lib/test/data/test/500/500_SUE.csv'
    label = 'test'

    calc_SUE(label, peak_properties, sue_weights_path, compounds, mass_axis, masses_dist_max, savepath)

    output_base = pd.read_csv(savepath)
    # test that SUE gets bigger and smaller as expected
    peak_properties = pd.DataFrame(data={'mass_idx': [10,20,104], 'time_idx': [4,5,6], 'zscore': [10,20,30]})
    calc_SUE(label, peak_properties, sue_weights_path, compounds, mass_axis, masses_dist_max, savepath)
    output_higher = pd.read_csv(savepath)

    assert output_base['SUE'].to_numpy() < output_higher['SUE'].to_numpy()

    peak_properties = pd.DataFrame(data={'mass_idx': [10, 20, 30, 30], 'time_idx': [4, 5, 6,7], 'zscore': [10, 20, 30, 20]})
    calc_SUE(label, peak_properties, sue_weights_path, compounds, mass_axis, masses_dist_max, savepath)
    output_higher = pd.read_csv(savepath)

    assert output_base['SUE'].to_numpy() < output_higher['SUE'].to_numpy()

    peak_properties = pd.DataFrame(data={'mass_idx': [10, 20, 30], 'time_idx': [4, 5, 6], 'zscore': [10, 20, 20]})
    calc_SUE(label, peak_properties, sue_weights_path, compounds, mass_axis, masses_dist_max, savepath)
    output_lower = pd.read_csv(savepath)

    assert output_base['SUE'].to_numpy() > output_lower['SUE'].to_numpy()
    #test saturation
    peak_properties = pd.DataFrame(data={'mass_idx': [10, 20, 30], 'time_idx': [4, 5, 6], 'zscore': [100, 100, 100]})
    calc_SUE(label, peak_properties, sue_weights_path, compounds, mass_axis, masses_dist_max, savepath)
    output_sat1 = pd.read_csv(savepath)

    assert output_base['SUE'].to_numpy() > output_lower['SUE'].to_numpy()

    peak_properties = pd.DataFrame(data={'mass_idx': [10, 20, 30], 'time_idx': [4, 5, 6], 'zscore': [1000, 200, 200]})
    calc_SUE(label, peak_properties, sue_weights_path, compounds, mass_axis, masses_dist_max, savepath)
    output_sat2 = pd.read_csv(savepath)

    assert output_sat1['SUE'].to_numpy() == output_sat2['SUE'].to_numpy()

    # test no peaks found
    peak_properties = pd.DataFrame(data={'mass_idx': [], 'time_idx': [], 'zscore': []})
    calc_SUE(label, peak_properties, sue_weights_path, compounds, mass_axis, masses_dist_max, savepath)

def test_diversity_descriptor():
    a = [10, 20, 30]
    b = [1000,2000,2000]
    peak_properties = pd.DataFrame(data={
        'height': a,
         'zscore': a,
         'volume': b,
         'volume_top': b,
         'volume_zscore': a,
         'peak_base_width': a,
         'mass_idx': a,
         'time_idx': a,
         'background_abs': a,
         'background_std': a,
         'background_ratio': a,
         'background_diff': a})

    dd_weights_path = 'cli/configs/acme_dd_weights.yml'
    compounds = {10: 'test1', 20.5: 'test2', 104: 'test104'}
    mass_axis = np.arange(0, 1000)
    masses_dist_max = 1
    savepath = 'acme_cems/lib/test/data/test/500/500_DD.csv'
    label = 'test'

    diversity_descriptor(label, peak_properties, dd_weights_path, compounds, mass_axis, masses_dist_max, savepath)
    output_base = pd.read_csv(savepath)

    peak_properties['height'] = [10, 20, 100]
    diversity_descriptor(label, peak_properties, dd_weights_path, compounds, mass_axis, masses_dist_max, savepath)
    output_1 = pd.read_csv(savepath)

    assert output_base['a_height'].to_numpy() < output_1['a_height'].to_numpy()
    assert output_base['s_height'].to_numpy() < output_1['s_height'].to_numpy()
    assert output_base['a_volume'].to_numpy() == output_1['a_volume'].to_numpy()

    # test saturation
    peak_properties['height'] = [10000, 20000, 1000]
    diversity_descriptor(label, peak_properties, dd_weights_path, compounds, mass_axis, masses_dist_max, savepath)
    output_sat_1 = pd.read_csv(savepath)

    peak_properties['height'] = [100000, 200000, 100000]
    diversity_descriptor(label, peak_properties, dd_weights_path, compounds, mass_axis, masses_dist_max, savepath)
    output_sat_2 = pd.read_csv(savepath)

    assert output_sat_1['a_height'].to_numpy() == output_sat_2['a_height'].to_numpy()

    # test no peaks found
    a = []
    b = []
    peak_properties = pd.DataFrame(data={
        'height': a,
        'zscore': a,
        'volume': b,
        'volume_top': b,
        'volume_zscore': a,
        'peak_base_width': a,
        'mass_idx': a,
        'time_idx': a,
        'background_abs': a,
        'background_std': a,
        'background_ratio': a,
        'background_diff': a})
    diversity_descriptor(label, peak_properties, dd_weights_path, compounds, mass_axis, masses_dist_max, savepath)
