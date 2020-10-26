import os
import pytest
import glob
import json
import shutil
import tempfile

import numpy as np

from skimage.io    import imread
from scipy.stats   import describe as desc
from numpy.testing import assert_array_equal
from collections   import defaultdict

from helm_dhm.tracker.tracker import *

TRACK_INIT_PARAMS = [
    ('helm_orig',
     {'helm_orig': {'max_assignment_sigma': 2.5, 'min_track_obs': 5, 'max_position_uncertainty': 50},
      'nearest_neighbor': {'search_range': 40, 'memory': 5, 'filter_stubs_threshold': 20},
      'dbscan': {'sigma': 3.0, 'threshold': 0.05, 'time_scale': 1.0, 'dbscan_eps': 8, 'min_samples': 9, 'cut_off': 17}},
     ['Max_Assignment_Dist', 'Min_Track_Obs', 'Max_Position_Uncertainty', 'Velocity_Prior', 'Velocity_Variance_Prior',
      'Current_Tracks', 'Average_Velocity', 'Acceleration_Prior', 'Acceleration_Variance_Prior']),
    ('nearest_neighbor',
     {'helm_orig': {'max_assignment_sigma': 2.5, 'min_track_obs': 5, 'max_position_uncertainty': 50},
      'nearest_neighbor': {'search_range': 40, 'memory': 5, 'filter_stubs_threshold': 20},
      'dbscan': {'sigma': 3.0, 'threshold': 0.05, 'time_scale': 1.0, 'dbscan_eps': 8, 'min_samples': 9, 'cut_off': 17}},
     ['Search_Range', 'Memory', 'Filter_Stubs_Threshold', 'Current_Tracks', 'Average_Velocity']),
    ('dbscan',
     {'helm_orig': {'max_assignment_sigma': 2.5, 'min_track_obs': 5, 'max_position_uncertainty': 50},
      'nearest_neighbor': {'search_range': 40, 'memory': 5, 'filter_stubs_threshold': 20},
      'dbscan': {'sigma': 3.0, 'threshold': 0.05, 'time_scale': 1.0, 'dbscan_eps': 8, 'min_samples': 9, 'cut_off': 17}},
     ['Sigma', 'Threshold', 'Time_Scale', 'DBSCAN_Eps', 'Min_Samples', 'Cut_Off', 'EPS']
     )
]


def test_helm_orig(difference_image):
    config = {'threshold': 100, 'epsilon_px': 3.0, 'min_weight': 512, 'min_px': 5, 'noise_px': 0.5, 'max_uncert_px': 50.0}
    particles = get_particles(difference_image, config)
    assert len(particles) == 156
    assert pytest.approx(particles[0]['Particle_Position'][0], rel=1e-3) == 79.5714285714
    assert pytest.approx(particles[0]['Particle_Position'][1], rel=1e-3) == 1648.28571428


def test_percentile_transformation():
    out = percentile_transformation(np.array([[0, 3], [1, 3]]))

    expected = np.array([[0, 212],
                         [85, 212]])

    assert np.allclose(out, expected)

    out = percentile_transformation(percentile_transformation(np.array([[0, 3], [1, 3]])))

    expected = np.array([[0, 212],
                         [85, 212]])

    assert np.allclose(out, expected)

    with pytest.raises(ValueError) as excinfo:
        out = percentile_transformation(np.array([[-1, 3], [1, 3]]))
    assert str(excinfo.value) == "Invalid entries for conversion to uint8"

    with pytest.raises(ValueError) as excinfo:
        out = percentile_transformation(np.array([[1, 256], [1, 3]]))
    assert str(excinfo.value) == "Invalid entries for conversion to uint8"

    with pytest.raises(ValueError) as excinfo:
        out = percentile_transformation(np.array([[], []]))
    assert str(excinfo.value) == "Image is empty."

@pytest.mark.skip
def test_get_diff():
    config = {'lag': 300, 'absthresh': 6, 'pcthresh': 99.5}
    img = np.array([[5, 0, 15], [0, 5, 0], [15, 0, 5]])
    rng_diff, aux = get_diff(img, {}, config, True)

    assert np.all(rng_diff == 0) & np.all(aux['minmax']['last_min'] == 0) & np.all(
        aux['minmax']['last_max'] == 0) == True

    assert np.all(aux['minmax']['last_range'] == 0) == True

    assert np.all(aux['minmax']['rolling_min'] == img) & np.all(aux['minmax']['rolling_max'] == img) == True

    test_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'test', 'data',
                             '2015.07.02_11-03_sub')

    my_files = sorted(glob.glob(os.path.join(test_data, 'Holograms', '*.tif')))

    rng_diff, aux = get_diff(imread(my_files[0]), {}, config, True)

    for my_file in my_files[1:3]:
        rng_diff, aux = get_diff(imread(my_file), aux, config, False)

    nobs, minmax, mean, variance, skewness, kurtosis = desc(rng_diff, None)

    assert np.max(rng_diff) == max(minmax)
    assert np.min(rng_diff) == min(minmax)
    assert pytest.approx(mean, rel=1e-3) == 0.05670595169067383
    assert pytest.approx(variance, rel=1e-3) == 0.946002933750372


@pytest.mark.skip
def test_run(sample_dataset, test_config):

    name, directories = sample_dataset

    tracker = Tracker()
    tracker.from_configuration(test_config)
    out = tracker.run(dataset_name=name, holograms=directories['Holograms'])
    out = [os.path.basename(x) for x in out]

    expected = ['000000000.track', '000000001.track', '000000002.track', '000000003.track',
                '000000004.track', '000000005.track', '000000006.track', '000000007.track',
                '000000008.track', '000000009.track', '000000010.track', '000000011.track',
                '000000012.track', '000000013.track', '000000014.track', '000000015.track',
                '000000016.track', '000000017.track']

    assert len(out) == len(expected) and sorted(out) == sorted(expected)


PROJECT_EXPECTED = [
    (1.0, (np.array([3.0, 2.0]), np.array([[2.0, -1.5], [-1.5, 3.0]]))),
    (-1.0, ValueError('Timestep must be positive'))
]

def compare_dicts(dict1, dict2):
    # Checks if both dictionaries are the same
    try:
        assert (dict1.keys() == dict2.keys())
        for key in dict1.keys():
            assert (np.allclose(dict1[key], dict2[key]))
        return True
    except:
        return False

@pytest.mark.parametrize('timestep, expected', PROJECT_EXPECTED)
def test_project(project_params, timestep, expected):
    position_current, position_variance_current, velocity_current, velocity_variance_current = project_params
    if isinstance(expected, ValueError):
        with pytest.raises(ValueError):
            project(position_current,
                                   position_variance_current,
                                   velocity_current,
                                   velocity_variance_current,
                                   timestep)
    else:
        position_projected, position_variance_projected = project(position_current,
                                                                                 position_variance_current,
                                                                                 velocity_current,
                                                                                 velocity_variance_current,
                                                                                 timestep)
        assert_array_equal(position_projected, expected[0])
        assert_array_equal(position_variance_projected, expected[1])


AGGREGATE_PARAMS = [
    (np.array([[2, 3], [-2, -3]]), np.array([[[1, -.5], [-.5, 1]], [[1, .5], [.5, 0]]]), [np.array([0, 0]),
                                                                                          np.array([[1., 0.],
                                                                                                    [0., 0.5]])]),
    (np.array([[2, 3], [-2, -3]]), np.array([[1, -.5], [-.5, 1]]), [np.array([0., 0.]),
                                                                    np.array([[0.25, 0.25]])])
]


@pytest.mark.parametrize('position_array, position_variance_array, expected', AGGREGATE_PARAMS)
def test_aggregate(position_array, position_variance_array, expected):
    aggregated_position, aggregated_variance = aggregate(position_array, position_variance_array)
    assert_array_equal(aggregated_position, expected[0])
    assert_array_equal(aggregated_variance, expected[1])


def test_sigma_to_mahalanobis():
    out = sigma_to_mahalanobis(0.5)
    assert out == 0.9655291620673471
    with pytest.raises(ValueError) as excinfo:
        out = sigma_to_mahalanobis(-0.5)
    assert str(excinfo.value) == 'Sigma cannot be negative.'


def test_particle_track(particle_track_input, particle_track_output):
    time_0, position, position_variance, velocity, velocity_variance, acceleration, acceleration_variance = particle_track_input
    _particle_track = particle_track(time_0, position, position_variance, velocity,
                                                    velocity_variance, acceleration, acceleration_variance)

    assert _particle_track.keys() == particle_track_output.keys()

    for i, val in enumerate(particle_track_output['Particles_Position']):
        assert np.allclose(_particle_track['Particles_Position'][i], particle_track_output['Particles_Position'][i])

    for i, val in enumerate(particle_track_output['Particles_Variance']):
        assert np.allclose(_particle_track['Particles_Variance'][i], particle_track_output['Particles_Variance'][i])

def test_update():
    track = {'Times': [17], 'Particles_Position': [np.array([271.5263157894737, 777.9473684210526])],
             'Particles_Variance': [np.array([[2.318713450292397, -1.3040935672514622],
                                              [-1.3040935672514622, 4.219298245614036]])],
             'Particles_Estimated_Position': [np.array([271.5263157894737, 777.9473684210526])],
             'Particles_Estimated_Position_Variance': [np.array([[2.318713450292397, -1.3040935672514622],
                                                                 [-1.3040935672514622, 4.219298245614036]])],
             'Particles_Estimated_Velocity': [np.array([7.643962848297235, -8.170278637770934])],
             'Particles_Estimated_Velocity_Variance': [np.array([[4.804007567939457, -2.6643876848985215],
                                                                 [-2.6643876848985215, 7.454592363261096]])],
             'Particles_Estimated_Acceleration': np.array([0.0, 0.0]),
             'Particles_Estimated_Acceleration_Variance': np.array([[0.5, 0.0], [0.0, 0.5]]),
             'Average_Velocity': [np.array([0.1291750759206047, -0.7791218811635623])],
             'Track_ID': 5}

    particle_track = update(18, None, None, track)

    expected = {'Times': [17, 18],
                'Particles_Position': [np.array([271.52631579, 777.94736842]), None],
                'Particles_Variance': [np.array([[2.31871345, -1.30409357],
                                                 [-1.30409357, 4.21929825]]), None],
                'Particles_Estimated_Position': [np.array([271.52631579, 777.94736842]),
                                                 np.array([279.17027864, 769.77708978])],
                'Particles_Estimated_Position_Variance': [np.array([[2.31871345, -1.30409357],
                                                                    [-1.30409357, 4.21929825]]),
                                                          np.array([[7.12272102, -3.96848125],
                                                                    [-3.96848125, 11.67389061]])],
                'Particles_Estimated_Velocity': [np.array([7.64396285, -8.17027864]),
                                                 np.array([7.64396285, -8.17027864])],
                'Particles_Estimated_Velocity_Variance': [np.array([[4.80400757, -2.66438768],
                                                                    [-2.66438768, 7.45459236]]),
                                                          np.array([[5.30400757, -2.66438768],
                                                                    [-2.66438768, 7.95459236]])],
                'Particles_Estimated_Acceleration': np.array([0., 0.]),
                'Particles_Estimated_Acceleration_Variance': np.array([[0.5, 0.],
                                                                       [0., 0.5]]),
                'Average_Velocity': [np.array([0.12917508, -0.77912188])], 'Track_ID': 5}

    for i, val in enumerate(expected['Particles_Position']):
        if val is not None:
            assert np.allclose(particle_track['Particles_Position'][i], expected['Particles_Position'][i])
        else:
            assert particle_track['Particles_Position'][i] is expected['Particles_Position'][i] is None

    for i, val in enumerate(expected['Particles_Variance']):
        if val is not None:
            assert np.allclose(particle_track['Particles_Variance'][i], expected['Particles_Variance'][i])
        else:
            assert particle_track['Particles_Position'][i] is expected['Particles_Position'][i] is None

    for i, val in enumerate(expected['Particles_Estimated_Position_Variance']):
        if val is not None:
            assert np.allclose(particle_track['Particles_Estimated_Position_Variance'][i],
                               expected['Particles_Estimated_Position_Variance'][i])
        else:
            assert particle_track['Particles_Position'][i] is expected['Particles_Position'][i] is None

    position_new = np.array([277.57142857142856, 773.2142857142858])
    position_variance_new = np.array([[1.0238095238095237, -0.04761904761904762],
                                      [-0.04761904761904762, 1.3095238095238093]])

    particle_track = update(19, position_new, position_variance_new, particle_track)

    expected = {'Times': [17, 18, 19],
                'Particles_Position': [np.array([271.52631579, 777.94736842]),
                                       None,
                                       np.array([277.57142857, 773.21428571])],
                'Particles_Variance': [np.array([[2.31871345, -1.30409357],
                                                 [-1.30409357, 4.21929825]]),
                                       None,
                                       np.array([[1.02380952, -0.04761905],
                                                 [-0.04761905, 1.30952381]])],
                'Particles_Estimated_Position': [np.array([271.52631579, 777.94736842]),
                                                 np.array([279.17027864, 769.77708978]),
                                                 np.array([277.57142857, 773.21428571])],
                'Particles_Estimated_Position_Variance': [np.array([[2.31871345, -1.30409357],
                                                                    [-1.30409357, 4.21929825]]),
                                                          np.array([[7.12272102, -3.96848125],
                                                                    [-3.96848125, 11.67389061]]),
                                                          np.array([[1.02380952, -0.04761905],
                                                                    [-0.04761905, 1.30952381]])],
                'Particles_Estimated_Velocity': [np.array([7.64396285, -8.17027864]),
                                                 np.array([7.64396285, -8.17027864]),
                                                 np.array([-1.59885007, 3.43719593])],
                'Particles_Estimated_Velocity_Variance': [np.array([[4.80400757, -2.66438768],
                                                                    [-2.66438768, 7.45459236]]),
                                                          np.array([[5.30400757, -2.66438768],
                                                                    [-2.66438768, 7.95459236]]),
                                                          np.array([[8.14653054, -4.0161003],
                                                                    [-4.0161003, 12.98341442]])],
                'Particles_Estimated_Acceleration': np.array([0., 0.]),
                'Particles_Estimated_Acceleration_Variance': np.array([[0.5, 0.],
                                                                       [0.,
                                                                        0.5]]),
                'Average_Velocity': [np.array([0.12917508, -0.77912188])], 'Track_ID': 5}

    for i, val in enumerate(expected['Particles_Position']):
        if val is not None:
            assert np.allclose(particle_track['Particles_Position'][i], expected['Particles_Position'][i])
        else:
            assert particle_track['Particles_Position'][i] is expected['Particles_Position'][i] is None

    for i, val in enumerate(expected['Particles_Variance']):
        if val is not None:
            assert np.allclose(particle_track['Particles_Variance'][i], expected['Particles_Variance'][i])
        else:
            assert particle_track['Particles_Position'][i] is expected['Particles_Position'][i] is None

    for i, val in enumerate(expected['Particles_Estimated_Position_Variance']):
        if val is not None:
            assert np.allclose(particle_track['Particles_Estimated_Position_Variance'][i],
                               expected['Particles_Estimated_Position_Variance'][i])
        else:
            assert particle_track['Particles_Position'][i] is expected['Particles_Position'][i] is None


@pytest.mark.skip
def test_tracker_save(particle_track_for_save):
    output_dir = tempfile.mkdtemp()
    tracker_save(os.path.join(output_dir, "sample.track"), particle_track_for_save)
    with open(os.path.join(output_dir, "sample.track"), 'r') as f:
        data = json.load(f)

    for i, val in enumerate(particle_track_for_save['Particles_Position']):
        if val is not None:
            assert np.allclose(data['Particles_Position'][i], particle_track_for_save['Particles_Position'][i])
        else:
            assert data['Particles_Position'][i] is particle_track_for_save['Particles_Position'][i] is None

    for i, val in enumerate(particle_track_for_save['Particles_Variance']):
        if val is not None:
            assert np.allclose(data['Particles_Variance'][i], particle_track_for_save['Particles_Variance'][i])
        else:
            assert data['Particles_Position'][i] is particle_track_for_save['Particles_Position'][i] is None

    for i, val in enumerate(particle_track_for_save['Particles_Estimated_Position_Variance']):
        if val is not None:
            assert np.allclose(particle_track_for_save['Particles_Estimated_Position_Variance'][i],
                               data['Particles_Estimated_Position_Variance'][i])
        else:
            assert data['Particles_Position'][i] is particle_track_for_save['Particles_Position'][i] is None

    shutil.rmtree(output_dir)

def test_uncertainty_major_radius():
    out = uncertainty_major_radius(np.array([[.2, .3], [.1, .2]]), 1)
    assert out == pytest.approx(1.85, rel=1e-3)
    with pytest.raises(ValueError) as excinfo:
        out = uncertainty_major_radius(np.array([[.2, .3, .3], [.1, .2, .2]]))
    assert str(excinfo.value) == 'Covariance must be square matrix'


def test_get_particle_tracks():
    assignments = defaultdict(list)
    particle = {'Particle_Position': np.array([2, 3]), 'Particle_Variance': np.array([[1, 0.1], [0.1, 1]])}
    assignments[np.argmin(np.array([20, 15, 78]))].append(particle)
    particle = {'Particle_Position': np.array([-1, -3]), 'Particle_Variance': np.array([[2, 0.1], [0.1, 3]])}
    assignments[np.argmin(np.array([10, 5, 78]))].append(particle)
    particle = {'Particle_Position': np.array([0, 0]), 'Particle_Variance': np.array([[2, 0.0], [0.0, 5]])}
    assignments[np.argmin(np.array([1, 5, 0.2]))].append(particle)

    assert np.allclose(np.array([assignments[1][0]['Particle_Position']]), np.asarray([[2, 3]]))
    assert np.allclose(np.array([assignments[1][0]['Particle_Variance']]), np.asarray(np.asarray([[1.0, 0.1],
                                                                                                  [0.1, 1.0]])))


@pytest.mark.skip
def test_get_xyt(postprocessors_sample_track):
    x, y, t = get_xyt(postprocessors_sample_track)

    assert np.allclose(np.array(x), np.array([271.52631579, 279.17027864, 277.57142857]))
    assert np.allclose(np.array(y), np.array([777.94736842, 769.77708978, 773.21428571]))
    assert np.allclose(np.array(t), np.array([17, 18, 19]))


@pytest.mark.skip
def test_update_positions(postprocessors_sample_track):

    # Check if input data matches expected
    x, y, t = get_xyt(postprocessors_sample_track)

    assert np.allclose(np.array(x), np.array([271.52631579, 279.17027864, 277.57142857]))
    assert np.allclose(np.array(y), np.array([777.94736842, 769.77708978, 773.21428571]))
    assert np.allclose(np.array(t), np.array([17, 18, 19]))

    newx = [2, 3, 4]
    newy = [4, 5, 6]

    track = update_positions(postprocessors_sample_track, newx, newy)
    x, y, t = get_xyt(track)

    assert np.allclose(np.array(x), np.array([2, 3, 4]))
    assert np.allclose(np.array(y), np.array([4, 5, 6]))

    with pytest.raises(ValueError) as excinfo:
        newxfail = [1, 2, 3, 4]
        newyfail = [1, 2, 3, 4]
        track = update_positions(track, newxfail, newyfail)
        x, y, t = get_xyt(track)
    assert str(excinfo.value) == 'x and y arrays must be equal in length along interpolation axis.'

    with pytest.raises(ValueError) as excinfo:
        newxfail = [1, 2]
        newyfail = [1, 2]
        track = update_positions(track, newxfail, newyfail)
        x, y, t = get_xyt(track)
    assert str(excinfo.value) == 'x and y arrays must be equal in length along interpolation axis.'


@pytest.mark.skip
def test_smooth_gaussian(postprocessors_sample_track):
    x, y, t = get_xyt(postprocessors_sample_track)

    assert np.allclose(np.array(x), np.array([271.52631579, 279.17027864, 277.57142857]))
    assert np.allclose(np.array(y), np.array([777.94736842, 769.77708978, 773.21428571]))
    assert np.allclose(np.array(t), np.array([17, 18, 19]))

    track = smooth_gaussian(postprocessors_sample_track, 2.0)
    x, y, t = get_xyt(track)

    assert np.allclose(np.array(x), np.array([275.75189091, 276.08977132, 276.42636077]))
    assert np.allclose(np.array(y), np.array([773.9105597,  773.64570756, 773.38247665]))
    assert np.allclose(np.array(t), np.array([17, 18, 19]))


@pytest.mark.skip
def test_process_invalid_cases(postprocessors_sample_track):
    postproc = 'smoothing_median'
    config = {'smoothing': {'window_size': 5.0}}
    with pytest.raises(TypeError) as excinfo:
        postprocess(postprocessors_sample_track, postproc, config)
    assert(str(excinfo.value) == "\'numpy.float64\' object cannot be interpreted as an integer")

    config = {'smoothing': {'window_size': 2}}
    with pytest.raises(ValueError) as excinfo:
        postprocess(postprocessors_sample_track, postproc, config)
    assert(str(excinfo.value) == "Each element of kernel_size should be odd.")

    postproc = 'smooth'
    config = {'smoothing': {'window_size': 3}}
    with pytest.raises(ValueError) as excinfo:
        postprocess(postprocessors_sample_track, postproc, config)
    assert(str(excinfo.value) == "Postprocessing smoothing filter specified does not exist. "
                                 "Filters to chose from: median, average, gaussian")
