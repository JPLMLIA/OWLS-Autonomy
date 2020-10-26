import os
import glob
import json
import pytest
from helm_dhm.tracker import *

import numpy as np

@pytest.fixture
def postprocessors_sample_track():
    test_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sample.track')
    with open(test_data) as f:
        track = json.load(f)
    return track

@pytest.fixture
def sample_dataset():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', '2015.07.02_11-03_sub')
    name = os.path.basename(path)
    holograms = {'Holograms': glob.glob(os.path.join(path, 'Holograms', '*.tif'))}

    return name, holograms

@pytest.fixture
def test_config():
    return os.path.join(os.path.dirname(__file__), 'data', 'test_config.yml')

@pytest.fixture
def difference_image():
    return np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'img_diff_00030.npy'))


@pytest.fixture
def project_params():
    return np.array([2, 3]), np.array([[1, -.5], [-.5, 2]]), np.array([1, -1]), np.array([[1, -1], [-1, 1]])


@pytest.fixture
def particle_track_input():
    # time_0, position, position_variance, velocity, velocity_variance, acceleration, acceleration_variance
    return 1, np.array([-2, 3]), np.array([[1, -.5], [-.5, 2]]), np.array([1, 1]), np.array([[1, .5], [.5, 2]]), \
           np.array([.1, .2]), np.array([[1, .1], [.1, -1]])


@pytest.fixture
def particle_track_output():
    return {'Times': [1], 'Particles_Position': [np.array([-2, 3])], 'Particles_Variance': [np.array([[1., -0.5],
                                                                                                      [-0.5, 2.]])],
            'Particles_Estimated_Position': [np.array([-2, 3])],
            'Particles_Estimated_Position_Variance': [np.array([[1., -0.5],
                                                                [-0.5, 2.]])],
            'Particles_Estimated_Velocity': [np.array([1, 1])],
            'Particles_Estimated_Velocity_Variance': [np.array([[1., 0.5],
                                                                [0.5, 2.]])],
            'Particles_Estimated_Acceleration': np.array([0.1, 0.2]),
            'Particles_Estimated_Acceleration_Variance': np.array([[1., 0.1],
                                                                   [0.1, -1.]]), 'Average_Velocity': [None],
            'Track_ID': None}


@pytest.fixture
def particle_track_for_save():
    particle_track = {'Times': [17, 18, 19],
                      'Particles_Position': [np.array([271.52631579, 777.94736842]), None,
                                             np.array([277.57142857, 773.21428571])],
                      'Particles_Variance': [np.array([[2.31871345, -1.30409357], [-1.30409357, 4.21929825]]),
                                             None, np.array([[1.02380952, -0.04761905], [-0.04761905, 1.30952381]])],
                      'Particles_Estimated_Position': [np.array([271.52631579, 777.94736842]),
                                                       np.array([279.17027864, 769.77708978]),
                                                       np.array([277.57142857, 773.21428571])],
                      'Particles_Estimated_Position_Variance': [
                          np.array([[2.31871345, -1.30409357], [-1.30409357, 4.21929825]]),
                          np.array([[7.12272102, -3.96848125], [-3.96848125, 11.67389061]]),
                          np.array([[1.02380952, -0.04761905], [-0.04761905, 1.30952381]])],
                      'Particles_Estimated_Velocity': [np.array([7.64396285, -8.17027864]),
                                                       np.array([7.64396285, -8.17027864]),
                                                       np.array([-1.59885007, 3.43719593])],
                      'Particles_Estimated_Velocity_Variance': [
                          np.array([[4.80400757, -2.66438768], [-2.66438768, 7.45459236]]),
                          np.array([[5.30400757, -2.66438768], [-2.66438768, 7.95459236]]),
                          np.array([[8.14653054, -4.0161003], [-4.0161003, 12.98341442]])],
                      'Particles_Estimated_Acceleration': np.array([0., 0.]),
                      'Particles_Estimated_Acceleration_Variance': np.array([[0.5, 0.], [0., 0.5]]),
                      'Average_Velocity': [np.array([0.12917508, -0.77912188])],
                      'Track_ID': 5}
    return particle_track


@pytest.fixture
def nearest_neighbor_track():
    return {'Times': [17, 18],
            'Particles_Position': [np.array([271.52631579, 777.94736842]), None],
            'Particles_Variance': [np.array([[2.31871345, -1.30409357], [-1.30409357, 4.21929825]]), None],
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
            'Average_Velocity': [np.array([0.12917508, -0.77912188])],
            'Current_Memory': 5,
            'Track_ID': 5
            }


@pytest.fixture
def nearest_neighbor_three_times():
    return {'Times': [17, 18, 19],
            'Particles_Position':
                [np.array([271.52631579, 777.94736842]), None, np.array([277.57142857, 773.21428571])],
            'Particles_Variance':
                [np.array([[2.31871345, -1.30409357], [-1.30409357, 4.21929825]]),
                 None,
                 np.array([[1.02380952, -0.04761905], [-0.04761905, 1.30952381]])],
            'Particles_Estimated_Position':
                [np.array([271.52631579, 777.94736842]),
                 np.array([279.17027864, 769.77708978]),
                 np.array([277.57142857, 773.21428571])],
            'Particles_Estimated_Position_Variance':
                [np.array([[2.31871345, -1.30409357],
                           [-1.30409357, 4.21929825]]), np.array([[7.12272102, -3.96848125],
                                                                  [-3.96848125, 11.67389061]]),
                 np.array([[1.02380952, -0.04761905],
                           [-0.04761905, 1.30952381]])],
            'Particles_Estimated_Velocity':
                [np.array([7.64396285, -8.17027864]),
                 np.array([7.64396285, -8.17027864]),
                 np.array([-1.59885007, 3.43719593])],
            'Particles_Estimated_Velocity_Variance':
                [np.array([[4.80400757, -2.66438768], [-2.66438768, 7.45459236]]),
                 np.array([[5.30400757, -2.66438768], [-2.66438768, 7.95459236]]),
                 np.array([[8.14653054, -4.0161003], [-4.0161003, 12.98341442]])],
            'Particles_Estimated_Acceleration': np.array([0., 0.]),
            'Particles_Estimated_Acceleration_Variance': np.array([[0.5, 0.], [0., 0.5]]),
            'Average_Velocity': [np.array([0.12917508, -0.77912188])],
            'Current_Memory': 5,
            'Track_ID': 5}


@pytest.fixture
def random_dbscan_movie():
    np.random.seed(0)
    return np.random.rand(20, 1024, 1024)


@pytest.fixture()
def dbscan_tracker_expected_tracks_list():
    return {'Times': [6, 7, 8, 9, 10, 11],
            'Cloud': {0: np.array(
                [[380, 247], [380, 248], [380, 249], [381, 247], [381, 248], [381, 249], [381, 250], [382, 247],
                 [382, 248], [382, 249], [382, 250]]),
                1: None,
                2: np.array([[379, 248], [379, 249], [380, 247], [380, 248], [380, 249], [381, 246], [381, 247],
                             [381, 248],
                             [381, 249], [382, 245], [382, 246], [382, 247], [382, 248], [382, 249], [382, 250],
                             [383, 245],
                             [383, 246], [383, 247], [383, 248], [383, 249], [383, 250], [384, 244], [384, 245],
                             [384, 246],
                             [384, 247], [384, 248], [384, 249], [384, 250], [385, 244], [385, 245], [385, 246],
                             [385, 247],
                             [385, 248], [385, 249], [385, 250], [386, 244], [386, 245], [386, 246], [386, 247],
                             [386, 248],
                             [386, 249], [386, 250], [387, 245], [387, 246], [387, 247], [387, 248], [387, 249],
                             [387, 250],
                             [388, 245], [388, 246], [388, 247], [388, 248], [388, 249], [388, 250], [389, 245],
                             [389, 246],
                             [389, 247], [389, 248], [389, 249], [390, 246], [390, 247], [390, 248]]),
                3: np.array([[377, 245], [377, 246], [377, 247], [378, 245], [378, 246], [378, 247], [378, 248],
                             [379, 244],
                             [379, 245], [379, 246], [379, 247], [379, 248], [380, 244], [380, 245], [380, 246],
                             [380, 247],
                             [380, 248], [381, 244], [381, 245], [381, 246], [381, 247], [382, 244], [382, 245],
                             [382, 246]]),
                4: None,
                5: np.array([[377, 246], [377, 247], [377, 248], [378, 245], [378, 246], [378, 247], [378, 248],
                             [378, 249],
                             [378, 250], [379, 245], [379, 246], [379, 247], [379, 248], [379, 249], [379, 250],
                             [379, 251],
                             [380, 246], [380, 247], [380, 248], [380, 249], [380, 250], [380, 251], [381, 247],
                             [381, 248],
                             [381, 249], [381, 250], [381, 251]])},
            'Particles_Position': [np.array([381.09090909, 248.36363636]), None, np.array([384.87096774, 247.32258065]),
                                   np.array([379.5, 245.875]), None, np.array([379.14814815, 248.07407407])],
            'Particles_Variance': [np.array([[0.69090909, 0.16363636], [0.16363636, 1.25454545]]), None,
                                   np.array([[8.76996298, -0.62982549], [-0.62982549, 2.97620307]]),
                                   np.array([[2.52173913, -0.58695652],
                                             [-0.58695652, 1.67934783]]), None,
                                   np.array([[1.66951567, 0.83475783], [0.83475783, 3.3019943]])],
            'Particles_Estimated_Position': [np.array([381.09090909, 248.36363636]),
                                             np.array([381.09090909, 248.36363636]),
                                             np.array([384.87096774, 247.32258065]), np.array([379.5, 245.875]),
                                             np.array([374.12903226, 244.42741935]),
                                             np.array([379.14814815, 248.07407407])],
            'Particles_Estimated_Position_Variance': [np.array([[0.69090909, 0.16363636], [0.16363636, 1.25454545]]),
                                                      np.array([[0.69090909, 0.16363636], [0.16363636, 1.25454545]]),
                                                      np.array([[8.76996298, -0.62982549],
                                                                [-0.62982549, 2.97620307]]),
                                                      np.array([[2.52173913, -0.58695652], [-0.58695652, 1.67934783]]),
                                                      np.array([[13.81344124, -1.80373853], [-1.80373853, 6.33489872]]),
                                                      np.array([[1.66951567, 0.83475783],
                                                                [0.83475783, 3.3019943]])],
            'Particles_Estimated_Velocity': [None, np.array([0., 0.]), np.array([3.78005865, -1.04105572]),
                                             np.array([-5.37096774, -1.44758065]), np.array([-5.37096774, -1.44758065]),
                                             np.array([5.01911589, 3.64665472])],
            'Particles_Estimated_Velocity_Variance': [None,
                                                      np.array([[1.38181818, 0.32727273], [0.32727273, 2.50909091]]),
                                                      np.array([[9.46087207, -0.46618913], [-0.46618913, 4.23074852]]),
                                                      np.array([[11.29170211, -1.21678201],
                                                                [-1.21678201, 4.65555089]]),
                                                      np.array([[16.33518037, -2.39069505], [-2.39069505, 8.01424655]]),
                                                      np.array([[15.48295691, -0.9689807], [-0.9689807, 9.63689302]])],
            'Particles_Estimated_Acceleration': [None, None, np.array([3.78005865, -1.04105572]),
                                                 np.array([-9.15102639, -0.40652493]), np.array([0., 0.]),
                                                 np.array([10.39008363, 5.09423536])],
            'Particles_Estimated_Acceleration_Variance': [None, None, np.array([[10.84269026, -0.1389164],
                                                                                [-0.1389164, 6.73983943]]),
                                                          np.array(
                                                              [[20.75257419, -1.68297114], [-1.68297114, 8.88629941]]),
                                                          np.array(
                                                              [[27.62688249, -3.60747707], [-3.60747707, 12.66979744]]),
                                                          np.array([[31.81813729, -3.35967575],
                                                                    [-3.35967575, 17.65113957]])],
            'Average_Velocity': [np.array([-1.60604706, 4.71995019]),
                                 np.array([-1.40571219, 3.61376749]),
                                 np.array([2.65312405, -1.80290545]),
                                 np.array([-2.81725992, 1.50508258]),
                                 np.array([0.42382931, 0.27260142]),
                                 np.array([0.05587251, -0.50865689])],
            'Track_ID': 6}

@pytest.fixture
def test_tracks():
    return list(glob.glob(os.path.join(os.path.dirname(__file__), 'data', '2019.11.12_09.26.26.655', '*.track')))

