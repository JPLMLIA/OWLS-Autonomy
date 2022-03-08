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


