import os
import glob
import tempfile

import pytest
from helm_dhm.validate import products
import pylab as P
import numpy as np
from numpy.testing import assert_array_equal
from scipy.ndimage import gaussian_filter


dummy_image = np.ones((4, 4))
dummy_image[2:, 1:3] = 0

def test_detect_defocus():
    test_image = 6
    path = os.path.join(os.path.dirname(__file__), 'data')
    files = list(glob.glob(os.path.join(path, "*.tif")))
    newsize = 2048


    images = np.zeros((newsize, newsize, len(files)), dtype=float)
    for i, _file in enumerate(files):
        temp = P.imread(_file)
        if i == test_image:
            temp = gaussian_filter(temp, 3)
        images[:,:,i] = temp

    result = products.detect_defocus(path=os.path.join(tempfile.mkdtemp(), 'temp'), images=images, threshold=2)
    assert (result == [test_image])


    images = np.zeros((newsize, newsize, len(files)), dtype=float)
    for i, _file in enumerate(files):
        temp = P.imread(_file)
        images[:, :, i] = temp

    result = products.detect_defocus(path=os.path.join(tempfile.mkdtemp(), 'temp'), images=images, threshold=2)
    assert (result == [])

class TestDensityEstimation:
    """Tests to see if density/crowdedness estimation are functioning"""

    def test_stdev_calcs(self):
        std_vals, std_viz_image = products.blockwise_img_stdev(dummy_image, 2)
        assert_array_equal(std_vals, np.array([[0, 0], [0.5, 0.5]]))
        assert_array_equal(std_viz_image, np.array([[0, 0, 0, 0],
                                                    [0, 0, 0, 0],
                                                    [0.5, 0.5, 0.5, 0.5],
                                                    [0.5, 0.5, 0.5, 0.5]]))

        std_vals, std_viz_image = products.blockwise_img_stdev(dummy_image, 4)
        assert_array_equal(std_vals, np.array([[0.4330127018922193]]))
        assert_array_equal(std_viz_image,
                           np.ones_like(dummy_image) * 0.4330127018922193)

