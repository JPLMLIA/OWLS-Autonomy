import os
import os.path as op
import glob
import tempfile

import pytest
import pylab as P
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.ndimage import gaussian_filter

from utils.file_manipulation import tiff_read
from helm_dhm.validate import products


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


class TestFourierTransform:
    """Tests that the 2D Fourier transform functionality"""

    def test_image_power(self):
        # Load first image, compute FFT and log power. Don't scale from [0, 1]

        image = tiff_read(op.join(os.path.dirname(os.path.realpath(__file__)),
                                                  'data', '00001_holo.tif'))
        img_power = products.fourier_transform_image(image, scale=False)

        # Load ground truth power calculation
        fpath_gt_power = op.join(op.dirname(op.realpath(__file__)),
                                            'data', 'test', '00001_holo_log_power.npy')
        with open(fpath_gt_power, 'rb') as np_file:
            gt_power = np.load(np_file)

        assert_array_almost_equal(img_power, gt_power)
