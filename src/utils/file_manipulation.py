import os
import os.path as op
import logging

import PIL
import cv2
import numpy as np
from skimage.io        import imread
from skimage.transform import resize

def read_image(tiff_path, raw_dims, resize_dims=None, flatten=False):
    """ Read a tiff or raw image with error handling and optional resizing.

    Parameters
    ----------
    tiff_path: str
        Path to tiff image
    raw_dims: tuple
        Specify expected dimensions of read file.
    resize_dims: tuple
        Optional. Specify to force resize after image read.
    flatten: bool
        Optional. Flatten multichannel to single by averaging the last channel.

    Returns
    -------
    Array of image if successfully read.
    None if image is corrupt or does not exist.
    """

    if not os.path.exists(tiff_path):
        logging.error("File does not exist.")
        return None

    _, file_extension = op.splitext(tiff_path)

    if file_extension in [".tif", ".tiff"]:
        try:
            image = imread(tiff_path)
        except:
            logging.error(f"Corrupt image: {tiff_path}")
            return None

        if image.size == 0:
            logging.error(f"Corrupt image: {tiff_path}")
            return None

    elif file_extension == ".raw":
        try:
            with open(tiff_path, mode='rb') as f:
                image = np.fromfile(f, dtype=np.uint8,count=raw_dims[0]*raw_dims[1]).reshape(raw_dims[0],raw_dims[1])
        except:
            logging.error(f"Corrupt image: {tiff_path}")
            return None

        # if 3-channel raw is expected
        if len(raw_dims) == 3 and raw_dims[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BayerBG2RGB)

    if flatten and len(image.shape) == 3:
        # We use averaging instead of rgb2gray because it uses
        # CRT luminance, which is a weighted mean:
        # https://scikit-image.org/docs/dev/api/skimage.color.html#rgb2gray
        image = np.squeeze(np.round(np.mean(image, axis=-1)).astype(np.uint8))

    if resize_dims:
        image = resize(image, resize_dims, anti_aliasing=True)

    return image

def write_image(image, save_path):
    """ Write an image array to a path as a tiff.

    Parameters
    ----------
    image: array
        Image to be saved to save_path
    save_path: str
        Filepath for the image to be saved to
    """

    save_folder = op.dirname(save_path)
    if not os.path.exists(save_folder):
        logging.warning(f"{save_folder} folder does not exist, creating.")
        os.makedirs(save_folder)

    pil_img = PIL.Image.fromarray(image)
    pil_img.save(save_path, compression='tiff_lzw')
