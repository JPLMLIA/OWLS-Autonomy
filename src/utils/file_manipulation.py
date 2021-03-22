import os
import os.path as op
import logging

import PIL
import numpy as np
from skimage.io        import imread
from skimage.transform import resize

def tiff_read(tiff_path, resize_dims=None):
    """ Read a tiff image with error handling and optional resizing.

    Parameters
    ----------
    tiff_path: str
        Path to tiff image
    resize_dims: tuple
        Optional. Specify to force resize after image read.
    
    Returns
    -------
    Array of image if successfully read.
    None if image is corrupt or does not exist.
    """

    if os.path.exists(tiff_path):
        try:
            image = imread(tiff_path)
            if image.size == 0:
                logging.error(f"Corrupt image: {tiff_path}")
                return None
        except:
            logging.error(f"Corrupt image: {tiff_path}")
            return None
    else:
        logging.error("File doesn't exist")
        return None

    if resize_dims:
        image = resize(image, resize_dims, anti_aliasing=True)

    return image

def tiff_write(image, save_path):
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
