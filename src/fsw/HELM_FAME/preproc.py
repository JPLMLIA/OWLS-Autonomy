'''
Functions for running the HELM/FAME preprocessing pipeline stage.
'''
import os
import logging
import multiprocessing
import glob

from tqdm                    import tqdm
from pathlib                 import Path

import numpy                 as np
import os.path               as op

from utils.dir_helper        import get_exp_subdir
from utils.file_manipulation import read_image
from utils.file_manipulation import write_image

def mp_resize(args):
    """ Multiprocess function for resizing """
    image = read_image(args['raw_path'],
                        raw_dims=args['raw_shape'],
                        resize_dims=args['resize_shape'],
                        flatten=args['flatten'])
    if image is None:
        image = np.zeros((args['resize_shape']))

    image = (image*255).astype(np.uint8)
    write_image(image, args['resize_path'])

def resize_holograms(holo_fpaths, outdir, raw_shape, resize_shape, n_workers=1):
    """ Writes resized holograms to output directory

    holo_fpaths: list of str
        List of filepaths to the hologram files
    outdir: str
        Path to output directory
    raw_shape: tuple
        Shape of raw image. Dim 2 for grayscale, 3 for RGB
    resize_shape: tuple
        Shape of resized image. Dim 2 for grayscale, 3 for RGB
    n_workers: int
        Number of cores for multiprocessing
    """

    # Setup multiprocessed resizing and saving
    mp_args = []
    for i in range(len(holo_fpaths)):
        arg = {
            'raw_path': holo_fpaths[i],
            'resize_path': op.join(outdir, Path(holo_fpaths[i]).stem + ".tif"),
            'raw_shape': raw_shape,
            'resize_shape': resize_shape,
            'flatten': len(resize_shape) == 2
        }
        mp_args.append(arg)
    
    with multiprocessing.Pool(n_workers) as pool:
        _ = list(tqdm(pool.imap_unordered(mp_resize, mp_args), total=len(holo_fpaths), desc='Resizing Holograms'))