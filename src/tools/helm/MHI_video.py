"""
Standalone script to generate a video from MHI products.
Each frame in the video is a unique argmax value.
"""

import sys, os
import os.path as op
from glob import glob
import argparse

from tqdm import tqdm
from PIL import Image
import numpy as np

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('mhi_ind',      help='Path to MHI index .npy file.')
    parser.add_argument('mhi_diff',     help='Path to MHI diff .npy file.')
    parser.add_argument('--outdir',     default='Holograms/',
                                        help='Output directory for generated video frames. Defaults to Holograms/')
    parser.add_argument('--n_frames',   default=-1,
                                        type=int,
                                        help='Number of frames in the original experiment. -1 uses the maximum index found in mhi-ind. Defaults to -1.')
    args = parser.parse_args()

    # Load MHI files
    mhi_ind = np.load(args.mhi_ind)
    mhi_diff = np.load(args.mhi_diff)

    # Determine # of frames in video
    if args.n_frames == -1:
        # Use max of mhi_ind
        n_frames = int(np.max(mhi_ind))
    else:
        n_frames = args.n_frames
    
    # Determine resolution of video
    dims = mhi_ind.shape

    # Initialize output directory
    if not op.isdir(args.outdir):
        os.mkdir(args.outdir)

    # Iterate through frames
    for i in tqdm(range(n_frames)):
        curr = np.zeros(dims)
        # Keep only diffs if it's currently the maximum value
        curr[mhi_ind == i] = mhi_diff[mhi_ind == i]

        # Save to output directory as 00001.tif, 00002.tif, etc.
        Image.fromarray(curr.astype(np.uint8)).save(op.join(args.outdir, f"{i:05d}.tif"))