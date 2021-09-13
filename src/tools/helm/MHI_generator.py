"""
Standalone script to calculate various MHI products.

The approach taken here is slower, but it uses less memory at once.
Loading all frames into the script at once would use ~1.5GB of memory.

Output files:

mhi_orig_plot.png
    The original MHI plot - argmax(diff) plotted over a black background with 
    transparency max(diff).
    Uses the --mhi_cmap colormap.

mhi_orig.png
    The original MHI plot without axes and colorbars.
    Uses the --mhi_cmap colormap.

mhi_ind_plot.png
mhi_ind.npy
    argmax(diff) directly plotted and saved.
    Uses the --debug_cmap colormap.

mhi_diff_plot.png
mhi_diff.npy
    max(diff) directly plotted and saved.
    Uses the --debug_cmap colormap.

mhi_f0int_plot.png
mhi_f0int.npy
    Whenever the abs diff between frames f0 and f1 is the largest, this array
    saved the pixel value of f0.
    Uses the --debug_cmap colormap.

mhi_f1int_plot.png
mhi_f1int.npy
    Whenever the abs diff between frames f0 and f1 is the largest, this array
    saved the pixel value of f1.
    Uses the --debug_cmap colormap.

mhi_maxint_plot.png
mhi_maxint.npy
    Whenever the abs diff between frames f0 and f1 is the largest, this array
    saved the pixel value max(f0, f1).
    Uses the --debug_cmap colormap.

mhi_minint_plot.png
mhi_minint.npy
    Whenever the abs diff between frames f0 and f1 is the largest, this array
    saved the pixel value min(f0, f1).
    Uses the --debug_cmap colormap.

v1: 07/27/2021 Jake Lee, jake.h.lee@jpl.nasa.gov
"""
import sys, os
import os.path as op
from glob import glob
import argparse
from matplotlib.colors import Normalize

from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

if __name__ == "__main__":

    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('holo_dir',     help='Path to hologram directory. Filenames inside should be sortable.')

    parser.add_argument('--holo_ext',   default='.tif',
                                        help='File extension for holograms. Default .tif')
    
    parser.add_argument('--mhi_cmap',   default='gist_rainbow',
                                        help='Colormap for the plotted MHI. Default gist_rainbow')

    parser.add_argument('--debug_cmap', default='viridis',
                                        help='Colormap for the debug plots. Default viridis')

    args = parser.parse_args()

    # Get list of all hologram files, sorted
    holo_fpaths = sorted(glob(op.join(args.holo_dir, '*'+args.holo_ext)))
    if len(holo_fpaths) < 2:
        print(f'Error: Found {len(holo_fpaths)} frames of filetype {args.holo_ext} in {args.holo_dir}')
        print('Error: Could not find enough holograms for MHI')
        sys.exit(1)

    # Open the first image to get hologram resolution
    holo_res = np.array(Image.open(holo_fpaths[0])).shape

    # Define MHI products
    mhi_ind = np.zeros(holo_res)        # frame index @ max diff
    mhi_diff = np.zeros(holo_res)       # diff value @ max diff
    mhi_0val = np.zeros(holo_res)       # frame 0 value @ max diff
    mhi_1val = np.zeros(holo_res)       # frame 1 value @ max diff
    mhi_maxval = np.zeros(holo_res)     # max(f0, f1) value @ max diff
    mhi_minval = np.zeros(holo_res)     # min(f0, f1) value @ max diff

    # For frames 1 to n (frame 0 has no diff from prev frame)
    for i in tqdm(range(1, len(holo_fpaths))):
        # Define the two frames for diff
        frame_0 = np.array(Image.open(holo_fpaths[i-1])).astype(np.int)
        frame_1 = np.array(Image.open(holo_fpaths[i])).astype(np.int)

        # Calculate the absolute difference
        curr_diff = np.abs(frame_0 - frame_1)

        # Only update the products where the current diff is larger than history
        mhi_ind[curr_diff > mhi_diff] = i
        mhi_0val[curr_diff > mhi_diff] = frame_0[curr_diff > mhi_diff]
        mhi_1val[curr_diff > mhi_diff] = frame_1[curr_diff > mhi_diff]
        mhi_maxval[curr_diff > mhi_diff] = np.maximum(frame_0, frame_1)[curr_diff > mhi_diff]
        mhi_minval[curr_diff > mhi_diff] = np.minimum(frame_0, frame_1)[curr_diff > mhi_diff]

        # Update largest diff values seen so far
        mhi_diff = np.maximum(mhi_diff, curr_diff)


    ### ORIGINAL MHI IMAGE
    # Indices are plotted on top of black background with alpha values
    # varying depending on how large the difference itself was

    # Get colors
    MHI_colors = mpl.cm.get_cmap(args.mhi_cmap)(mpl.colors.Normalize()(mhi_ind))

    # Get transparencies
    # Clip to percentile to avoid too large of a distribution
    _upper = np.percentile(mhi_diff, 99)
    MHI_alphas = Normalize(0, _upper, clip=True)(mhi_diff)
    MHI_alphas = np.clip(MHI_alphas, 0.25, 1)

    # Add transparencies channel to colors
    MHI_colors[..., -1] = MHI_alphas

    # Black background
    MHI_image = Image.new('RGBA', size=holo_res, color=(0,0,0,255))

    # Combine MHI colors with background
    MHI_image.alpha_composite(Image.fromarray(img_as_ubyte(MHI_colors)))

    # Save original MHI with colorbar, etc.
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(MHI_image, cmap=args.mhi_cmap, vmin=np.min(mhi_ind), vmax=np.max(mhi_ind))
    fig.colorbar(im, ax=ax)
    ax.set_title('Original MHI Image')
    fig.savefig('mhi_orig_plot.png')
    plt.close()

    # Save original MHI raw
    MHI_image.save('mhi_orig.png')


    ### MHI IND PLOT
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(mhi_ind, cmap=args.debug_cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title('MHI Ind Plot')
    fig.savefig('mhi_ind_plot.png')
    plt.close()

    # Save array
    np.save('mhi_ind.npy', mhi_ind)


    ### MHI DIFF PLOT
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(mhi_diff, cmap=args.debug_cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title('MHI Diff Plot')
    fig.savefig('mhi_diff_plot.png')
    plt.close()

    # Save array
    np.save('mhi_diff.npy', mhi_diff)


    ### MHI 0 PLOT
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(mhi_0val, cmap=args.debug_cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title('MHI Frame0 Intensity Plot')
    fig.savefig('mhi_f0int_plot.png')
    plt.close()

    # Save array
    np.save('mhi_f0int.npy', mhi_0val)


    ### MHI 1 PLOT
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(mhi_1val, cmap=args.debug_cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title('MHI Frame1 Intensity Plot')
    fig.savefig('mhi_f1int_plot.png')
    plt.close()

    # Save array
    np.save('mhi_f1int.npy', mhi_1val)


    ### MHI MAX(0,1) PLOT
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(mhi_maxval, cmap=args.debug_cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title('MHI MAX Intensity Plot')
    fig.savefig('mhi_maxint_plot.png')
    plt.close()

    # Save array
    np.save('mhi_maxint.npy', mhi_maxval)

    ### MHI MIN(0,1) PLOT
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(mhi_minval, cmap=args.debug_cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title('MHI MIN Intensity Plot')
    fig.savefig('mhi_minint_plot.png')
    plt.close()

    # Save array
    np.save('mhi_minint.npy', mhi_minval)