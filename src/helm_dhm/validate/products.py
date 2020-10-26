'''
Library for creating products in the HELM validate pipeline stage.
'''
import os
import glob
import logging
import signal
import subprocess
import skimage

import os.path           as op
import numpy             as np
import matplotlib.pyplot as plt

from pathlib import Path

from helm_dhm.validate import utils


def fourier_transform_image(image):
    """Compute a fourier transform of an image"""

    image_k = np.fft.fft2(image, axes=(0, 1))
    image_k = np.fft.fftshift(image_k, axes=(0, 1))
    power_image = np.absolute(image_k)
    log_power_image = np.log(power_image + 1)
    log_power_image = utils.scale_from_minmax(log_power_image)

    return log_power_image


def calc_max_diff(images):
    """Calculate the maximum 1-step time difference in an image stack"""

    if images.shape[2] > 1:
        return np.max(np.abs(np.diff(images, axis=-1)))
    return np.nan


def calc_subtracted_frames(images, subtract_arr):
    '''Subtract a 2D array from all images in a 3D image stack.'''

    # Cast subtraction array to `images` shape to broadcast subtraction
    return images - subtract_arr[:, :, np.newaxis]


def detect_defocus(path, images, threshold):
    """Report any possible defocus events to a text file and return them

    Parameters
    ----------
    path: str
        Directory and filepath. '_defocus.txt' will be added to end when saving
    images: numpy.ndarray
        Images to check for defocus
    threshold: float
        Threshold when checking contrast. This value sets the number of std.
        deviations the contrast can vary from the mean.

    Returns
    -------
    defocus_frames: list
        Boolean list of frames that met the defocused threshold
    """

    shifted = np.roll(images, 1, axis=1)
    shifted[:, -1, :] = images[:, -1, :]
    contrasts = np.sum(np.sum(np.abs(shifted - images), axis=0), axis=0)

    con_mean = contrasts.mean()
    con_std = contrasts.std()

    # Compare contrast against threshold
    defocus_frames = [frame for frame in range(len(contrasts))
                      if (contrasts[frame] > con_mean + threshold * con_std or
                          contrasts[frame] < con_mean - threshold * con_std)]

    # Write out the defocused frames to a text file and return
    if not defocus_frames:
        report = "No defocus problems detected for any images."
    else:
        report = "Defocus detected in the following images:\n" + ', '.join(str(f) for f in defocus_frames) + "\n"

    with open(path + "_defocus.txt", 'w') as txt_file:
        txt_file.write(report)

    return defocus_frames


def generate_text_report(save_fpath, bad_files, duplicate_frames,
                         total_frames, intensities, differences,
                         exp_is_dense, density_val):
    """Write some general information about the experiment to a txt file

    Parameters
    ----------
    save_fpath: str
        Full save filepath for text report.
    bad_files: list
        List of files that couldn't be loaded. Each item in the list should
        itself be an iterable of the index and filename.
    duplicate_frames: list
        List of files that were repeats. Each item in the list should itself be
        an iterable of the index and filename.
    total_frames: int
        Total number of images in the experiment.
    intensities: np.ndarray
        Per image intensity value.
    differences: np.ndarray
        Intensity difference between each image and previous image in the time
        series.
    exp_is_dense: bool
        Experiment exceeding particle density threshold and is considered dense.
    density_val: float
        Proportion of experiment's first image that is considered dense.
    """

    str_width = 25  # Number of chars/spaces to use during string formatting

    with open(save_fpath, 'w') as txt_file:
        # Write out hologram files that couldn't be loaded
        pct_bad_files = len(bad_files) / total_frames * 100
        txt_file.write(f'{"Loading errors:":<{str_width}} '
                       f'{len(bad_files)} unreadable hologram images ({pct_bad_files:0.2f} %)')
        for bad_file in bad_files:
            txt_file.write(f'\n\tBad image name:{bad_file[1]}, index:{bad_file[0]}')

        # Write out repeated hologram images
        pct_dup_frames = len(duplicate_frames) / total_frames * 100
        txt_file.write(f'\n{"Repeated hologram images: ":<{str_width}} '
                       f'{len(duplicate_frames)} duplicate hologram images ({pct_dup_frames:0.2f} %)')
        for dup_file in duplicate_frames:
            txt_file.write(f'\n\tDuplicate images name:{dup_file[1]}, index:{dup_file[0]}')

        # Write out density metrics
        txt_file.write(f'\n\n"Experiment exceeded density threshold: "{exp_is_dense}')
        txt_file.write(f'\n"Mean density estimate across all images: "{density_val}')

        # Write out dataset metrics
        txt_file.write('\n\nPer-image (and not per-pixel) statistics:')
        metrics_names_vals = [('Intensity mean:', intensities.mean()),
                              ('Intensity stddev:', intensities.std()),
                              ('Intensity min:', intensities.min()),
                              ('Intensity max:', intensities.max()),
                              ('Intensity change mean:', differences.mean()),
                              ('Intensity change stddev:', differences.std()),
                              ('Intensity change min:', differences.min()),
                              ('Intensity change max:', differences.max())]
        for metric, val in metrics_names_vals:
            txt_file.write(f'\n{metric:<{str_width}}{val:> 12.4f}')


def make_movie(save_fpath, images_dir, fname_temp="%4d.png"):
    """Make a movie from a directory of PNG images

    Parameters
    ----------
    save_fpath: str
        Filepath at which the movie will be saved. .mp4 suffix recommended.
    images_dir: str
        Directory path with image frames.
    fname_temp: str
        Template for image filenames found in images_dir.
        Default: "%4d.png", indicating 0001.png, 0002.png, etc.
    """

    # Get list of all images
    template_suffix = op.splitext(fname_temp)[-1]
    image_fpaths = [Path(fpath) for fpath in os.scandir(images_dir)
                    if Path(fpath).suffix == template_suffix]

    if len(image_fpaths) < 1:
        logging.warning("Not enough images found to make a movie.")
        return

    # Set output name and run an FFMPEG command on the command line
    ffmpeg_input = str(images_dir) + "/" + fname_temp
    ffmpeg_command = ['ffmpeg', '-framerate', '5', '-i', ffmpeg_input, '-y', '-vf', 'format=yuv420p', save_fpath]

    cmd = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = cmd.communicate()
    cmd.send_signal(signal.SIGINT)
    cmd.wait()


def make_gif(inputs, output):
    """Generate a gif from an existing movie

    Parameters
    ----------
    inputs: str
        Filepath of input movie to be converted.
    output: str
        Filepath of output gif to be generated.
        If it exists, it will be overwritten.
    """

    if os.path.exists(output):
        os.remove(output)
        logging.warning("Removing old gif")

    # Run the FFMPEG command to convert a movie to a gif
    command = ['ffmpeg', '-i', inputs, '-vf', 'fps=2,scale=256:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse', '-loop', '0', output]

    cmd = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = cmd.communicate()
    cmd.send_signal(signal.SIGINT)
    cmd.wait()

def make_trail_frames(input_dir, output_dir, max_diff_all, trail_length=5, ext=".png"):
    """Generate trail frames from existing frames

    Parameters
    ----------
    input_dir: str
        Path of the input directory with existing frames
    output_dir: str
        Path of the empty output directory where trail frames will be saved
    max_diff_all: int/float
        Maximum value for heatmap
    trail_length: 5
        Length of the trail in frames, including "current" frame. Defaults to 5
    ext: str
        Expected file extension of image frames. Defaults to ".png"
    """

    # Get image filepaths in the input dir
    files = sorted(glob.glob(op.join(input_dir, "*"+ext)))

    window = None
    for ind in range(len(files)):
        if window is None:
            # first image, read in the single image
            window, _skip = utils.read_images([files[ind]])
        else:
            # read in new image
            new, _skip = utils.read_images([files[ind]])
            # add it to current sliding window
            window = np.concatenate((window, new), axis=2)
            # if window size is greater than specified, drop oldest
            if window.shape[2] > trail_length:
                window = np.delete(window, 0, axis=2)

        # max over all images currently in the window
        window_max = np.amax(window, axis=2)

        # get 99.9 percentile to use as colormap max
        # TODO 7/13/2020 Jake: colormap max is set to the 99.9 percentile
        # value per frame. An equivalent global value should be found for
        # comparisons between experiments. See PR #237
        cmap_max = np.percentile(window, 99.9)

        # get filepath and save
        fpath = op.join(output_dir, f"{ind+1 :04d}.png")
        plt.imsave(fpath, window_max, cmap="viridis",
                    vmin=0, vmax=cmap_max)


def plot_duplicate_frames(save_fpath, image_count, mask_dup_frames,
                          num_dup_frames):
    """Given a set of potentially duplicate images, create a time series plot"""

    percentage = (num_dup_frames / image_count) * 100
    utils.plot_timeseries(save_fpath,
                          np.arange(image_count),
                          mask_dup_frames.astype(int),
                          x_label='Time index',
                          y_label='Duplicate',
                          title=f"{num_dup_frames}/{image_count} ({percentage:0.1f}%) images were duplicates",
                          binary=True)
    logging.info("Plotted time series of duplicate images over time")

def make_histogram(input_image, output_path):
    """Given an image, save its pixel histogram"""
    flat = input_image.flatten()
    fig, ax = plt.subplots()
    ax.hist(flat, bins=256, range=(0, 256))
    ax.set_title("Pixel Histogram (mean = {:.2f}, stddev = {:.2f})".format(np.mean(flat), np.std(flat)))
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Number of Pixels")

    fig.savefig(output_path)
    plt.close()


def blockwise_img_stdev(image, block_size):
    """Compute standard deviation in blocks of an image

    Parameters
    ----------
    image: np.ndarray
        Array to compute the standard deviation of pixel values for
    block_size: int
        Size of each block to chunk the image into.

    Returns
    -------
    std_vals: np.ndarray
        Standard deviation in each image block with shape (n_blocks x n_blocks)
    viz_std_image: np.ndarray
        Standard deviation in each image block but at the original `image`
        resolution. Std. dev. values are tiled to match the shape of `image`.
    """

    # Initialize two variables to track data
    std_vals = []
    viz_std_image = np.zeros_like(image)

    # Loop over image
    for row_val in range(0, image.shape[0], block_size):
        row_stds = []
        for col_val in range(0, image.shape[1], block_size):
            # Store standard deviation for one block
            row_stds.append(np.std(image[row_val:row_val+block_size,
                                         col_val:col_val+block_size]))
            # Store standard deviation w/out block downsampling
            viz_std_image[row_val:row_val+block_size,
                          col_val:col_val+block_size] = row_stds[-1]

        std_vals.append(row_stds)
    std_vals = np.array(std_vals)  # Convert to array

    return std_vals, viz_std_image


def estimate_density(first_image_arr, median_image_arr, config,
                     density_plot_fpath, density_gif_fpath):
    """Compute density metrics and create plots

    Parameters
    ----------
    first_image_arr
    median_image_arr
    config: dict
        HELM configuration dictionary
    density_plot_fpath: str
        Filepath to save density plot containing Std. Dev. vals to.
    density_gif_fpath: str
        Filepath to save density GIF showing original image and Std. Dev. vals.

    Returns
    -------
    density: float
        Value on [0, 1] that represents the proportion of blocks that met the
        density threshold (as defined in the config).
    """

    # Subtract background and downsize if needed
    orig_diff_image = first_image_arr - median_image_arr
    if orig_diff_image.shape != config['validate']['density_downscale_shape']:
        diff_image = skimage.transform.resize(orig_diff_image,
                                              config['validate']['density_downscale_shape'])
        downfactor = [orig // new for orig, new
                      in zip(orig_diff_image.shape[:2], config['validate']['density_downscale_shape'])]
    else:
        diff_image = orig_diff_image
        downfactor = [1, 1]

    # Get the standard deviation information
    std_vals, viz_std_image = blockwise_img_stdev(diff_image,
                                                  config['validate']['density_block_size'])

    # Create plot showing raw Std. Dev. vals
    utils.plot_labeled_density_estimate(density_plot_fpath, std_vals,
                                        config['validate']['density_thresh_block'])

    # Create GIF showing original image and Std. Dev. vals
    # Undo resizing using a Kronecker product to tile blocks of downscaled array
    viz_std_image = np.kron(viz_std_image, np.ones(downfactor, dtype=int))
    utils.plot_density_estimate_gif(density_gif_fpath, orig_diff_image, viz_std_image)

    # Compute/return proportion of image that's dense
    n_dense_blocks = np.sum(std_vals >= config['validate']['density_thresh_block'])
    return  n_dense_blocks / std_vals.size
