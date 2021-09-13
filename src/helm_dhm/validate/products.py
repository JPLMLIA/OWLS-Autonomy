'''
Library for creating products in the HELM validate pipeline stage.
'''
import os
import os.path as op
import glob
import logging
import signal
import subprocess
import csv
from pathlib import Path

import cv2
import skimage
import numpy             as np
import matplotlib.pyplot as plt
from skimage.transform import resize

from helm_dhm.validate import utils


def validate_fourier_transform(image, config):
    """Validate that a calculated fourier transform of an image
        matches the reference laser configuration

    Parameters
    ----------
    image: numpy.ndarray
        Fourier transform for the experiment being processed
    config: dict
        HELM configuration dictionary

    Returns
    -------
    status: bool
        Boolean indicating whether or not the DHM laser is configured as expected
    """

    ref_points = config["fourier_image"]["ref_points"]

    downsample_size = 100
    scaling_factor = image.shape[0] / downsample_size

    img = resize(image, (downsample_size, downsample_size)).astype(np.uint8)
    circles = cv2.HoughCircles(img,
                               cv2.HOUGH_GRADIENT,
                               config["fourier_image"]["params"]["dp"],
                               config["fourier_image"]["params"]["minDist"],
                               param1=config["fourier_image"]["params"]["param1"],
                               param2=config["fourier_image"]["params"]["param2"],
                               minRadius=config["fourier_image"]["params"]["minRadius"],
                               maxRadius=config["fourier_image"]["params"]["maxRadius"])

    if circles is not None:
        circles = circles[0]
        found_circles, coords = circles.shape

        # Check if the same number of lobes were found, if not fail
        if found_circles != len(ref_points):
            logging.warning("Failed to find sufficient fourier lobes.")
            return False, None

        confirmed_lobes = 0

        row_epsilon = config["fourier_image"]["params"]["row_epsilon"]
        col_epsilon = config["fourier_image"]["params"]["col_epsilon"]
        radius_epsilon = config["fourier_image"]["params"]["radius_epsilon"]
        intensity_epsilon = config["fourier_image"]["params"]["intensity_epsilon"]

        # Check if the lobes are within the expected regions/intensities
        for x in range(0, found_circles):
            for y in range(0, len(ref_points)):
                row_check = abs(circles[x,0] - ref_points[y]["row"]) < row_epsilon
                col_check = abs(circles[x,1] - ref_points[y]["col"]) < col_epsilon
                radius_check = abs(circles[x,2] - ref_points[y]["radius"]) < radius_epsilon
                intensity_check = abs(img[int(circles[x,0]), int(circles[x,1])] - ref_points[y]["intensity"]) < intensity_epsilon
                if row_check and col_check and radius_check and intensity_check:
                    confirmed_lobes += 1

        if confirmed_lobes != found_circles:
            logging.warning("Fourier lobes did not meet reference criteria.")
            return False, None

        circles = circles * scaling_factor
        return True, circles

    else:
        logging.warning("No fourier lobes identified.")
        return False, None


def fourier_transform_image(image, scale=True):
    """Compute the log power of a fourier transformed image"""

    image_k = np.fft.fft2(image, axes=(0, 1))
    image_k = np.fft.fftshift(image_k, axes=(0, 1))
    power_image = np.absolute(image_k)
    log_power_image = np.log(power_image + 1)

    if scale:
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


def data_quality_log_and_estimate(log_fpath, metric_fpath, bad_files,
                                  duplicate_frames, total_frames, intensities,
                                  differences, density_mean, fourier_valid,
                                  config):
    """Write results from data validation to a txt log and metric CSV

    Parameters
    ----------
    log_fpath: str
        Full filepath to save text report.
    metric_fpath: str
        Full filepath to save the data quality estimate (DQE).
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
    density_mean: float
        Proportion of experiment's first image that is considered dense.
    fourier_valid: bool
        Whether or not spectral plot passed automated checks.
    config: dict
        HELM config dictionary.
    """

    # Validation checks on mean intensity and mean difference
    pct_bad_files = len(bad_files) / total_frames * 100
    pct_dup_frames = len(duplicate_frames) / total_frames * 100

    i_lbound = config['validate']['intensity_lbound']
    i_ubound = config['validate']['intensity_ubound']
    d_lbound = config['validate']['diff_lbound']
    d_ubound = config['validate']['diff_ubound']

    intensity_mean = intensities.mean()
    diff_mean = differences.mean()
    intensity_valid = i_lbound <= intensity_mean <= i_ubound
    diff_valid = d_lbound <= diff_mean <= d_ubound
    density_valid = density_mean <= config['validate']['density_thresh_exp']

    ### Data quality estimate
    # Pull necessary weighting information from config
    weight_dict = config['data_quality']['weights']

    # Construct dictionary of bools representing validation check results
    measured_val_dict = {'intensity_valid': intensity_valid,
                         'diff_valid': diff_valid,
                         'density_valid': density_valid,
                         'fourier_valid': fourier_valid,
                         'no_duplicates': pct_dup_frames == 0,
                         'no_bad_files': pct_bad_files == 0}

    # Compute and save the health metric
    dqe = utils.weighted_mean_dicts(measured_val_dict, weight_dict)
    with open(metric_fpath, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['DQE'])
        writer.writeheader()
        writer.writerow({'DQE': dqe})

    ### Experiment log
    str_width = 32  # Number of chars/spaces to use during string formatting

    with open(log_fpath, 'w') as txt_file:
        # Write out hologram files that couldn't be loaded
        txt_file.write(f'{"Loading errors:":<{str_width}} '
                       f'{len(bad_files)} unreadable hologram images ({pct_bad_files:0.2f} %)')
        for bad_file in bad_files:
            txt_file.write(f'\n\tBad image name:{bad_file[1]}, index:{bad_file[0]}')

        # Write out repeated hologram images
        txt_file.write(f'\n{"Repeated hologram images: ":<{str_width}} '
                       f'{len(duplicate_frames)} duplicate hologram images ({pct_dup_frames:0.2f} %)')
        for dup_file in duplicate_frames:
            txt_file.write(f'\n\tDuplicate images name:{dup_file[1]}, index:{dup_file[0]}')

        # Write out intensity, difference, density metrics
        txt_file.write(f'\n\nMean intensity within expected bounds: {intensity_valid}')
        txt_file.write(f'\nMean frame difference within expected bounds: {diff_valid}')
        txt_file.write(f'\nMean density within expected bounds: {density_valid}')
        txt_file.write(f'\nFourier spectrum passed validation checks: {fourier_valid}')
        txt_file.write(f'\n\nData Quality Estimate: {dqe}')

        # Write out dataset metrics
        txt_file.write('\n\nPer-image (and not per-pixel) statistics:')
        metrics_names_vals = [('Intensity mean:', intensity_mean),
                              ('Intensity stddev:', intensities.std()),
                              ('Intensity min:', intensities.min()),
                              ('Intensity max:', intensities.max()),
                              ('Intensity change (diff) mean:', diff_mean),
                              ('Intensity change (diff) stddev:', differences.std()),
                              ('Intensity change (diff) min:', differences.min()),
                              ('Intensity change (diff) max:', differences.max()),
                              ('Density mean:', density_mean)]
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


def get_interframe_intervals(timestamps_fpath):
    """Calculate and the time between frame collections from a timestamps.txt file

    Parameters
    ----------
    timestamps_fpath: str
        Path to timestamps file. Expects 4 columns with timestamps in the 2nd
        column in format %H:%M:%S.%f

    Returns
    -------
    interframe_intervals: np.ndarray
        Time between frames in seconds
    """
    times = []

    with open(timestamps_fpath, 'r') as txt_file:
        for row in txt_file:
            str_time = row.split(' ')[3]
            times.append(float(str_time) / 1000)  # Convert from ms to s

        return np.diff(times)