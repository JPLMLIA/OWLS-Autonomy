'''
Functions for running the HELM/FAME validate pipeline stage.
'''
import os
import logging
import math
import multiprocessing
import glob
import csv
import PIL
import signal
import subprocess
import cv2
import skimage

import os.path           as op
import numpy             as np
import matplotlib.pyplot as plt

from tqdm                    import tqdm
from skimage.transform       import resize
from pathlib                 import Path
from fsw.HELM_FAME           import utils
from utils.dir_helper        import get_exp_subdir
from utils.file_manipulation import read_image

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
                                  duplicate_frames, dropped_frames, total_frames,
                                  intensities, differences, density_mean,
                                  fourier_valid, config):
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
    dropped_frames: list
        List of files that were dropped. Each item in the list should itself be
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
    pct_drop_frames = len(dropped_frames) / total_frames * 100

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
                         'no_drops': pct_drop_frames == 0,
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
        # Write out number of frames in experiment
        txt_file.write(f'Number of experiment frames: {total_frames}\n')
        # Write out frame dimensions
        txt_file.write(f'Frame dimensions: {config["raw_hologram_resolution"][0]} x '
                       f'{config["raw_hologram_resolution"][1]}\n')
        # Write out hologram files that couldn't be loaded
        txt_file.write(f'\n{"Loading errors:":<{str_width}} '
                       f'{len(bad_files)} unreadable hologram images ({pct_bad_files:0.2f} %)')
        for bad_file in bad_files:
            txt_file.write(f'\n\tBad image name:{bad_file[1]}, index:{bad_file[0]}')

        # Write out repeated hologram images
        txt_file.write(f'\n{"Repeated hologram images: ":<{str_width}} '
                       f'{len(duplicate_frames)} duplicate hologram images ({pct_dup_frames:0.2f} %)')
        for dup_file in duplicate_frames:
            txt_file.write(f'\n\tDuplicate images name:{dup_file[1]}, index:{dup_file[0]}')
        txt_file.write(f'\n{"Dropped hologram images: ":<{str_width}} '
                       f'{len(dropped_frames)} dropped hologram images ({pct_drop_frames:0.2f} %)')

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
    template_suffix = Path(fname_temp).suffix
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
            str_time = row.split()[3]
            times.append(float(str_time))

        return np.diff(times)

def calc_median_image(args):
    """Load a batch of images and calculate the single median image"""

    # Read and resize images
    images, _ = utils.read_images(args["file_batch"], args["raw_dims"])

    # Return median image
    return np.median(images, axis=2)


def load_image_batch(args):
    """Loads, preprocesses, and checks images"""

    ###################################
    # Read and resize images
    images, bad_files = utils.read_images(args["file_batch"], args["raw_dims"])
    args["bad_files"] = bad_files

    ###################################
    # Find duplicates
    images, args["mask_dup_frames"], args["num_dup_frames"] = utils.handle_duplicate_frames(images)

    ###################################
    # Calculate image differences
    args["diff"] = calc_max_diff(images)

    ###################################
    # Save near-min/near-max (without actually saving baseline-zeroed images)
    # Use percentile rather than true min/max to better handle a few noisy pixels
    baseline_sub_frames = calc_subtracted_frames(images, args["zeroing_image"])
    args['batch_nearmin'], args['batch_nearmax'] = \
        np.percentile(baseline_sub_frames, [args['vmin'], args['vmax']])

    return args


def multiprocess_image_batch(args):
    """Processing images for intensity, zeroing, and history of motion image

    Processing includes:
    1. computing total intensity and stddev of intensity per image
    2. computing total intensity differences and stddev of intensity difference per image
    3. computing differences between each image
    4. computing time index of max difference at each pixel
    5. computing an estimate of density per image

    Parameters
    ----------
    args: dict
        Input arguments for multiprocessing. This includes the following keys:
        `batch_index`, `file_index`, `zeroing_image`,
        `validation_dir`, `mp_batch_size`, `num_files`, `max_diff`,
        `holo_baseline_dir`, `holo_diff_dir`, `holo_diffvis_dir`,
        `baseline_vmin`, `baseline_vmax`, and `prepend_image_fpath`,
        `baseline_subtracted_ext`, `density_downscale_shape`,
        `density_block_size`, `density_thresh_block`.

    Returns
    -------
    results: dict
        The computation results added to the original `args` dict, which include
        `intensities`, `intensities_stddev`, `intensities_diff`,
        `intensities_diff_stddev`, `mhi_ind_image`, `mhi_val_image`, and
        `density_prop`.
    """

    config = args["config"]

    limit = args['file_index'] + args['mp_batch_size']
    if limit >= args["num_files"]:
        limit = args["num_files"]

    images, _ = utils.read_images(args["file_batch"], config["preproc_resolution"])

    ###################################
    # Calculate image mean/min/max intensities and stddev
    args["intensities_mean"] = np.mean(images, axis=(0, 1))
    args["intensities_stddev"] = np.std(images, axis=(0, 1))
    args["intensities_min"] = np.min(images, axis=(0, 1))
    args["intensities_max"] = np.max(images, axis=(0, 1))

    ###################################
    # Save zeroed frames
    baseline_sub_frames = calc_subtracted_frames(images, args["zeroing_image"])

    if not config['_space_mode']:
        ext = args['baseline_subtracted_ext']
        for count, ind in enumerate(range(args['file_index'], limit)):
            fpath = op.join(args["holo_baseline_dir"],
                            f'{ind + 1:04d}{ext}')  # Account for 1-indexing
            plt.imsave(fpath, baseline_sub_frames[:, :, count],
                        cmap=config['validate']['baseline_colormap'],
                        vmin=args['baseline_vmin'], vmax=args['baseline_vmax'])

    ###################################
    # Calculate standard deviation in image blocks as a density proxy

    std_vals = []
    for fi in range(baseline_sub_frames.shape[2]):
        baseline_sub_frame = np.copy(baseline_sub_frames[:, :, fi])
        # Downscale image if needed
        if baseline_sub_frame.shape != args['density_downscale_shape']:
            baseline_sub_frame = resize(baseline_sub_frame, args['density_downscale_shape'],
                                        anti_aliasing=True)

        # Compute block-wise standard deviation
        std_val, _ = blockwise_img_stdev(baseline_sub_frame, args['density_block_size'])
        std_vals.append(std_val)

    # Get number of dense blocks per image and record proportion of each frame that's dense
    n_dense_blocks = np.sum(np.array(std_vals) >= args['density_thresh_block'], axis=(1, 2))
    args['density_prop'] = n_dense_blocks / (std_vals[0].size)

    ###################################
    # Calculate motion history image (MHI)

    # Only false for first batch when prepend doesn't exist
    is_prepended_image = (args['prepend_image_fpath'] is not None)
    n_diffs = images.shape[2] - 1 + is_prepended_image

    if is_prepended_image:
        prepend_image, _ = utils.read_images(args['prepend_image_fpath'], config['preproc_resolution'])
        images = np.concatenate((prepend_image, images), axis=2)

    if n_diffs:
        # Get absolute difference between each image
        image_abs_diffs = np.abs(np.diff(images, axis=2))
        args['intensities_diff'] = np.mean(image_abs_diffs, axis=(0, 1))
        args['intensities_diff_stddev'] = np.std(image_abs_diffs, axis=(0, 1))

        # Calculate time index of that max val. Offset by batch num.
        batch_mhi_inds = np.argmax(image_abs_diffs, axis=2)
        args['mhi_ind_image'] = \
            batch_mhi_inds + \
            args['batch_index'] * args['mp_batch_size'] + 1

        # Store the actual value of largest diff
        args['mhi_val_image'] = np.take_along_axis(image_abs_diffs,
                                                   batch_mhi_inds[:, :, np.newaxis],
                                                   axis=2).squeeze()

    else:
        # Edge case with only single image. No diffs possible here
        args['mhi_ind_image'] = np.ones((images.shape[0], images.shape[1])) * args['file_index']
        args['mhi_val_image'] = np.zeros((images.shape[0], images.shape[1]))

    return args

def is_valid_image(path, target_res):
    '''Check that path is a valid image with resolution target_res'''
    image = read_image(path, target_res, flatten=False)
    if image is None:
        logging.warning("Skipping image {}: Failed to open".format(path))
        return False
    if not image.size > 0:
        logging.warning("Skipping image {}: Size = 0".format(path))
        return False
    if image.shape != target_res:
        logging.warning(
            "Skipping image {}: dimensions {} do not match expected {}"
            .format(path, image.shape, target_res))
        return False
    return True

def get_files(experiment, config):
    '''Returns a list of valid hologram file paths associated with experiment'''
    valid_exts = config['hologram_file_extensions']
    hdir = get_exp_subdir('hologram_dir', experiment, config)
    files = []

    # Filter image by extension
    for ext in valid_exts:
        files.extend(glob.glob(os.path.join(hdir, "*" + ext)))

    # Filter images by resolution/channels
    target_res = tuple(config['raw_hologram_resolution'])

    # Check if first image in sequence is correct resolution
    files = sorted(files)
    if not files:
        return []

    if is_valid_image(files[0], target_res):
        return files
    else:
        # don't throw an exception, a blank list will be caught
        return []

def get_preprocs(experiment, config):
    '''Returns a list of valid hologram file paths associated with experiment'''
    valid_exts = config['hologram_file_extensions']
    hdir = get_exp_subdir('preproc_dir', experiment, config)
    files = []

    # Filter image by extension
    for ext in valid_exts:
        files.extend(glob.glob(os.path.join(hdir, "*" + ext)))

    # Filter images by resolution/channels
    target_res = tuple(config['preproc_resolution'])

    # Check if first image in sequence is correct resolution
    files = sorted(files)
    if not files:
        return []

    if is_valid_image(files[0], target_res):
        return files
    else:
        # don't throw an exception, a blank list will be caught
        return []

def get_experiments(patterns, config):
    '''Return list of valid experiment dirs (according to config) matching any
    pattern in patterns

    Parameters
    ----------
    patterns : str
        Glob-able strings to match experiment dirs
    config : dict
        The loaded configuration yml for this run

    Returns
    -------
    experiments : list
        List of valid experiment dirs

    '''

    dirs = set()
    for pattern in patterns:
        curr_dirs = sorted([d for d in glob.glob(pattern) if op.isdir(Path(d))])
        dirs.update(curr_dirs)

    # Filter for valid hologram dir
    filtered_dirs = []
    for d in tqdm(dirs, desc="Verifying holo dirs"):
        if op.isdir(get_exp_subdir('hologram_dir', d, config)):
            filtered_dirs.append(op.realpath(d))
        else:
            logging.warning("Skipping experiment {}: Invalid hologram dir".format(d))

    experiments = []
    for exp in tqdm(filtered_dirs, desc="Verifying sequence lengths"):
        files = get_files(exp, config)

        # Ensure number of raw frames is within acceptable bounds
        min_hols = config['validate']['min_holograms']
        max_hols = config['validate']['max_holograms']
        if len(files) < min_hols:
            logging.warning("Skipping experiment {}: Number of valid images {} "
                        "does not meet minimum requirement {}".format(
                        exp, len(files), min_hols))
        elif len(files) > max_hols:
            logging.warning("Skipping experiment {}: Number of valid images {} "
                        "does not meet maximum requirement {}".format(
                        exp, len(files), max_hols))
        else:
            experiments.append(exp)

    return list(experiments)

def validate_data_flight(exp_dir, holo_fpaths, preproc_fpaths, config, instrument, n_workers=1, memory=None):
    """Run suite of algorithms to produce plots, text, and movies for holograms

    Parameters
    ----------
    exp_dir: str
        Path of working directory (usually the experiment folder)
    holo_fpaths: list
        Full filepaths to all hologram images
    preproc_fpaths: list
        Full filepaths to all preprocessed hologram images
    config: dictionary
        Expects the following keys under config['validate']
        mp_batch_size: int
            Number of hologram images to run in each multiprocessing batch
    instrument: string
        HELM | FAME - if HELM is set, a Fourier laser check will also be performed
    n_workers: int
        Number of processes to use in multiprocessing. The maximum value you
        should use is likely the number of cores on your machine
    memory
    """

    exp_name = Path(exp_dir).name

    logging.info(f'Validating {exp_name}')

    # Read config
    mp_batch_size = config['validate']['mp_batch_size']
    min_distinct_holograms = config['validate']['min_distinct_holograms']

    # Verify preprocessing step
    if len(preproc_fpaths) != len(holo_fpaths):
        logging.warning(f"Skipping {exp_name}: preproc frames ({len(preproc_fpaths)}) != raw frames ({len(holo_fpaths)})")
        return

    # Specify names of necessary directories
    validation_dir = Path(get_exp_subdir('validate_dir', exp_dir, config))
    holo_dir = Path(get_exp_subdir('hologram_dir', exp_dir, config))
    holo_baseline_dir = Path(get_exp_subdir('baseline_dir', exp_dir, config))

    first_image_orig_res = read_image(holo_fpaths[0], config['raw_hologram_resolution'])
    #first_image_orig_res = first_image_orig_res.squeeze()

    first_image = read_image(preproc_fpaths[0], config['preproc_resolution'])
    #first_image = first_image.squeeze()
    rows, cols = first_image.shape
    num_files = len(holo_fpaths)

    # Variables to help in dropped frame tracking
    #frame_nums = np.array([int(Path(holo_fpath).stem) for holo_fpath in holo_fpaths])
    #expected_frame_nums = np.arange(frame_nums[0], frame_nums[-1] + 1)  # Include last indexed frame
    #n_expected_frames = len(expected_frame_nums)

    if not config['_space_mode']:
        ###################################
        # Save out first image in the stack (RAW)
        plt.imsave(op.join(validation_dir, f'{exp_name}_first_image.png'),
                   first_image, cmap='gray')

        ###################################
        # Save out histogram of first image (RAW)
        make_histogram(first_image, op.join(validation_dir, f'{exp_name}_first_hist.png'))

    ###################################
    fourier_status = True
    if instrument == "HELM":

        # Compute, plot 2D Fourier transform of first image (RAW)
        log_power_image = fourier_transform_image(first_image_orig_res)

        fig, ax = plt.subplots()

        if "fourier_image" in config.keys():
            fourier_status, fourier_coords = validate_fourier_transform(log_power_image, config)
            logging.info(f'Laser configured correctly: {fourier_status}')

            if fourier_coords is not None:
                for x in range(0, len(fourier_coords)):
                    circle = fourier_coords[x]
                    ax.add_artist(plt.Circle((circle[0], circle[1]), circle[2], color='r', fill=False))

        ax.imshow(log_power_image)
        fig.savefig(op.join(validation_dir, f'{exp_name}_k_powerspec_orig.png'))
        ax.cla()
        logging.info(f'Saved fourier transform: {exp_name}_k_powerspec_orig.png')

    ###################################
    # Calculate median image of dataset (PREPROC)
    mp_batches = []

    # Compile multiprocessing batch information (just filenames and how to resize)
    for i in range(0, num_files, mp_batch_size):
        file_batch = preproc_fpaths[i:i + mp_batch_size]
        batch_info = {
            'file_batch': file_batch,
            'raw_dims': config['preproc_resolution']
        }
        mp_batches.append(batch_info)

    # Get the median image for each batch of images
    with multiprocessing.Pool(n_workers) as pool:
        results = list(tqdm(pool.imap_unordered(calc_median_image, mp_batches),
                            total=math.ceil(num_files / mp_batch_size),
                            desc='Calculate median'))

    # Take the median of medians to get a single image for baseline subtraction
    median_dataset_image = np.median(np.array(results), axis=0).astype(np.uint8)
    med_pil_img = PIL.Image.fromarray(median_dataset_image)
    med_pil_img.save(op.join(validation_dir, f'{exp_name}_median_image.tif'), compression='tiff_lzw')
    logging.info(f'Saved median image: {exp_name}_median_image.tif')

    ###########################################
    # Calculate particle density of first image and save related plots (PREPROC)
    density_first_img = estimate_density(first_image, median_dataset_image, config,
                                                  op.join(validation_dir, f'{exp_name}_density_first_image_stdevs.png'),
                                                  op.join(validation_dir, f'{exp_name}_density_first_image_viz.gif'))
    logging.info(f'Saved density of 1st frame: {exp_name}_density_first_image_*')

    ###################################
    # Read hologram images, calculate diffs, identify bad frames/duplicates (PREPROC)
    mp_batches = []
    for i in range(0, num_files, mp_batch_size):
        file_batch = preproc_fpaths[i:i + mp_batch_size]
        batch_info = {
            'file_batch': file_batch,
            'raw_dims': config['preproc_resolution'],
            'zeroing_image': first_image,
            'vmin': config['validate']['baseline_vmin'],
            'vmax': config['validate']['baseline_vmax']
        }
        mp_batches.append(batch_info)

    nearmin_baseline_intensity = []
    nearmax_baseline_intensity = []

    with multiprocessing.Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(load_image_batch, mp_batches),
                            total=math.ceil(num_files / mp_batch_size),
                            desc='Load images'))

    # Variables to hold high-level results
    total_dup_frames = 0
    full_mask_dup_frames = []
    max_diff = []
    bad_files = []

    # Iterate over multiproc results and store results for validation output
    for result in results:
        # Duplicate image information
        total_dup_frames += result['num_dup_frames']
        full_mask_dup_frames.extend(result['mask_dup_frames'])
        bad_files.extend(result['bad_files'])
        # Store max pixel difference
        max_diff.append(result['diff'])
        # Calculate the near-min and near-max intensity value
        nearmin_baseline_intensity.append(result['batch_nearmin'])
        nearmax_baseline_intensity.append(result['batch_nearmax'])

    distinct = num_files - total_dup_frames
    if distinct < min_distinct_holograms:
        logging.error(f"Skipping {exp_name}: # unique frames {distinct} < {min_distinct_holograms} req")
        return

    # Calculate the vmin/vmax for the zeroed hologram images (used to normalize the movie)
    baseline_zeroed_vmin = np.min(nearmin_baseline_intensity)
    baseline_zeroed_vmax = np.max(nearmax_baseline_intensity)

    max_diff = max(max_diff)
    if memory:
        memory.event.put('Max difference')

    ###################################
    # Investigate for any frame drops
    logging.info('Beginning check for dropped frames...')
    """
    # Get indices that were dropped
    dropped_inds = np.array(list(set(expected_frame_nums).difference(set(frame_nums))))
    n_frame_drops = len(dropped_inds)
    dropped_frame_info = [(drop_ind, f'{drop_ind:05d}.raw') for drop_ind in dropped_inds]  # Compile for text report

    # Get a binary trace of any dropped frames (for plotting)
    drop_trace = np.zeros((n_expected_frames))
    if dropped_inds:
        # Subtract ind of 0th frame, ensure we have int indices
        drop_trace[(dropped_inds - frame_nums[0]).astype(int)] = 1.

    # Calculate proportion of frames dropped, and warn if necessary
    frame_drop_proportion = n_frame_drops / n_expected_frames
    if n_frame_drops:
        logging.warning(f'{n_frame_drops} frame drops present. Drop rate: {frame_drop_proportion * 100:0.2f}%')

    # Save data as CSV
    x_vals, y_vals = expected_frame_nums, drop_trace
    x_label, y_label = 'Time Index', 'Frame dropped'
    savefpath_template = op.join(validation_dir,
                                 f'{exp_name}_timestats_dropped_frames')
    utils.save_timeseries_csv(np.column_stack((x_vals, y_vals)),
                              [x_label, y_label],
                              save_fpath=savefpath_template + '.csv')
    # Plot dropped frames
    utils.plot_timeseries(savefpath_template + '.png',
                          x_vals=x_vals,
                          y_vals=y_vals,
                          x_label=x_label,
                          y_label=y_label,
                          title=f'{n_frame_drops} frames were dropped ({frame_drop_proportion * 100:0.2f}%)',
                          binary=True)
    """
    logging.info(f'Saved dropped frames info: {exp_name}_timestats_dropped_frames.*')

    dropped_frame_info = []
    n_expected_frames = num_files
    ###################################
    # Generate a histogram of interframe intervals
    timestamp_fpath = op.join(exp_dir, 'timestamps.txt')
    if op.exists(timestamp_fpath):
        if_intervals = get_interframe_intervals(timestamp_fpath)

        # Calculate mean interval and mean frames per second
        mean_interval, mean_fps = np.mean(if_intervals), 1 / np.mean(if_intervals)
        # Set title and compute reasonable x limits
        title = 'Distribution of times between frames\n' \
                f'Mean interval: {mean_interval * 1000:0.2f} ms, Mean FPS: {mean_fps:0.2f}'
        x_lims = (np.max([0, np.min(if_intervals) - 0.25]),
                  np.max(if_intervals) + 0.25)
        utils.plot_histogram(op.join(validation_dir, f'{exp_name}_interframe_intervals.png'),
                             vals=if_intervals, bins=100, x_label='Interframe interval (s)',
                             y_label='Frame count', title=title, x_lims=x_lims)

        # Save CSV with this interframe intervals
        time_inds = np.arange(if_intervals.shape[0])
        utils.save_timeseries_csv(np.column_stack((time_inds, if_intervals)),
                                  ['Time index', 'Interframe interval (s)'],
                                  save_fpath=op.join(validation_dir,
                                                     f'{exp_name}_interframe_intervals.csv'),
                                  n_dec=6)
        logging.info(f'Saved interval check: {exp_name}_interframe_intervals.*')
    else:
        logging.warning(f'No timestamps file found at {timestamp_fpath}')

    ###################################
    # Compile multiprocessing batches for intensity/diff/MHI calculations
    mp_batches = []
    for batch_i, file_i in enumerate(range(0, num_files, mp_batch_size)):
        file_batch = preproc_fpaths[file_i:file_i + mp_batch_size]
        args = {"config":config,
                "batch_index":batch_i,
                "file_index":file_i,
                "file_batch":file_batch,
                "zeroing_image":median_dataset_image,
                "validation_dir":validation_dir,
                "mp_batch_size":mp_batch_size,
                "num_files":num_files,
                "max_diff":max_diff,
                "holo_baseline_dir":holo_baseline_dir,
                "baseline_vmin":baseline_zeroed_vmin,
                "baseline_vmax":baseline_zeroed_vmax,
                "prepend_image_fpath":None,
                "baseline_subtracted_ext": config['validate']['baseline_subtracted_ext'],
                "density_block_size": config['validate']['density_block_size'],
                "density_downscale_shape": config['validate']['density_downscale_shape'],
                "density_thresh_block": config['validate']['density_thresh_block']
                }
        # Prepend the last image of the previous batch for difference calcs
        if batch_i > 0:
            args["prepend_image_fpath"] = preproc_fpaths[file_i - 1]
        mp_batches.append(args)

    # Initialize various arrays that will be filled in with MP results
    intensities_mean_accum = np.empty((num_files))
    intensities_min_accum = np.empty((num_files))
    intensities_max_accum = np.empty((num_files))
    intensities_stddev_accum = np.empty((num_files))
    intensities_diff_accum = np.empty((num_files - 1))
    intensities_diff_stddev_accum = np.empty((num_files - 1))
    max_diff_per_batch = np.empty((rows, cols, len(mp_batches)), dtype=float)
    max_diff_ind_per_batch = np.empty((rows, cols, len(mp_batches)), dtype=int)
    density_accum = np.empty((num_files))

    # Run multiprocessing batches for intensity/diff/MHI calculations
    with multiprocessing.Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(multiprocess_image_batch, mp_batches),
                            total=math.ceil(num_files / mp_batch_size),
                            desc='Int, Diff, Dens, MHI'))

    # Calculate/store various image stats for plotting
    for result in results:

        # Compile intensity information
        start_i = result['file_index']
        end_i = np.minimum(start_i + mp_batch_size, num_files)
        intensities_mean_accum[start_i:end_i] = result['intensities_mean']
        intensities_min_accum[start_i:end_i] = result['intensities_min']
        intensities_max_accum[start_i:end_i] = result['intensities_max']
        intensities_stddev_accum[start_i:end_i] = result['intensities_stddev']

        # Compile difference in intensity information
        # Need start and index 0 and end at end a single index shorter than intensity
        diff_start_i = 0 if result['batch_index'] == 0 else start_i - 1
        diff_end_i = end_i - 1
        intensities_diff_accum[diff_start_i:diff_end_i] = result['intensities_diff']
        intensities_diff_stddev_accum[diff_start_i:diff_end_i] = result['intensities_diff_stddev']

        # Store MHI information
        max_diff_per_batch[:, :, result["batch_index"]] = result["mhi_val_image"]
        max_diff_ind_per_batch[:, :, result["batch_index"]] = result["mhi_ind_image"]

        # Store proxy for particle density at each frame
        density_accum[start_i:end_i] = result['density_prop']

    density_mean = np.mean(density_accum)
    ###################################
    # Create motion history image

    # Find batch ind with biggest difference
    max_diff_batch_ind = np.argmax(max_diff_per_batch, axis=2)
    # Use batch ind to get the time index
    max_diff_ind_all = np.take_along_axis(max_diff_ind_per_batch,
                                          max_diff_batch_ind[:, :, np.newaxis],
                                          axis=2).squeeze()
    max_diff_all = np.max(max_diff_per_batch, axis=2).squeeze()

    if memory:
        memory.event.put('Zeroing + history of motion')

    # Plot time index of largest change
    # Creates labeled plot and unlabeled raw image
    utils.plot_mhi_image(op.join(validation_dir, f'{exp_name}_mhi_labeled.png'),
                         max_diff_ind_all,
                         'Largest pixel change',
                         max_diff_all,
                         cmap=utils.get_aug_rainbow(),
                         savepath_unlabeled_img=op.join(validation_dir,
                                                        f'{exp_name}_mhi.jpg'),
                         savepath_numpy_array=op.join(validation_dir,
                                                      f'{exp_name}_mhi.npy'))

    logging.info(f'Saved Motion History Image: {exp_name}_mhi.*')

    if not config['_space_mode']:

        # Plot time-series of mean intensity values
        x_vals, y_vals = np.arange(intensities_mean_accum.shape[0]), intensities_mean_accum
        x_label, y_label = 'Time index', 'Mean intensity'
        savefpath_template = op.join(validation_dir,
                                     f'{exp_name}_timestats_mean_intensity')

        # Save normal time series plots
        stddevs = np.row_stack((intensities_mean_accum - intensities_stddev_accum,
                                intensities_mean_accum + intensities_stddev_accum))

        # Hlines for thresholds
        intensity_hlines = []
        intensity_hlines.append({
            'y': config['validate']['intensity_lbound'],
            'xmin': 0, 'xmax': len(x_vals),
            'colors': 'r', 'linestyles': 'dashed', 'label': 'threshold'
        })

        intensity_hlines.append({
            'y': config['validate']['intensity_ubound'],
            'xmin': 0, 'xmax': len(x_vals),
            'colors': 'r', 'linestyles': 'dashed'
        })

        utils.plot_timeseries(savefpath_template + '.png',
                              x_vals, y_vals,
                              x_label, y_label,
                              f'Intensity per image across {y_vals.shape[0]} images',
                              binary=False,
                              show_mean=True,
                              hlines=intensity_hlines,
                              stddev_bands=stddevs)

    # Save data as CSV
    x_vals, y_vals = np.arange(intensities_mean_accum.shape[0]), intensities_mean_accum
    x_label, y_label = 'Time index', 'Mean intensity'
    savefpath_template = op.join(validation_dir,
                                 f'{exp_name}_timestats_mean_intensity')
    utils.save_timeseries_csv(np.column_stack((x_vals, y_vals)),
                              [x_label, y_label],
                              save_fpath=savefpath_template + '.csv')

    logging.info(f'Saved mean intensity: {exp_name}_timestats_mean_intensity.*')

    ###################################

    # Save max intensity as CSV for ASDP data visualization step
    utils.save_timeseries_csv(np.column_stack((np.arange(intensities_max_accum.shape[0]), intensities_max_accum)),
                              ['Time index', 'Max intensity'],
                              save_fpath=op.join(validation_dir,
                                                 f'{exp_name}_timestats_max_intensity.csv'))

    logging.info(f'Saved max intensity: {exp_name}_timestats_max_intensity.csv')

    ###################################
    # Plot time-series of intensity differences
    x_vals, y_vals = np.arange(1, intensities_diff_accum.shape[0] + 1), intensities_diff_accum
    stddevs = np.row_stack((intensities_diff_accum - intensities_diff_stddev_accum,
                            intensities_diff_accum + intensities_diff_stddev_accum))

    x_label, y_label = 'Time index', 'Mean pixel intensity difference'
    savefpath_template = op.join(validation_dir,
                                 f'{exp_name}_timestats_pixeldiff')

    # Hlines for thresholds
    diff_hlines = []
    diff_hlines.append({
        'y': config['validate']['diff_lbound'],
        'xmin': 0, 'xmax': len(x_vals),
        'colors': 'r', 'linestyles': 'dashed', 'label': 'threshold'
    })

    diff_hlines.append({
        'y': config['validate']['diff_ubound'],
        'xmin': 0, 'xmax': len(x_vals),
        'colors': 'r', 'linestyles': 'dashed'
    })

    # Plot normal time series info
    if not config['_space_mode']:
        utils.plot_timeseries(savefpath_template + '.png',
                              x_vals, y_vals,
                              x_label, y_label,
                              f'Intensity change per image across {y_vals.shape[0]} images',
                              binary=False,
                              show_mean=True,
                              hlines=diff_hlines,
                              stddev_bands=stddevs)

    # Save data as CSV
    utils.save_timeseries_csv(np.column_stack((x_vals, y_vals)),
                              [x_label, y_label],
                              save_fpath=savefpath_template + '.csv')

    logging.info(f'Saved intensity diff: {exp_name}_timestats_pixeldiff.*')

    ###################################
    # Plot time-series of image density
    x_vals, y_vals = np.arange(density_accum.shape[0]), density_accum

    x_label, y_label = 'Time index', 'Proportion of dense image blocks'
    savefpath_template = op.join(validation_dir,
                                 f'{exp_name}_timestats_density')

    hlines = []
    hlines.append({
        'y': config['validate']['density_thresh_exp'],
        'xmin': 0, 'xmax': len(x_vals),
        'colors': 'r', 'linestyles': 'dashed', 'label': 'Mean density\nthreshold'
    })

    if not config['_space_mode']:
        utils.plot_timeseries(savefpath_template + '.png',
                              x_vals, y_vals,
                              x_label, y_label,
                              f'Estimated particle density across {y_vals.shape[0]} images',
                              binary=False,
                              show_mean=True,
                              hlines=hlines)

    # Save data as CSV
    utils.save_timeseries_csv(np.column_stack((x_vals, y_vals)),
                              [x_label, y_label],
                              save_fpath=savefpath_template + '.csv')

    logging.info(f'Saved density: {exp_name}_timestats_density.*')

    ###################################
    # Make animations
    if not config['_space_mode']:
        if Path(holo_fpaths[0]).suffix == ".tif":
            # Generate .mp4 movie of original hologram frames
            movie_orig_fpath = op.join(validation_dir, f'{exp_name}_orig_movie.mp4')
            make_movie(movie_orig_fpath, holo_dir, fname_temp="%5d_holo.tif")
            # Generate quicklook gif of original hologram frames
            gif_orig_fpath = op.join(validation_dir, f'{exp_name}_orig_movie.gif')
            make_gif(movie_orig_fpath, gif_orig_fpath)

            logging.info(f'Saved video: {exp_name}_orig_movie.*')

        # Generate .mp4 movie of median subtracted frames
        movie_base_fpath = op.join(validation_dir, f'{exp_name}_base_movie.mp4')
        make_movie(movie_base_fpath, holo_baseline_dir)
        # Generate quicklook gif of median subtracted frames
        gif_base_fpath = op.join(validation_dir, f'{exp_name}_base_movie.gif')
        make_gif(movie_base_fpath, gif_base_fpath)

        logging.info(f'Saved video: {exp_name}_base_movie.*')

    ###################################
    # Generate high level text report

    # Get duplicate frame names and indices
    dup_frame_info = [(fname, ind) for ind, (fname, is_dup)
                      in enumerate(zip(holo_fpaths, full_mask_dup_frames))
                      if is_dup]

    data_quality_log_and_estimate(op.join(validation_dir,
                                                   f'{exp_name}_processing_report.txt'),
                                           op.join(validation_dir,
                                                   f'{exp_name}_dqe.csv'),
                                           bad_files,
                                           dup_frame_info,
                                           dropped_frame_info,
                                           n_expected_frames,
                                           intensities_mean_accum,
                                           intensities_diff_accum,
                                           density_mean,
                                           fourier_status,
                                           config)

    logging.info(f"Saved data quality report {exp_name}_processing_report.txt")
    logging.info(f"Saved data quality estimate {exp_name}_dqe.csv")

def validate_data_ground(exp_dir, holo_fpaths, preproc_fpaths, config, instrument, n_workers=1, memory=None):
    """Run suite of algorithms to produce plots, text, and movies for holograms

    Parameters
    ----------
    exp_dir: str
        Path of working directory (usually the experiment folder)
    holo_fpaths: list
        Full filepaths to all hologram images
    preproc_fpaths: list
        Full filepaths to all preprocessed hologram images
    config: dictionary
        Expects the following keys under config['validate']
        mp_batch_size: int
            Number of hologram images to run in each multiprocessing batch
    instrument: string
        HELM | FAME - if HELM is set, a Fourier laser check will also be performed
    n_workers: int
        Number of processes to use in multiprocessing. The maximum value you
        should use is likely the number of cores on your machine
    memory
    """

    exp_name = Path(exp_dir).name
    validation_dir = Path(get_exp_subdir('validate_dir', exp_dir, config))
    num_files = len(holo_fpaths)


def global_stats(exp_dirs, out_dir, config):
    """ Generate global plots and statistics given a set of experiments

    Parameters
    ----------
    exp_dirs: list
        List of experiment directory paths. Likely returned from glob.
    out_dir: string
        Directory path in which to store generated products and logs.
    config: dictionary
        Expects the following keys under config['validate']
        intensity_lbound: float
            Lower bound threshold for the intensity metric.
        intensity_ubound: float
            Upper bound threshold for the intensity metric.
        diff_lbound: float
            Lower bound threshold for the difference metric.
        diff_ubound: float
            Upper bound threshold for the difference metric.
    """

    ### VARIABLE INITIALIZATION
    # lists to store intensities and pixeldiffs from all exps
    global_intensities = []
    global_diffs = []
    global_pairs =  []

    # list to store threshold violation log
    threshold_logs = [["experiment path", "intensity viols", "difference viols"]]

    # thresholds defined in config
    i_lbound = config['validate']['intensity_lbound']
    i_ubound = config['validate']['intensity_ubound']
    d_lbound = config['validate']['diff_lbound']
    d_ubound = config['validate']['diff_ubound']


    ### READING EXPERIMENT CSVS
    for exp in exp_dirs:
        # build filenames
        exp_name = Path(exp).name
        validate_dir = get_exp_subdir('validate_dir', exp, config)
        int_csv_fn = op.join(validate_dir, exp_name+"_timestats_mean_intensity.csv")
        diff_csv_fn = op.join(validate_dir, exp_name+"_timestats_pixeldiff.csv")

        # read intensity CSV
        curr_i = []     # curr exp intensity values
        viol_i = -1      # number of times intensity threshold was violated
        if op.exists(int_csv_fn):
            viol_i = 0
            with open(int_csv_fn, 'r') as i_file:
                i_reader = csv.reader(i_file)
                # skip the header row
                next(i_reader)
                for row in i_reader:
                    val = float(row[1])
                    curr_i.append(val)
                    # check if this value violates set thresholds
                    if val < i_lbound or val > i_ubound:
                        viol_i += 1
        else:
            logging.warning(f"No intensity statistics found for {exp_name}")

        # read difference CSV
        curr_d = []     # curr exp difference values
        viol_d = -1      # number of times diff threshold was violated
        if op.exists(diff_csv_fn):
            viol_d = 0
            with open(diff_csv_fn, 'r') as d_file:
                d_reader = csv.reader(d_file)
                # skip the header row
                next(d_reader)
                for row in d_reader:
                    val = float(row[1])
                    curr_d.append(val)
                    # check if this value violates set thresholds
                    if val < d_lbound or val > d_ubound:
                        viol_d += 1
        else:
            logging.warning(f"No difference statistics found for {exp_name}")

        # add exp metrics to global lists
        global_intensities += curr_i
        global_diffs += curr_d
        for i in range(len(curr_d)):
            # pair up intensity and difference frames for scatterplotting
            # first intensity frame doesn't have a diff frame so offset by one
            global_pairs.append([curr_i[i+1], curr_d[i]])
        threshold_logs.append([exp, viol_i, viol_d])


    ### SAVING THRESHOLD LOGS
    # export threshold log
    log_fn = op.join(out_dir, "threshold_viols.csv")
    with open(log_fn, 'w', newline='') as lf:
        log_writer = csv.writer(lf)
        log_writer.writerows(threshold_logs)

    ### SAVING GLOBAL STATS
    # convert stats to numpy arrays
    global_intensities = np.array(global_intensities)
    global_diffs = np.array(global_diffs)
    global_pairs = np.array(global_pairs)

    # calculate and write global statistics
    global_stats = []
    if global_intensities.size != 0:
        global_stats.append(["Mean Intensity", np.mean(global_intensities)])
        global_stats.append(["Median Intensity", np.median(global_intensities)])
        global_stats.append(["Std Intensity", np.std(global_intensities)])
    else:
        logging.warning("No global intensity statistics")

    if global_diffs.size != 0:
        global_stats.append(["Mean Difference", np.mean(global_diffs)])
        global_stats.append(["Median Difference", np.median(global_diffs)])
        global_stats.append(["Std Difference", np.std(global_diffs)])
    else:
        logging.warning("No global difference statistics")

    stats_fn = op.join(out_dir, "batch_stats.txt")
    with open(stats_fn, 'w', newline='') as sf:
        stats_writer = csv.writer(sf)
        stats_writer.writerows(global_stats)

    ### SAVING HISTOGRAMS
    if not config['_space_mode']:
        if global_intensities.size != 0:
            int_vlines = []
            int_vlines.append({
                'x': config['validate']['intensity_lbound'],
                'colors': 'r', 'linestyles': 'dashed', 'label': 'threshold'
            })

            int_vlines.append({
                'x': config['validate']['intensity_ubound'],
                'colors': 'r', 'linestyles': 'dashed'
            })

            ihist_fn = op.join(out_dir, "intensity_hist.png")
            utils.plot_histogram(ihist_fn, global_intensities, bins=100,
                                x_label="Intensity", y_label="Frames",
                                title="Image Intensity Distribution (n = {})".format(global_intensities.shape[0]),
                                vlines=int_vlines)
        else:
            logging.warning("No global intensity statistics to plot")

        if global_diffs.size != 0:
            diff_vlines = []
            diff_vlines.append({
                'x': config['validate']['diff_lbound'],
                'colors': 'r', 'linestyles': 'dashed', 'label': 'threshold'
            })

            diff_vlines.append({
                'x': config['validate']['diff_ubound'],
                'colors': 'r', 'linestyles': 'dashed'
            })

            dhist_fn = op.join(out_dir, "difference_hist.png")
            utils.plot_histogram(dhist_fn, global_diffs, bins=100,
                                x_label="Intensity Difference from Previous Frame", y_label="Frames",
                                title="Intensity Difference Distribution (n = {})".format(global_diffs.shape[0]),
                                vlines=diff_vlines)
        else:
            logging.warning("No global difference statistics to plot")

        ### SAVING SCATTERPLOT
        if len(global_pairs.shape) == 2:
            int_vlines = []
            int_vlines.append({
                'x': config['validate']['intensity_lbound'],
                'colors': 'r', 'linestyles': 'dashed', 'label': 'threshold'
            })

            int_vlines.append({
                'x': config['validate']['intensity_ubound'],
                'colors': 'r', 'linestyles': 'dashed'
            })
            diff_hlines = []
            diff_hlines.append({
                'y': config['validate']['diff_lbound'],
                'colors': 'r', 'linestyles': 'dashed', 'label': 'threshold'
            })

            diff_hlines.append({
                'y': config['validate']['diff_ubound'],
                'colors': 'r', 'linestyles': 'dashed'
            })


            pair_fn = op.join(out_dir, "intdiffpair_scatter.png")
            utils.plot_scatter(pair_fn, global_pairs[:,0], global_pairs[:,1],
                                x_label="Intensity", y_label="Difference from Previous",
                                title="Per-Frame Metric Scatterplot (n = {})".format(global_pairs.shape[0]),
                                vlines=int_vlines, hlines=diff_hlines)
        else:
            logging.warning(f"Not enough global statistics dimensions ({len(global_pairs.shape)}) for scatterplot")
