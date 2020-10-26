'''
Processing helper functions for running the HELM validate pipeline stage.
'''
import os
import logging
import math
import multiprocessing
import glob
import csv
import PIL

import os.path           as op
import numpy             as np
import matplotlib.pyplot as plt

from tqdm              import tqdm
from skimage.transform import resize
from pathlib           import Path
from helm_dhm.validate import utils
from helm_dhm.validate import products
from utils.dir_helper  import get_exp_subdir

def calc_median_image(args):
    """Load a batch of images and calculate the single median image"""

    # Read and resize images
    images, _ = utils.read_images(args["file_batch"], args["resize_ratio"])

    # Return median image
    return np.median(images, axis=2)


def load_image_batch(args):
    """Loads, preprocesses, and checks images"""

    ###################################
    # Read and resize images
    images, bad_files = utils.read_images(args["file_batch"],
                                          args["resize_ratio"])
    args["bad_files"] = bad_files

    ###################################
    # Find duplicates
    images, args["mask_dup_frames"], args["num_dup_frames"] = utils.handle_duplicate_frames(images)

    ###################################
    # Calculate image differences
    args["diff"] = products.calc_max_diff(images)

    ###################################
    # Save near-min/near-max (without actually saving baseline-zeroed images)
    # Use percentile rather than true min/max to better handle a few noisy pixels
    baseline_sub_frames = products.calc_subtracted_frames(images,
                                                          args["zeroing_image"])
    args['batch_nearmin'], args['batch_nearmax'] = \
        np.percentile(baseline_sub_frames, [0.1, 99.9])

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
        `batch_index`, `file_index`, `resize_ratio`, `zeroing_image`,
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

    limit = args['file_index'] + args['mp_batch_size']
    if limit >= args["num_files"]:
        limit = args["num_files"]

    images, _ = utils.read_images(args["file_batch"], args['resize_ratio'])

    ###################################
    # Calculate image intensities and stddev
    args["intensities"] = np.mean(images, axis=(0, 1))
    args["intensities_stddev"] = np.std(images, axis=(0, 1))

    ###################################
    # Save zeroed frames
    baseline_sub_frames = products.calc_subtracted_frames(images,
                                                          args["zeroing_image"])
    ext = args['baseline_subtracted_ext']
    for count, ind in enumerate(range(args['file_index'], limit)):
        fpath = op.join(args["holo_baseline_dir"],
                        f'{ind + 1:04d}{ext}')  # Account for 1-indexing
        plt.imsave(fpath, baseline_sub_frames[:, :, count], cmap="viridis_r",
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
        std_val, _ = products.blockwise_img_stdev(baseline_sub_frame,
                                                  args['density_block_size'])
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
        prepend_image, _ = utils.read_images(args['prepend_image_fpath'],
                                             args['resize_ratio'])
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

        # Save diff frames to diff directory
        for count, ind in enumerate(range(args['file_index'], limit)):
            if not is_prepended_image:
                # first batch - first image can't have a diff, so indices shift by 1
                if count == 0:
                    continue
                else:
                    #fpath = op.join(args['holo_diff_dir'], f"{ind :04d}.png")
                    # since this is an intermediate step to trailing video, no heatmap
                    #sio.imsave(fpath, image_abs_diffs[:, :, count-1].astype(np.uint8),
                    #            check_contrast=False)
                    # saved in case diff frames are final vis product and we want heatmaps
                    vpath = op.join(args['holo_diffvis_dir'], f"{ind :04d}.png")
                    plt.imsave(vpath, image_abs_diffs[:, :, count-1], cmap="viridis",
                            vmin=0, vmax=np.percentile(image_abs_diffs[:, :, count-1], 99.9))
            else:
                #fpath = op.join(args['holo_diff_dir'], f"{ind :04d}.png")
                # since this is an intermediate step to trailing video, no heatmap
                #sio.imsave(fpath, image_abs_diffs[:, :, count].astype(np.uint8),
                #            check_contrast=False)
                # saved in case diff frames are final vis product and we want heatmaps
                vpath = op.join(args['holo_diffvis_dir'], f"{ind :04d}.png")
                plt.imsave(vpath, image_abs_diffs[:, :, count], cmap="viridis",
                        vmin=0, vmax=np.percentile(image_abs_diffs[:, :, count], 99.9))
    else:
        # Edge case with only single image. No diffs possible here
        args['mhi_ind_image'] = np.ones((images.shape[0], images.shape[1])) * args['file_index']
        args['mhi_val_image'] = np.zeros((images.shape[0], images.shape[1]))

    return args

def is_valid_image(path, target_res):
    '''Check that path is a valid image with resolution target_res'''
    try:
        image = np.asarray(PIL.Image.open(path))
    except:
        logging.info("Skipping image {}: Failed to open".format(path))
        return False
    if not image.size > 0:
        logging.info("Skipping image {}: Size = 0".format(path))
        return False
    if image.shape != target_res:
        logging.info(
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

    experiments = set()
    for pattern in patterns:
        dirs = sorted([d for d in glob.glob(pattern) if op.isdir(Path(d))])

        # Filter for valid hologram dir
        filtered_dirs = []
        for d in tqdm(dirs, desc="Verifying holo dirs"):
            if op.isdir(get_exp_subdir('hologram_dir', d, config)):
                filtered_dirs.append(d)
            else:
                logging.warning("Skipping experiment {}: Invalid hologram dir"
                            .format(d))

        for exp in tqdm(filtered_dirs, desc="Verifying sequence lengths"):
            files = get_files(exp, config)

            # Ensure sufficient holograms
            min_hols = config['min_holograms']
            if len(files) < min_hols:
                logging.warning("Skipping experiment {}: Number of valid images {} "
                            "does not meet minimum requirement {}".format(
                            exp, len(files), min_hols))
            else:
                experiments.add(exp)

    return list(experiments)


def validate_data(exp_dir, holo_fpaths, config, n_workers=1, overwrite=False,
                  memory=None):
    """Run suite of algorithms to produce plots, text, and movies for holograms

    Parameters
    ----------
    exp_dir: str
        Path of working directory (usually the experiment folder)
    holo_fpaths: list
        Full filepaths to all hologram images
    config: dictionary
        Expects the following keys under config['validate']
        mp_batch_size: int
            Number of hologram images to run in each multiprocessing batch
        resize_ratio: float
            Resize images by this factor directly after they're loaded
        trail_length: int
            Number of frames to include when generating the trail video
    n_workers: int
        Number of processes to use in multiprocessing. The maximum value you
        should use is likely the number of cores on your machine
    overwrite: bool
        Whether or not to overwrite previous summaries
    memory
    """

    exp_name = Path(exp_dir).name

    logging.info(f'Processing validation for {exp_name}')

    # Read config
    mp_batch_size = config['validate']['mp_batch_size']
    resize_ratio = config['validate']['resize_ratio']
    trail_length = config['validate']['trail_length']
    min_distinct_holograms = config['validate']['min_distinct_holograms']

    # Specify names of necessary directorys
    validation_dir = Path(get_exp_subdir('validate_dir', exp_dir, config))
    holo_dir = Path(get_exp_subdir('hologram_dir', exp_dir, config))
    holo_baseline_dir = Path(get_exp_subdir('baseline_dir', exp_dir, config))
    holo_diff_dir = Path(get_exp_subdir('diff_dir', exp_dir, config))
    holo_diffvis_dir = Path(get_exp_subdir('diffvis_dir', exp_dir, config))
    holo_trail_dir = Path(get_exp_subdir('trail_dir', exp_dir, config))

    # Load first image and get image sizes
    first_image_orig_res, _skip = utils.read_images([holo_fpaths[0]], 1)
    first_image_orig_res = first_image_orig_res.squeeze()

    first_image, _skip = utils.read_images([holo_fpaths[0]], resize_ratio)
    first_image = first_image.squeeze()
    rows, cols = first_image.shape

    num_files = len(holo_fpaths)

    ###################################
    # Save out first image in the stack
    plt.imsave(op.join(validation_dir, f'{exp_name}_first_image.png'),
               first_image, cmap='gray')

    ###################################
    # Save out histogram of first image
    products.make_histogram(first_image, op.join(validation_dir, f'{exp_name}_first_hist.png'))

    ###################################
    # Compute, plot 2D Fourier transform of first image
    log_power_image = products.fourier_transform_image(first_image_orig_res)
    plt.imsave(op.join(validation_dir, f'{exp_name}_k_powerspec_orig.png'),
               log_power_image, cmap='viridis')
    logging.info('Fourier transform computation complete.')

    ###################################
    # Calculate median image of dataset
    mp_batches = []

    # Compile multiprocessing batch information (just filenames and how to resize)
    for i in range(0, num_files, mp_batch_size):
        file_batch = holo_fpaths[i:i + mp_batch_size]
        batch_info = dict(file_batch=file_batch,
                          resize_ratio=resize_ratio)
        mp_batches.append(batch_info)

    # Get the median image for each batch of images
    with multiprocessing.Pool(n_workers) as pool:
        results = list(tqdm(pool.imap_unordered(calc_median_image, mp_batches),
                            total=math.ceil(num_files / mp_batch_size),
                            desc='Calculate median'))

    # Take the median of medians to get a single image for baseline subtraction
    median_dataset_image = np.median(np.array(results), axis=0)
    med_pil_img = PIL.Image.fromarray(median_dataset_image.astype(np.uint8))
    med_pil_img.save(op.join(validation_dir, f'{exp_name}_median_image.tif'),
                     compression='tiff_lzw')

    ###########################################
    # Calculate particle density of first image and save related plots
    density_first_img = products.estimate_density(first_image, median_dataset_image, config,
                                                  op.join(validation_dir, f'{exp_name}_density_first_image_stdevs.png'),
                                                  op.join(validation_dir, f'{exp_name}_density_first_image_viz.gif'))

    ###################################
    # Read hologram images, calculate diffs, identify bad frames/duplicates
    mp_batches = []
    for i in range(0, num_files, mp_batch_size):
        file_batch = holo_fpaths[i:i + mp_batch_size]
        batch_info = dict(file_batch=file_batch,
                          resize_ratio=resize_ratio,
                          zeroing_image=first_image)
        mp_batches.append(batch_info)

    nearmin_baseline_intensity = []
    nearmax_baseline_intensity = []

    with multiprocessing.Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(load_image_batch, mp_batches),
                            total=math.ceil(num_files / mp_batch_size),
                            desc='Load and preproc images'))

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
        logging.warning("Skipping experiment {}: Number of unique images {} "
                        "does not meet minimum requirement {}".format(
                        exp_name, distinct, min_distinct_holograms))
        return

    # Calculate the vmin/vmax for the zeroed hologram images (used to normalize the movie)
    baseline_zeroed_vmin = np.min(nearmin_baseline_intensity)
    baseline_zeroed_vmax = np.max(nearmax_baseline_intensity)

    max_diff = max(max_diff)
    if memory:
        memory.event.put('Max difference')

    logging.info('Read and preprocessed all images.')

    ###################################
    # Compile multiprocessing batches for intensity/diff/MHI calculations
    mp_batches = []
    for batch_i, file_i in enumerate(range(0, num_files, mp_batch_size)):
        file_batch = holo_fpaths[file_i:file_i + mp_batch_size]
        args = {"batch_index":batch_i,
                "file_index":file_i,
                "file_batch":file_batch,
                "resize_ratio":resize_ratio,
                "zeroing_image":median_dataset_image,
                "validation_dir":validation_dir,
                "mp_batch_size":mp_batch_size,
                "num_files":num_files,
                "max_diff":max_diff,
                "holo_baseline_dir":holo_baseline_dir,
                "holo_diff_dir":holo_diff_dir,
                "holo_diffvis_dir":holo_diffvis_dir,
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
            args["prepend_image_fpath"] = holo_fpaths[file_i - 1]
        mp_batches.append(args)

    # Initialize various arrays that will be filled in with MP results
    intensities_accum = np.empty((num_files))
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
                            desc='Calculate intensity, diff, MHI, density'))

    # Calculate/store various image stats for plotting
    for result in results:

        # Compile intensity information
        start_i = result['file_index']
        end_i = np.minimum(start_i + mp_batch_size, num_files)
        intensities_accum[start_i:end_i] = result['intensities']
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
                                                        f'{exp_name}_mhi.png'),
                         savepath_numpy_array=op.join(validation_dir,
                                                      f'{exp_name}_mhi.npy'))

    logging.info('Time indices of maximum change identified and plotted.')
    ###################################
    # Plot time-series of intensity values
    x_vals, y_vals = np.arange(intensities_accum.shape[0]), intensities_accum
    x_label, y_label = 'Time index', 'Mean intensity'
    savefpath_template = op.join(validation_dir,
                                 f'{exp_name}_timestats_intensity')

    # Save normal time series plots
    stddevs = np.row_stack((intensities_accum - intensities_stddev_accum,
                            intensities_accum + intensities_stddev_accum))

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
    utils.save_timeseries_csv(np.column_stack((x_vals, y_vals)),
                              [x_label, y_label],
                              save_fpath=savefpath_template + '.csv')

    logging.info('Intensity time series info plotted and saved as CSVs.')

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

    logging.info('Pixel difference information plotted and saved as CSVs.')

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

    logging.info('Density information plotted and saved as a CSV.')

    ###################################
    # Make animations

    # Generate .mp4 movie of original hologram frames
    movie_orig_fpath = op.join(validation_dir, f'{exp_name}_orig_movie.mp4')
    products.make_movie(movie_orig_fpath, holo_dir, fname_temp="%5d_holo.tif")
    # Generate quicklook gif of original hologram frames
    gif_orig_fpath = op.join(validation_dir, f'{exp_name}_orig_movie.gif')
    products.make_gif(movie_orig_fpath, gif_orig_fpath)

    # Generate .mp4 movie of median subtracted frames
    movie_base_fpath = op.join(validation_dir, f'{exp_name}_base_movie.mp4')
    products.make_movie(movie_base_fpath, holo_baseline_dir)
    # Generate quicklook gif of median subtracted frames
    gif_base_fpath = op.join(validation_dir, f'{exp_name}_base_movie.gif')
    products.make_gif(movie_base_fpath, gif_base_fpath)

    # Generate .mp4 movie of diff frames
    movie_diff_fpath = op.join(validation_dir, f'{exp_name}_diff_movie.mp4')
    products.make_movie(movie_diff_fpath, holo_diffvis_dir)
    # Generate quicklook gif of diff frames
    gif_diff_fpath = op.join(validation_dir, f'{exp_name}_diff_movie.gif')
    products.make_gif(movie_diff_fpath, gif_diff_fpath)

    # Generate trail frames
    #products.make_trail_frames(holo_diff_dir, holo_trail_dir, np.max(max_diff_all), trail_length=trail_length)
    # Generate .mp4 movie of trail frames
    #movie_trail_fpath = op.join(validation_dir, f'{exp_name}_trail_movie.mp4')
    #products.make_movie(movie_trail_fpath, holo_trail_dir)
    # Generate quicklook gif of trail frames
    #gif_trail_fpath = op.join(validation_dir, f'{exp_name}_trail_movie.gif')
    #products.make_gif(movie_trail_fpath, gif_trail_fpath)


    logging.info('Animations complete.')

    # TODO: Decide whether we want to keep diff frames. For now, wipe.
    utils._check_create_delete_dir(holo_diff_dir, True)
    utils._check_create_delete_dir(holo_diffvis_dir, True)
    utils._check_create_delete_dir(holo_trail_dir, True)

    ###################################
    # Plot duplicated frames report
    products.plot_duplicate_frames(op.join(validation_dir,
                                           f'{exp_name}_timestats_duplicate_frames.png'),
                                   num_files,
                                   np.asarray(full_mask_dup_frames),
                                   total_dup_frames)
    logging.info('Duplicate frames plot created.')

    ###################################
    # Generate high level text report
    density_mean = np.mean(density_accum)
    exp_is_dense = density_mean > config['validate']['density_thresh_exp']

    # Get duplicate frame names and indices
    dup_frame_info = [(fname, ind) for ind, (fname, is_dup)
                      in enumerate(zip(holo_fpaths, full_mask_dup_frames))
                      if is_dup]
    products.generate_text_report(op.join(validation_dir,
                                          f'{exp_name}_processing_report.txt'),
                                  bad_files,
                                  dup_frame_info,
                                  num_files,
                                  intensities_accum,
                                  intensities_diff_accum,
                                  exp_is_dense,
                                  density_mean)
    logging.info('Text report saved.')

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
    global_ints = []
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
        int_csv_fn = op.join(validate_dir, exp_name+"_timestats_intensity.csv")
        diff_csv_fn = op.join(validate_dir, exp_name+"_timestats_pixeldiff.csv")

        # read intensity CSV
        curr_i = []     # curr exp intensity values
        viol_i = 0      # number of times intensity threshold was violated
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

        # read difference CSV
        curr_d = []     # curr exp difference values
        viol_d = 0      # number of times diff threshold was violated
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

        # add exp metrics to global lists
        global_ints += curr_i
        global_diffs += curr_d
        for i in range(len(curr_d)):
            # pair up intensity and difference frames for scatterplotting
            # first intensity frame doesn't have a diff frame so offset by one
            global_pairs.append([curr_i[i+1], curr_d[i]])
        threshold_logs.append([exp, viol_i, viol_d])


    ### SAVING THRESHOLD LOGS
    # export threshold log
    log_fn = op.join(out_dir, "threshold_viols.csv")
    with open(log_fn, 'w') as lf:
        log_writer = csv.writer(lf)
        log_writer.writerows(threshold_logs)

    ### SAVING GLOBAL STATS
    # convert stats to numpy arrays
    global_ints = np.array(global_ints)
    global_diffs = np.array(global_diffs)
    global_pairs = np.array(global_pairs)

    # calculate and write global statistics
    global_stats = []
    global_stats.append(["Mean Intensity", np.mean(global_ints)])
    global_stats.append(["Median Intensity", np.median(global_ints)])
    global_stats.append(["Std Intensity", np.std(global_ints)])
    global_stats.append(["Mean Difference", np.mean(global_diffs)])
    global_stats.append(["Median Difference", np.median(global_diffs)])
    global_stats.append(["Std Difference", np.std(global_diffs)])

    stats_fn = op.join(out_dir, "batch_stats.txt")
    with open(stats_fn, 'w') as sf:
        stats_writer = csv.writer(sf)
        stats_writer.writerows(global_stats)

    ### SAVING HISTOGRAMS

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
    utils.plot_histogram(ihist_fn, global_ints, bins=100,
                         x_label="Intensity", y_label="Frames",
                         title="Image Intensity Distribution (n = {})".format(global_ints.shape[0]),
                         vlines=int_vlines)

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

    ### SAVING SCATTERPLOT
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
