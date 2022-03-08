'''
Support functions for generating HELM validation products.
'''
import shutil
import logging
import csv

import numpy             as np
import matplotlib        as mpl
import matplotlib.pyplot as plt

from pathlib                 import Path
from skimage                 import img_as_ubyte
from skimage.transform       import rescale
from PIL                     import Image
from matplotlib.colors       import Normalize
from mpltools.color          import LinearColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.file_manipulation import tiff_read

def _check_create_delete_dir(dirname, overwrite):
    """Helper to check if a Path is a directory, and delete it if specified."""
    if isinstance(dirname, str):
        dirname = Path(dirname)

    # Create necessary file tree (or empty it out if overwrite=True)
    if dirname.is_dir():
        if overwrite:
            # Delete contents
            shutil.rmtree(dirname)
            logging.warning(f'Deleting {dirname} directory (as `overwrite`=True.)')
        else:
            raise RuntimeError(f'Directory ({dirname}) exists and `overwrite`=False')

    dirname.mkdir()


def scale_from_minmax(var):
    """ Scale a variable to fit between 0 and 1"""
    return var - np.min(var) / (np.max(var) - np.min(var))


def plot_mhi_image(save_fpath, hmi_ind_image, title, hmi_val_image=None,
                   cmap='nipy_spectral_r', include_cbar=True,
                   savepath_unlabeled_img=None,
                   savepath_numpy_array=None):
    """Create and save motion history image indicating time index of largest pixel change

    Parameters
    ----------
    save_fpath: str
        Filepath for saving image
    hmi_ind_image: np.ndarray
        Array containing time indicies of largest change per pixel
    title: str
        Title for the plot
    hmi_val_image:
        Optional array containing the value of the change at the time of the
        largest change. If specified, pixels with smaller relative changes will
        be made partially transparent.
    cmap: str
        Matplotlib colormap name
    include_cbar: bool
        Whether or not to add colorbar to plot
    savepath_unlabeled_img: str
        If specified, save a raw image version of the HMI without any labels
    savepath_numpy_array: str
        If specified, save a numpy array of raw hmi data
    """

    fig, ax = plt.subplots()

    # Plot image
    if hmi_val_image is None:
        im = plt.imshow(hmi_ind_image, cmap=cmap)

        # Save the raw image (without labels/colorbar)
        if savepath_unlabeled_img:
            plt.imsave(savepath_unlabeled_img, hmi_ind_image, cmap=cmap)
        if savepath_numpy_array:
            np.save(savepath_numpy_array, hmi_ind_image)
    else:
        # Convert time indicies indices into color
        colors = Normalize()(hmi_ind_image)
        colors = mpl.cm.get_cmap(cmap)(colors)

        # Create transparency layer for relatively small differences
        upper_clip_bound = np.percentile(hmi_val_image, 99)
        alphas = Normalize(0, upper_clip_bound, clip=True)(hmi_val_image)
        alphas = np.clip(alphas, 0.25, 1)
        # Set alpha channel
        colors[..., -1] = alphas

        # Create the combine image with a black background
        combined_image = Image.new('RGBA', size=hmi_val_image.shape[0:2],
                                   color=(0, 0, 0, 255))
        combined_image.alpha_composite(Image.fromarray(img_as_ubyte(colors)))

        # Plot in matplotlib
        im = ax.imshow(combined_image, cmap=cmap, vmin=np.min(hmi_ind_image),
                       vmax=np.max(hmi_ind_image))

        # Save the raw image (without labels/colorbar)
        if savepath_unlabeled_img:
            combined_image.save(savepath_unlabeled_img)
        if savepath_numpy_array:
            np.save(savepath_numpy_array, hmi_ind_image)

    # Remove ticks
    ax.set_axis_off()
    ax.set_title(title)

    # Add colorbar indicating time frame
    if include_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='5%', pad=0.1)
        fig.colorbar(im, cax=cax, orientation='horizontal')
        cax.set_xlabel('Time index')

    plt.tight_layout()
    plt.savefig(save_fpath, dpi=150, bbox_inches='tight')
    plt.close('all')


def plot_labeled_density_estimate(save_fpath, block_std_vals, threshold):
    """Plot stdev vals in a blocked image density estimates

    Parameters
    ----------
    save_fpath: str
        Filepath to save plot to.
    block_std_vals: np.ndarray
        Array containing the block-wise standard deviation values.
    threshold: float
        Threshold for the standard deviation values. If a standard deviation
        meets or exceeds this value, the block is considered "dense."
    """

    perc_dense = np.sum(block_std_vals >= threshold) / np.size(block_std_vals) * 100
    # Create plot and save raw Std. Dev. vals as an image
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(block_std_vals)

    # Add text to label each block
    for ri, row_val in enumerate(block_std_vals):
        for ci, std_val in enumerate(row_val):
            # Write in bold, red (if above thresh), or white (if below)
            if std_val >= threshold:
                ax.text(ci, ri, f'{std_val:0.2f}', ha="center", va="center",
                        color="r", weight="bold")
            else:
                ax.text(ci, ri, f'{std_val:0.2f}', ha="center", va="center",
                        color="w")

    ax.set_title(f'Std. dev. vals at each image block.'
                 f'\nThreshold: {threshold:0.2f}, Density percentage: {perc_dense:0.2f}%')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(save_fpath)
    plt.close('all')


def plot_density_estimate_gif(save_fpath, baseline_sub_arr, block_std_vals_arr):
    """Save a GIF showing the first image and density estimates

    Parameters
    ----------
    save_fpath: str
        Filepath to save GIF to.
    baseline_sub_arr: np.ndarray
        Array representing the baseline subtracted image.
    block_std_vals_arr: np.ndarray
        Array of the same size as baseline_sub_arr with the standard deviation
        values.
    """
    if baseline_sub_arr.shape != block_std_vals_arr.shape:
        logging.warning('Baseline image shape should match std. dev. image shape.')

    # Compute PIL versions of images so they can be easily saved as GIF
    my_cm = mpl.cm.get_cmap('viridis')

    # Generate PIL image from baseline subtracted array
    baseline_sub_arr = (baseline_sub_arr - np.min(baseline_sub_arr)) / (np.max(baseline_sub_arr) - np.min(baseline_sub_arr))
    baseline_sub_arr = my_cm(baseline_sub_arr, bytes=True)[:, :, :3]
    baseline_sub_img = Image.fromarray(baseline_sub_arr)

    # Generate PIL image from standard deviation array
    block_std_vals_arr = (block_std_vals_arr - np.min(block_std_vals_arr)) / (np.max(block_std_vals_arr) - np.min(block_std_vals_arr))
    block_std_vals_arr = my_cm(block_std_vals_arr, bytes=True)[:, :, :3]
    block_std_vals_img = Image.fromarray(block_std_vals_arr)

    # Get PIL images and save as GIF
    baseline_sub_img.save(save_fpath, save_all=True,
                          append_images=[block_std_vals_img], duration=1000, loop=0)


def plot_timeseries(save_fpath, x_vals, y_vals, x_label, y_label, title,
                    y_lims=None, binary=False, show_mean=False, hlines=None,
                    stddev_bands=None):
    """Creates and saves a simple timeseries plot

    Parameters
    ----------
    save_fpath: str
        Full filepath of where to write plot image
    x_vals: np.ndarray
        Timesteps
    y_vals: np.ndarray
        Values at each timestep
    x_label: str
        X axis labels
    y_label: str
        Y axis labels
    title: str
        Title for the plot
    y_lims: 2-iterable
        Optional low and high limits for y-axis
    binary: bool
        Whether or not the values to plot are binary
    show_mean: bool
        Whether or not to include mean in plot
    hlines: list of dict
        Optional list of dicts specifying hlines to draw. Useful for plotting
        certain thresholds or other limits you want to visualize. See the
        Matplotlib API documentation here:
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hlines.html
    stddev_bands: np.ndarray or None
        Array containing low and high values for standard deviation bands. Array
        should be shape [2, len(x_vals)]. Row 0 should have mean + stddev and
        row 1 should have mean - stddev
    """

    fig, ax = plt.subplots()

    # Handle binary plots separately
    if binary:
        y_vals = y_vals.astype(dtype=bool)

        ax.set_yticks([0, 1])
        ax.set_ylim([-0.2, 1.2])
        ax.set_yticklabels(['False', 'True'])

        ax.plot(x_vals, y_vals)
    else:
        ax.plot(x_vals, y_vals)

        # Specify y-axis limits if provided
        if y_lims:
            ax.set_ylim(y_lims)

        # Calculate and plot mean
        if show_mean:
            mean_val = np.mean(y_vals)
            ax.hlines(y=mean_val, xmin=x_vals[0], xmax=x_vals[-1], color='blue',
                      label='Mean', lw=1, ls='--')
        # Plot any horizontal lines provided
        if hlines is not None:
            if isinstance(hlines, dict):
                hlines = [hlines]
            for hline_dict in hlines:
                ax.hlines(**hline_dict)
        # Plot band showing +/- 1 standard deviation if provided
        if stddev_bands is not None:
            ax.fill_between(x_vals, stddev_bands[0], stddev_bands[1],
                            color='blue', alpha=0.2, label='+/- 1 Std. Dev.')

    # Specify remaining labels/properties and save
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    _, labels = ax.get_legend_handles_labels()
    if len(labels):
        ax.legend(prop={'size': 6}, framealpha=0.95)

    plt.tight_layout()

    fig.savefig(save_fpath, dpi=150)
    plt.close('all')


def plot_histogram(save_fpath, vals, bins, x_label, y_label, title,
                    x_lims=None, vlines=None, force_lines=False):
    """Creates and saves a simple histogram plot

    Parameters
    ----------
    save_fpath: str
        Full filepath of where to write plot image
    vals: np.ndarray
        Values to be graphed
    bins: int
        Number of bins in histogram
    x_label: str
        X axis labels
    y_label: str
        Y axis labels
    title: str
        Title for the plot
    x_lims: 2-iterable
        Optional low and high limits for x-axis
    vlines: list of dict
        Optional list of dicts specifying vlines to draw. Useful for plotting
        certain thresholds or other limits you want to visualize. See the
        Matplotlib API documentation here:
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.vlines.html
        NOTE: ymin/ymax will be replaced with the plot's ymin/ymax.
    force_lines: bool
        Whether to force drawing of lines even if their positions would be
        normally out-of-frame. May result in unclear plots due to scaling.
        Defaults to False.
    """

    fig, ax = plt.subplots()

    ax.hist(vals, bins=bins, range=x_lims)

    # Plot any vertical lines provided
    if vlines is not None:
        if isinstance(vlines, dict):
            vlines = [vlines]
        for vline_dict in vlines:
            vline_dict['ymin'] = ax.get_ylim()[0]
            vline_dict['ymax'] = ax.get_ylim()[1]
            if force_lines:
                ax.vlines(**vline_dict)
            elif vline_dict['x'] > ax.get_xlim()[0] and vline_dict['x'] < ax.get_xlim()[1]:
                ax.vlines(**vline_dict)


    # Specify remaining labels/properties and save
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()

    fig.savefig(save_fpath, dpi=150)
    plt.close('all')


def plot_scatter(save_fpath, x_vals, y_vals, x_label, y_label, title,
                    x_lims=None, y_lims=None, hlines=None, vlines=None,
                    force_lines=False):
    """Creates and saves a simple scatter plot

    Parameters
    ----------
    save_fpath: str
        Full filepath of where to write plot image
    x_vals: np.ndarray
        Timesteps
    y_vals: np.ndarray
        Values at each timestep
    x_label: str
        X axis labels
    y_label: str
        Y axis labels
    title: str
        Title for the plot
    x_lims: 2-iterable
        Optional low and high limits for x-axis
    y_lims: 2-iterable
        Optional low and high limits for y-axis
    hlines: list of dict
    vlines: list of dict
        Optional list of dicts specifying hlines/vlines to draw. Useful
        for plotting certain thresholds or other limits you want to
        visualize. See the Matplotlib API documentation here:
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hlines.html
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.vlines.html
        NOTE: ymin/ymax/xmin/xmax will be replaced with the plot's borders.
    force_lines: bool
        Whether to force drawing of lines even if their positions would be
        normally out-of-frame. May result in unclear plots due to scaling.
        Defaults to False.
    """

    fig, ax = plt.subplots()

    ax.scatter(x_vals, y_vals, marker=',', s=1)

    # Specify y-axis limits if provided
    if y_lims:
        ax.set_ylim(y_lims)
    if x_lims:
        ax.set_xlim(x_lims)

    # Plot any horizontal lines provided
    if hlines is not None:
        if isinstance(hlines, dict):
            hlines = [hlines]
        for hline_dict in hlines:
            hline_dict['xmin'] = ax.get_xlim()[0]
            hline_dict['xmax'] = ax.get_xlim()[1]
            if force_lines:
                ax.hlines(**hline_dict)
            elif hline_dict['y'] > ax.get_ylim()[0] and hline_dict['y'] < ax.get_ylim()[1]:
                ax.hlines(**hline_dict)

    # Plot any vertical lines provided
    if vlines is not None:
        if isinstance(vlines, dict):
            vlines = [vlines]
        for vline_dict in vlines:
            vline_dict['ymin'] = ax.get_ylim()[0]
            vline_dict['ymax'] = ax.get_ylim()[1]
            if force_lines:
                ax.vlines(**vline_dict)
            elif vline_dict['x'] > ax.get_xlim()[0] and vline_dict['x'] < ax.get_xlim()[1]:
                ax.vlines(**vline_dict)

    # Specify remaining labels/properties and save
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    _, labels = ax.get_legend_handles_labels()
    if len(labels):
        ax.legend(prop={'size': 6}, framealpha=0.95)

    plt.tight_layout()

    fig.savefig(save_fpath, dpi=150)
    plt.close('all')


def save_timeseries_csv(column_vals, column_names, save_fpath):
    """Save a series of data points to CSV file"""

    if len(column_names) != column_vals.shape[1]:
        raise ValueError('Number of column names must match # cols in `column_vals`')

    with open(save_fpath, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write column header
        csv_writer.writerow(column_names)

        # Write data
        for row in column_vals:
            csv_writer.writerow([f"{int(v)}" if v.is_integer() else f"{round(v,3):.3f}" for v in row])


def read_images(files):
    """Read in hologram images. Channels are averaged.

    Parameters
    ----------
    files: list
        List of filepath to images that should be read in. All images should
        have the same size.

    Returns
    -------
    images: np.ndarray
        3D array with loaded images stacked along final axis
    bad_files:
        List of indices and filenames that didn't load properly
    """
    if not isinstance(files, list):
        files = [files]

    # Determine resized shape
    rows, cols = tiff_read(files[0], flatten=True).shape

    images = np.empty((rows, cols, len(files)), dtype=float)

    bad_files = []
    for i, fpath in enumerate(files):
        temp_image = tiff_read(fpath, flatten=True)
        if temp_image is None:
            # Failed to read image
            images[:, :, i] = np.zeros((rows, cols))
            bad_files.append((i, fpath))
            logging.error(f"validate failed to read: {fpath}")
        else:
            images[:, :, i] = temp_image

    return images, bad_files


def weighted_mean_dicts(unweighted_val_dict, weight_dict):
    """Compute the weighted mean given dict of boolean values and their weights

    Useful for calculating a weighted sum on interval [0, 1] (e.g., an
    data quality estimate).

    Parameters
    ----------
    unweighted_val_dict: dict
        Dictionary of boolean values with keys matching `weight_dict`
    weight_dict: dict
        Numerical weights for each key

    Returns
    -------
    weighted_normed_value: float
        Value on interval [0, 1] representing the weighted sum of the input
        values. The normalization is applied by dividing by total weight in
        `weight_dict`.
    """
    keys_to_include = list(unweighted_val_dict.keys())

    # Check that weights all exist and health checks have valid values
    for key in keys_to_include:
        if key not in weight_dict.keys():
            logging.warning(f'Key: {key} not found in configuration\'s weight '
                            'dictionary when calculating health metric. Removing.')
            keys_to_include.remove(key)
        if unweighted_val_dict[key] is None:
            logging.warning(f'Key: {key} had value `None` in dictionary when '
                            'calculating health metric. Removing.')
            keys_to_include.remove(key)

    # Compute health score, total weight, and return normalized value between [0, 1]
    weighted_val = np.sum([unweighted_val_dict[key] * weight_dict[key]
                          for key in keys_to_include])
    total_weight = np.sum([weight_dict[key] for key in keys_to_include])

    return weighted_val / total_weight


def handle_duplicate_frames(images=None):
    """Get list of frames without duplicates and report duplicate frame names

    Parameters
    ----------
    images: np.ndarray
        Array of images with images stacked along last axis

    Returns
    -------
    images: np.ndarray
        Image array with duplicates removed
    mask_dup_frames: list
        Boolean list indicated which frames were duplicates
    num_dup_frames: int
        Total number of frames in `images` that were duplicates
    """

    # Find all frames where image was identical to previous image
    mask_dup_frames = np.array([False] + [np.all(images[:, :, i] == images[:, :, i - 1])
                                          for i in range(1, images.shape[2])])

    # Get total number of duplicate frames
    num_dup_frames = np.sum(mask_dup_frames)

    return images[:, :, ~mask_dup_frames], mask_dup_frames, num_dup_frames


def get_aug_rainbow():
    """Rainbow colormap augmented with black"""

    colors = ((1, 1, 1),  # w
              (1, 0, 0),  # r
              (1, .5, 0), # o
              (1, 1, 0),  # y
              (0, 1, 0),  # g
              (0, 0, 1),  # b
              (1, 0, 1))  # v

    cmap = LinearColormap('aug_rainbow', colors)
    return cmap