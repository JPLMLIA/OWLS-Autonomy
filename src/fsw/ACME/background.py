# Functions for background compression of ACME
# these are called from analyzer.py
import os
import time
import pickle
import bz2
import glob
import math
import logging
import glymur

from pathlib               import Path
from sklearn.decomposition import PCA
from PIL                   import Image

import os.path as op
import numpy   as np
import matplotlib.pyplot as plt

from fsw.ACME.utils import make_crop

def write_pickle(package, filepath, compress=False):
    """ Writes the summarized background as a compressed binary file

    Parameters
    ----------
    package:
        Python object to be written
    filepath:
        Filepath to output file
    compress:
        Whether to use bz2 compression. Defaults to False.
    """
    if compress:
        with bz2.BZ2File(filepath, 'w') as f:
            pickle.dump(package, f)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(package, f)
    
    filesize_kB = op.getsize(filepath) / 1024

    return filesize_kB

def read_pickle(filepath, compressed=False):
    """ Reads the summarized background from a compressed binary file

    Parameters
    ----------
    filepath:
        Filepath to compressed file
    compressed:
        Whether to use bz2 compression. Defaults to False
    """
    if compressed:
        with bz2.BZ2File(filepath, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    
    return data

def write_csv(package, filepath):
    """ Writes the summarized background as a CSV file.

    Parameters
    ----------
    package:
        numpy array to be written
    filepath:
        Filepath to output file
    """

    package.tofile(filepath, sep=',', format='%10.5f')
    
    filesize_kB = op.getsize(filepath) / 1024

    return filesize_kB

def write_tic(package, filepath):
    """ Writes the TIC as a CSV file.

    Parameters
    ----------
    package:
        numpy array to be written
    filepath:
        Filepath to output file
    """

    package.tofile(filepath, sep=',', format='%i')
    
    filesize_kB = op.getsize(filepath) / 1024

    return filesize_kB

def read_csv(filepath):
    """ Reads the summarized background from CSV file.

    Parameters
    ----------
    filepath:
        Filepath to csv file
    """

    data = np.fromfile(filepath, sep=',')
    
    return data

def write_jpeg2000(package, c_ratio=1):
    """ Writes an array as a JPEG2k compressed image.

    Parameters
    ----------
    package: array
        Numpy array to be exported.
    c_ratio: int
        Compression ratio to be passed to glymur's Jp2k package.
        1 is no compression.
    
    Returns
    -------
    array:
        The compressed image as a numpy array. Equivalent to reading the
        image back in.
    """

    # scale to 0~255 space
    maxval = int(round(np.max(package)))
    scaled = (package / maxval) * 255.0
    scaled = np.round(scaled).astype(np.uint8)

    filepath = f"jpeg2k_background_{maxval}.jp2" 
    jp2 = glymur.Jp2k(filepath, data=scaled, cratios=[c_ratio])
    
    filesize_kB = op.getsize(filepath) / 1024
    logging.info(f"Saved Background, {filesize_kB:.2f} kB")

    return filesize_kB


def read_jpeg2000(filepath):
    """ Reads a JPEG2k compressed image as an array. Only reads the first
        (0 index) layer, if multiple layers exist.

    Parameters
    ----------
    filepath: str
        Filepath to the image. Will be passed to glob.glob()
        Filename must have max value used for scaling at the end, separated by
        an underscore. Just use write_jpeg2000().
    
    Returns
    -------
    array:
        Read and scaled array.
    """

    filepath = glob.glob(filepath)[0]

    data = glymur.Jp2k(filepath)
    data = data[:]

    filename = Path(filepath).stem
    maxval = int(filename.split('_')[-1])

    scaled = (data / 255.0) * maxval
    
    return scaled

def filter_consec_idx(idx):
    """ Filters out indices that are inside consecutive sequences, only leaving
    the outer two indices. Useful when only the range described by those indices
    are needed (for which consecutive values are redundant).

    e.g. [1, 4, 5, 6, 7, 9, 10] -> [1, 4, 7, 9, 10]

    Parameters
    ----------
    idx: array
        Input 1D array of indices. Should already be sorted.
    
    Returns
    -------
    array
        Output 1D array with consecutive values removed.
    """
    diff = np.hstack(([0], np.diff(idx)))
    diff_r = np.hstack(([0], np.abs(np.diff(idx[::-1]))))[::-1]

    filtered = []
    for i in range(len(idx)):
        if diff[i] == 1 and diff_r[i] == 1:
            pass
        else:
            filtered.append(idx[i])
    
    return np.array(filtered).astype(np.int)

def get_background_error(orig, recon):
    """ Calculate an error metric between the original and reconstructed
    experiment backgrounds.

    Parameters
    ----------
    orig: array
        Experiment background with peaks removed
    recon: array
        Reconstructed experiment background 

    Returns
    -------
    float
        Calculated MSE
    """

    # Calculated MSE
    return np.mean((orig - recon) ** 2) 

def get_n_regions(grid_enc):
    """ Get the number of regions encoded by compress_background_smartgrid().

    Parameters
    ----------
    grid_enc: dict
        Output from compress_background_smartgrid()
    
    Returns
    -------
    int
        Number of regions encoded
    """
    regions = 0
    for row in grid_enc.keys():
        if row == "shape":
            continue
        
        regions += len(grid_enc[row])
    
    return regions

def get_filesize_est(n_regions):
    """ Get a filesize estimate given the number of regions in grid encoding.

    Parameters
    ----------
    n_regions: int
        number of regions encoded by grid encoding
    
    Returns
    -------
    float
        Estimated filesize
    """

    return 0.00636654 * n_regions + 3.392864597


def compress_background_PCA(exp, config, n_comp=5):
    """ Compresses the background of the entire experiment using PCA

    Parameters
    ----------
    exp: array
        Experiment data with no zero padding
    config: dict
       Dictionary read in from YAML configuration
    n_comp: int
        Number of principal components to keep
    
    Returns
    -------
    scores: array
        Scores from PCA
    pcs: array
        Principal components from PCA
    mean: float
        Mean from PCA
    """

    exp = exp.copy()

    # Perform PCA
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(exp.T).astype(np.float32)
    pcs = pca.components_.astype(np.float32)
    mean = pca.mean_.astype(np.float32)

    logging.info(f"PCA Background Exp Var: {np.sum(pca.explained_variance_ratio_):.4f}")

    return scores, pcs, mean


def reconstruct_background_PCA(scores, pcs, mean, orig_exp=None, eval=False):
    """ Reconstructs the background of the experiment using PCA

    Parameters
    ----------
    scores: array
        Scores from PCA
    pcs: array
        Principal components from PCA
    mean: float
        Mean from PCA
    orig_exp: array
        Optional array of original experiment for error calc. Defaults to None.
    eval: bool
        Whether to calculate error for reconstructed background. Defaults to False.

    Returns
    -------
    recon: array
        Reconstructed background
    """

    # reconstruct from components
    recon = (np.dot(scores, pcs) + mean).T

    if eval:
        if orig_exp is None:
            logging.warning("No original exp provided, skipping eval")
            return
        
        # calculate mean error
        mean_error = np.mean(np.abs(recon - orig_exp))
        logging.info(f"PCA Background Recon ME: {mean_error:.2f}")
    
    return recon


def compress_background_smartgrid(exp, config, peaks, thresh_min=40, t_thresh_perc=98.5, m_thresh_perc=98.5):
    """ Detect grid edges and characterize the distribution within regions.

    Parameters
    ----------
    exp: array
        Experiment data with no zero padding.
    config: dict
        Dictionary read in from YAML configuration.
    peaks: dict
        Peaks read from peaks CSV file.
    thresh_min: float
        Minimum threshold for detecting edges on either axis. The distribution
        threshold clips to this lower bound. Used to prevent the algorithm from
        detecting edges where there are seemingly none. Defaults to 40.
    t_thresh_perc: float
        Percentile to use for time edge detection. Defaults to 98.5.
    m_thresh_perc: float
        Percentile to use for time edge detection. Defaults to 98.5.

    Returns
    -------
    grid_encoding: dict
        Grid encoding of the background.
    """
    exp = exp.copy()
    window_y = config['window_y']

    ### MASS AXIS EDGES
    # Max across time axis to find global noise patterns
    max_m = np.max(exp, axis=1)

    # 1st derivative to find edges
    diff_m = np.diff(max_m)

    # Calculate absolute threshold from percentile of diff distribution
    m_thresh = np.percentile(diff_m, m_thresh_perc)
    if m_thresh < thresh_min:
        m_thresh = thresh_min
    
    # Use threshold to find edges
    edge_m = np.nonzero(np.abs(diff_m) > m_thresh)[0]

    # Add edges defined by peak windows
    for row in peaks:
        peak_mass_idx = int(row['Mass (idx)'])
        if not np.any((peak_mass_idx+window_y >= edge_m) & (peak_mass_idx < edge_m)):
            edge_m = np.hstack((edge_m, [peak_mass_idx + window_y]))
        if not np.any((peak_mass_idx-window_y <= edge_m) & (peak_mass_idx > edge_m)):
            edge_m = np.hstack((edge_m, [peak_mass_idx - window_y]))

    # Removed edges out of bounds of the experiment 
    edge_m = edge_m[edge_m > 0]
    edge_m = edge_m[edge_m < exp.shape[0]]
    
    # Remove duplicate edges and re-sort
    edge_set = set(edge_m)
    edge_m = np.array(sorted(list(edge_set)))

    # Add limit edges

    if edge_m.shape[0] == 0:
        edge_m = np.array([0])

    if edge_m[0] != 0:
        edge_m = np.hstack(([0], edge_m))
    if edge_m[-1] != exp.shape[0]:
        edge_m = np.hstack((edge_m, [exp.shape[0]]))

    # Filter out consecutive edges
    edge_m = filter_consec_idx(edge_m)

    ### ENCODE PER ROW
    grid_encoding = {}
    grid_encoding['shape'] = exp.shape

    for i in range(1, len(edge_m)):
        # Crop out row region
        row = exp[edge_m[i-1]:edge_m[i]]
        grid_encoding[(edge_m[i-1], edge_m[i])] = []

        # Median across mass axis to find local noise patterns
        med_t = np.median(row, axis=0)

        # 1st derivative to find edges
        diff_t = np.abs(np.diff(med_t))

        # Calculated absolute threshold from percentile of diff distribution
        t_thresh = np.percentile(diff_t, t_thresh_perc)
        if t_thresh < thresh_min:
            t_thresh = thresh_min

        # Use threshold to find edges
        edge_t = np.nonzero(diff_t > t_thresh)[0]

        # Add limit edges
        if edge_t.shape[0] == 0:
            edge_t = np.array([0])

        if edge_t[0] != 0:
            edge_t = np.hstack(([0], edge_t))
        if edge_t[-1] != exp.shape[1]:
            edge_t = np.hstack((edge_t, [exp.shape[1]]))

        # Filter out consecutive edges
        edge_t = filter_consec_idx(edge_t)

        ### ENCODE PER REGION
        for j in range(1, len(edge_t)):
            # Crop out row and col region
            region = exp[edge_m[i-1]:edge_m[i], edge_t[j-1]:edge_t[j]]

            region_mean = np.mean(region).astype(np.float16)

            # Take 99th percentile to remove high outlier for better stddev calc
            clip = np.percentile(region, 95)
            #region = np.clip(region, a_min=None, a_max=clip)
            region = region[region <= clip]

            # Record distribution stats
            region_std = np.std(region).astype(np.float16)
            
            # Apppend to encoding dictionary
            grid_encoding[(edge_m[i-1], edge_m[i])].append([edge_t[j-1], edge_t[j], region_mean, region_std])

    return grid_encoding


def reconstruct_background_smartgrid(grid_array, orig_exp=None, eval=False):
    """ Reconstructs the background of the experiment using grids

    Parameters
    ----------
    grid_array: dict
        Encoded background
    orig_exp: array
        Original experiment array for evaluation. Defaults to None.
    eval: bool
        Whether to calculate error for background reconstruction. Defaults to False.

    Returns
    -------
    recon: array
        Reconstructed background.
    """

    grid_array = grid_array.copy()
    shape = grid_array['shape']
    grid_array.pop('shape')

    recon = np.zeros(shape)
    for row in grid_array.keys():
        for reg in grid_array[row]:
            m_min = int(row[0])
            m_max = int(row[1])
            t_min = int(reg[0])
            t_max = int(reg[1])

            region_shape = (m_max - m_min, t_max - t_min)
            region_recon = np.random.normal(loc=reg[2], scale=reg[3], size=region_shape)
            region_recon = np.clip(region_recon, a_min=0, a_max=None)
            recon[m_min:m_max, t_min:t_max] = region_recon

    if eval:

        if orig_exp is None:
            logging.warning("No original exp provided, skipping eval")
            return

        # calculate mean error
        mean_error = np.mean(np.abs(recon - orig_exp))
        logging.info(f"Grid Background Recon ME: {mean_error:.2f}")

    return recon


def reconstruct_stats_smartgrid(grid_array):
    """ Reconstructs the statistics of the experiment using grids.
    This is does not sample from the compressed distribution. It returns
    an array of means and an array of stddevs.

    Parameters
    ----------
    grid_array: dict
        Encoded background

    Returns
    -------
    tuple: (means:array, stds:array)
        Experiment-sized arrays of the mean and std of each background pixel

    """

    grid_array = grid_array.copy()
    shape = grid_array['shape']
    grid_array.pop('shape')

    means = np.zeros(shape)
    stds = np.zeros(shape)
    
    for row in grid_array.keys():
        for reg in grid_array[row]:
            m_min = int(row[0])
            m_max = int(row[1])
            t_min = int(reg[0])
            t_max = int(reg[1])

            region_shape = (m_max - m_min, t_max - t_min)

            means[m_min:m_max, t_min:t_max] = np.ones(region_shape) * reg[2]
            stds[m_min:m_max, t_min:t_max] = np.ones(region_shape) * reg[3]

    return means, stds

def remove_peaks(orig, peak_properties, config):
    """ Remove peaks from a given experiment. For each peak in the provided peak
        CSV, the window around the peak is replaced by sampling the gaussian
        distribution defined by background_abs and background_std of that peak.
    
    Parameters
    ----------
    orig: array
        The original experiment to remove peaks from.
    peak_properties: list of dicts
        Result of csv.DictReader() on the peak CSV file.
    config: dict
        Configuration read from YAML
    
    Returns
    -------
    array
        The experiment with identified peaks removed.
    """

    exp = orig.copy()

    ### PEAK REMOVE
    window_x = config['window_x']
    window_y = config['window_y']
    
    for row in peak_properties:
        peak_mass = int(row['Mass (idx)'])
        peak_time = int(row['Peak Central Time (idx)'])
        background_med = float(row['background_abs'])
        background_std = float(row['background_std'])

        upper_mass = int(peak_mass + (window_y // 2)) + 1
        upper_mass = min(exp.shape[0], upper_mass)

        lower_mass = int(peak_mass - (window_y // 2))
        lower_mass = max(0, lower_mass)

        upper_time = int(peak_time + (window_x // 2)) + 1
        upper_time = min(exp.shape[1], upper_time)

        lower_time = int(peak_time - (window_x // 2))
        lower_time = max(0, lower_time)

        window_shape = (upper_mass-lower_mass, upper_time-lower_time)

        exp[lower_mass:upper_mass, lower_time:upper_time] = \
            np.random.normal(loc=background_med, scale=background_std, size=window_shape)
    
    return exp
    

def overlay_peaks(background, peak_properties, mugshot_dir):
    """ Overlay peak mugshots on a reconstructed background by finding the
        associated mugshot for each identified peak and overlaying it.

    Parameters
    ----------
    background: array
        Reconstructed background onto which peaks will be overlaid.
    peak_properties: list of dicts
        Result of csv.DictReader() on the peak CSV file.
    mugshot_dir: str
        Path to the mugshot directory
    
    Returns
    -------
    array:
        Reconstructed background with overlaid peaks.

    """

    exp = background.copy()

    ### PEAK OVERLAY
    # values need to agree with mug shot geometry specified in plotting.py/plot_mugshots()
    mug_x = 121
    mug_y = 13

    total_filesize = 0
    for row in peak_properties:
        # Relevant CSV columns
        peak_mass = round(float(row['Mass (amu)']), 2)
        peak_time = round(float(row['Peak Central Time (Min)']), 2)

        peak_mass_idx = int(row['Mass (idx)'])
        peak_time_idx = int(row['Peak Central Time (idx)'])

        # Corresponding mugshot image
        mugshot_filepath = glob.glob(op.join(mugshot_dir, f"Time_Mass_Max_{peak_time:.2f}_{peak_mass:.2f}_*.tif"))[0]
        mugshot_filename = Path(mugshot_filepath).stem

        # Read and scale corresponding mugshot image
        total_filesize += op.getsize(mugshot_filepath) / 1024
        mugshot = Image.open(mugshot_filepath)
        mugshot = np.array(mugshot)
        # Scale 255 to max count
        mugshot_max = int(mugshot_filename.split('_')[-1])
        mugshot = np.around(mugshot * (mugshot_max / 255))

        # Get window indices, clip to shape of experiment
        upper_mugshot_mass = int(mug_y)
        upper_exp_mass = int(peak_mass_idx + mug_y // 2 + 1)
        if upper_exp_mass > exp.shape[0]:
            upper_mugshot_mass = upper_mugshot_mass - (upper_exp_mass - exp.shape[0])
            upper_exp_mass = exp.shape[0]
        
        lower_mugshot_mass = 0
        lower_exp_mass = int(peak_mass_idx - (mug_y // 2))
        if lower_exp_mass < 0:
            lower_mugshot_mass = lower_mugshot_mass - (lower_exp_mass)
            lower_exp_mass = 0

        upper_mugshot_time = int(mug_x)
        upper_exp_time = int(peak_time_idx + mug_x // 2 + 1)
        if upper_exp_time > exp.shape[1]:
            upper_mugshot_time = upper_mugshot_time - (upper_exp_time - exp.shape[1])
            upper_exp_time = exp.shape[1]

        lower_mugshot_time = 0
        lower_exp_time = int(peak_time_idx - mug_x // 2)
        if lower_exp_time < 0:
            lower_mugshot_time = lower_mugshot_time - (lower_exp_time)
            lower_exp_time = 0

        exp[lower_exp_mass:upper_exp_mass, lower_exp_time:upper_exp_time] = \
            mugshot[lower_mugshot_mass:upper_mugshot_mass, lower_mugshot_time:upper_mugshot_time]
    
    return exp, total_filesize

def total_ion_count(exp, n=4):
    """ Generates the total ion count of an experiment.

    Parameters
    ----------
    exp: array
        Experiment data.
    n: int
        Sampling, passed to numpy slicing's stop index.
        Returns every nth value in TIC.
    
    Returns
    -------
    array:
        array of the (possibly subsampled) TIC.
    """

    tic = np.sum(exp, axis=0)
    tic = 255 * tic / np.max(tic) #discretize to 8bit
    tic = tic[::n]

    #TODO return max TIC
    #TODO return TIC in the mass dimension
    return tic.astype(np.uint8) #convert to 8bit