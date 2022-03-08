# Finds peaks in ACME samples
#
#
#
# input:    mass spec data as pickle
#           args
#
# output:   List of peaks
#           Plots of peaks
#           Background information
#
# Steffen Mauceri
# Mar 2020
import yaml
import sys
import os
import logging
import timeit
import pickle
import csv
import scipy.signal
import multiprocessing
from functools import partial

import numpy   as np
import pandas  as pd
import os.path as op

from pathlib         import Path
from scipy           import ndimage
from scipy.optimize  import curve_fit
from skimage.feature import peak_local_max

from fsw.ACME.lib.experiments  import convert_file, read_csvs
from fsw.ACME.lib.plotting     import get_axes_ticks_and_labels, plot_heatmap, \
                                       plot_heatmap_with_peaks, plot_peak_vs_time, \
                                       plot_peak_vs_mass, plot_peak_vs_mass_time, \
                                       plot_mugshots 

from fsw.ACME.lib.utils        import make_crop, \
                                       find_nearest_index, write_filtered_csv, write_rawdata_csv, \
                                       write_peaks_csv, write_excel, find_known_traces

from fsw.ACME.lib.background   import write_pickle, read_pickle, write_csv, read_csv, \
                                       write_tic, write_jpeg2000, read_jpeg2000, \
                                       compress_background_PCA, reconstruct_background_PCA, \
                                       compress_background_smartgrid, reconstruct_background_smartgrid, \
                                       remove_peaks, overlay_peaks, total_ion_count

from fsw.ACME.lib.JEWEL_in     import calc_SUE, diversity_descriptor

from utils.manifest import AsdpManifest, load_manifest_metadata

def diff_gauss(sigma, ratio):
    '''calculate difference of gaussian kernel
    kernel is normalized so that the volume = 1

    Parameters
    ----------
    sigma: float
        standard deviation of gaussian kernel
    ratio: float
        ratio of second sigma to first sigma > 1

    Returns
    -------
    g: ndarray
        array with normalized gaussian kernel
    '''

    epsilon = 1E-2

    size = int(4 * (sigma * ratio))
    x, y = np.meshgrid(np.linspace(-(size // 2), size // 2, size), np.linspace(-(size // 1), (size // 1), size))
    d = np.sqrt(x * x + y * y)
    g1 = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))  # calc 1st gaussian filter
    g1 *= 1 / np.sum(g1)

    sigma *= ratio  # remove diff from sigma for 2nd gauss filter
    g2 = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))  # calc 2nd gaussian filter and subtract from 1st
    g2 *= 1 / np.sum(g2 + epsilon)

    g = g1 - g2

    g *= 1 / np.sum(g)  # normalize so that volume = 1

    return g


def filter_by_SNR(peaks, data, threshold, window_x, window_y, center_x):
    '''filters found peaks by thresholding their mean-height to surrounding standard deviation

    Parameters
    ----------
    peaks: ndarray
        peak location [y,x]
    data: ndarray
        matrix where peaks where found
    threshold: float
        threshold for SNR
    window_x: int
        total width of crop window. Must be odd
    window_y, int
        total height of crop window. Must be odd
    center_x: int
        center width that we cropped from window. Must be odd

    Returns
    -------
    peaks: ndarray
    '''
    SNR_peak = []
    for peak in peaks:
        crop_center, crop_l, crop_r = make_crop(peak, data, window_x, window_y, center_x)
        # calc ratio of peak to background. add small epsilon to deal with 0
        std = np.max([np.std(crop_l), np.std(crop_r)])
        peak_background = np.max([np.median(crop_l), np.median(crop_r)])
        SNR_peak.append(np.max(crop_center - peak_background) / (std + 1E-2))
    # convert to numpy array
    SNR_peak = np.asarray(SNR_peak)
    # remove peaks below SNR threshold
    peaks = peaks[SNR_peak > threshold, :]

    return peaks


def get_peak_center(roi_peaks, exp, w_y, c):
    '''adjusts center of peak to fall on max value withing a set window size

    Parameters
    ----------
    roi_peaks: ndarray
        peak mass_idx and time_idx
    exp: ndarray
        data
    w_y: int
        window width in mass [mass_idx]
    c: int
        window width in time [time_idx]

    Returns
    -------
    roi_Peaks: ndarray
        adjusted peak mass, time

    '''

    w_x = w_y + 2
    peak_time_idx_list = []
    peak_mass_idx_list = []
    for peak in roi_peaks:
        crop_center, crop_l, crop_r = make_crop(peak, exp, w_x, w_y, c)

        # get peak time and mass (position of max in center crop)
        indx = np.unravel_index(np.argmax(crop_center, axis=None), crop_center.shape)
        # correct peak time (if necessary)
        correction = indx[1] - (c // 2)
        peak_time_idx = int(correction + peak[1])
        peak_time_idx_list.append(peak_time_idx)

        # correct peak mass (if necessary)
        correction = indx[0] - (w_y // 2)
        peak_mass_idx = int(correction + peak[0])
        peak_mass_idx_list.append(peak_mass_idx)

    roi_peaks[:, 0] = np.asarray(peak_mass_idx_list)
    roi_peaks[:, 1] = np.asarray(peak_time_idx_list)

    return roi_peaks


def row_med_filter(row, denoise_x):
    return ndimage.median_filter(row, size=denoise_x)

def find_peaks(label, exp, time_axis, mass_axis, noplots, savedata, knowntraces, compounds, outdir, cores, config):
    '''finds peaks in 2D from raw ACME data

    Parameters
    ----------
    label: string
        Name of experiment, mainly used for logging
    exp: ndarray
        Raw experiment data
    time_axis: list
        List of minutes for each time axis
    mass_axis: list
        List of amu/z's for each mass axis
    noplots: bool
        Flag to enable/disable debug plot generation
    savedata: bool
        whether to save heatmap data with plots
    knowntraces: bool
        if true: only peaks at known masses are kept
    compounds: str
        name of file that contains known masses
    outdir: string
        Output directory for debug plotting
    cores: int
        Number of cores for multiprocessing
    config: dict
        Configuration dictionary read from YAML

    Returns
    -------
    roi_peaks: list
        found peaks
    background: ndarray
        raw background
    exp_no_backgroound: ndarray
        raw data with background removed
    '''
    axes = get_axes_ticks_and_labels(mass_axis, time_axis)

    window_x = config.get('window_x')
    window_y = config.get('window_y')
    center_x = config.get('center_x')
    denoise_x = config.get('denoise_x')
    masses_dist_max = config.get('masses_dist_max')
    sigma = config.get('sigma')
    sigma_ratio = config.get('sigma_ratio')
    min_filtered_threshold = config.get('min_filtered_threshold')
    min_SNR_conv = config.get('min_SNR_conv')

    #DEBUG
    if not noplots:
        n = 1
        plot_heatmap(exp, mass_axis, axes, 'Raw Data', '_' + str(n).zfill(2), label, outdir)

    # make copy of experiment data
    roi = np.copy(exp)

    ## MEDIAN FILTER BACKGROUND DETECTION

    if cores > 1:
        with multiprocessing.Pool(cores) as pool:
            background = np.array(pool.map(partial(row_med_filter, denoise_x=denoise_x), roi.tolist()))
    else: 
        background = ndimage.median_filter(roi, size=[1, denoise_x])
    
    roi -= background
    # copy var to return later
    exp_no_background = np.copy(roi)
    # zero out negative values
    roi[roi < 0] = 0

    #DEBUG
    if not noplots:
        n += 1
        plot_heatmap(background, mass_axis, axes, 'Background', '_' + str(n).zfill(2), label, outdir)
        n += 1
        plot_heatmap(roi, mass_axis, axes, 'Data with Background subtracted', '_' + str(n).zfill(2), label, outdir)


    ## PEAK DETECTION
    ## Find peaks using wavelet filter with 2D Mexican Hat ("Ricker") wavelet 
    # define filter
    blob_filter = diff_gauss(sigma, sigma_ratio)
    # convolve filter
    # NOTE: this function, by default, fills boundaries with zeros
    roi = scipy.signal.fftconvolve(roi, blob_filter, mode='same')

    #DEBUG
    if not noplots:
        n += 1
        plot_heatmap(roi, mass_axis, axes, 'Convolved with Filter', '_' + str(n).zfill(2), label, outdir)


    ## ZERO OUT WITH THRESHOLD
    roi[roi < min_filtered_threshold] = 0

    #DEBUG
    if not noplots:
        n += 1
        plot_heatmap(roi, mass_axis, axes, 'Convolved with Filter after thresholding', '_' + str(n).zfill(2), label, outdir)


    ## FIND PEAKS WITH NON MAX SUPRESSION

    roi_peaks = peak_local_max(roi, min_distance=7)

    ## Z-SCORE IN CONVOLVED DATA
    before = len(roi_peaks)
    # min_SNR_conv here is a configuration
    roi_peaks = filter_by_SNR(roi_peaks, roi, min_SNR_conv, window_x, window_y, center_x)
    logging.info(f'Removed ' + str(before - len(roi_peaks)) + f' peaks via conv z-score <{str(min_SNR_conv)}')

    #DEBUG
    if not noplots:
        peak_pos_df = {'x': roi_peaks[:, 1], 'y': roi_peaks[:, 0]}
        peak_pos_df = pd.DataFrame(data=peak_pos_df)
        n += 1
        plot_heatmap_with_peaks(exp_no_background, peak_pos_df, mass_axis, axes, 'Pre-filtered Peaks',
                                     '_' + str(n).zfill(2), label, outdir, savedata)

    ## SHIFT PEAK CENTER TO MAX VALUE
    w_y = 5  # max peak shift in mass [mass index] (int: odd)
    c = 5  # max peak shift in time [time index] (int: odd)
    roi_peaks = get_peak_center(roi_peaks, exp_no_background, w_y, c)

    #DEBUG
    if not noplots:
        peak_pos_df = {'x': roi_peaks[:, 1], 'y': roi_peaks[:, 0]}
        peak_pos_df = pd.DataFrame(data=peak_pos_df)
        n += 1
        plot_heatmap_with_peaks(exp_no_background, peak_pos_df, mass_axis, axes,
                                     'All Peaks found', '_' + str(n).zfill(2), label, outdir, savedata)

    ## KNOWN MASSES HANDLING
    if knowntraces:
        before = len(roi_peaks)
        # get list of known compounds
        known_peaks_bool = find_known_traces(mass_axis[roi_peaks[:,0]], compounds, masses_dist_max)
        roi_peaks = roi_peaks[known_peaks_bool]
        logging.info(f'Removed ' + str(before - len(roi_peaks)) + ' unknown peaks')

        #DEBUG
        if not noplots:
            n += 1
            peak_pos_df = {'x': roi_peaks[:, 1], 'y': roi_peaks[:, 0]}
            peak_pos_df = pd.DataFrame(data=peak_pos_df)
            plot_heatmap_with_peaks(exp_no_background, peak_pos_df, mass_axis, axes,
                                         'Known Masses Pre-filtered Peaks', '_' + str(n).zfill(2), label, outdir, savedata)

    return roi_peaks, background, exp_no_background


def zscore(trace, means, sigmas):
    '''Returns trace zscore given means and sigmas'''
    return (trace - means) / sigmas

def gaussian(x, b, amp, mean, std):
    '''gaussian function'''
    return b + (amp * np.exp(-((np.array(x) - mean) ** 2) / (2 * (std ** 2))))

def calc_peak_property(peak, exp, window_x, window_y, center_x, x_sigmas):
    '''Calculates a single peak's properties

    peak: list
        A single peak's coordinates
    exp: ndarray
        Raw data from file
    window_x: int
        Maximum size of window on time axis to consider prior to gaussian fit
    window_y: int
        Size of window on mass axis to consider
    center_x: int
        Default size of window on time axis if gaussian fit fails
    x_sigmas: float
        Number of standard deviations to mark peak start/stop after gaussian fit
    
    Returns
    ------
    dict with peak properties
    '''
    
    d = {}

    ## MAKE WIDE CROP
    # crop the peak wider than we normally would to perform gaussian fit
    crop_wide, _, _ = make_crop(peak, exp, window_x+2, window_y, window_x)
    crop_1d = crop_wide[window_y // 2]

    ## INITIAL GAUSSIAN PARAMS FOR X
    # guess the mean, stddev, and bias of the gaussian for faster opt
    gx_amp = exp[peak[0], peak[1]]
    gx_b = 0
    gx_mean = (window_x) // 2
    gx_std = window_x // 24

    ## GAUSSIAN FIT FOR CENTER WINDOW X SIZING
    # curve fit a gaussian curve onto the peak
    peak_center_x = window_x
    converged = 1
    try:
        # fit curve onto peak. raises RuntimeError if it doesn't converge.
        popt, _ = curve_fit(gaussian, range(window_x), crop_1d, p0=[gx_b, gx_amp, gx_mean, gx_std], maxfev=100)
        fit_bias = popt[0]
        fit_amp = popt[1]
        fit_mean = popt[2]
        fit_std = abs(popt[3])

        if fit_amp < 0:
            # if the fit has a negative amplitude, consider it not converged
            # throw a RuntimeError to get to except block
            raise RuntimeError
        if fit_amp < (exp[peak[0], peak[1]] - fit_bias) * 0.2:
            # if the fit has a amplitude less than a fifth of the peak height,
            # throw a RuntimeError to get to except block
            raise RuntimeError


        # how much the peak center shifted after fit
        mean_offset = abs(fit_mean - gx_mean)

        """
        JAKE: when determining the new window size and not just peak
        start/stop, we need to also consider the mean shift. Because the
        peak center is defined as the max count, the peak may not be
        symmetrical around the given coords. However, we must crop
        symmetrically around the given coords. Therefore, we increase the
        center window so that the entire peak fits, even if it means
        including some background in the peak window on the other side.
        """

        peak_center_x = int(round(x_sigmas * 2 * fit_std)) + \
                        int(round(2 * mean_offset))
        if peak_center_x > window_x:
            peak_center_x = window_x

        # get start offset from center
        start_diff = np.around((fit_mean - gx_mean) - (fit_std * x_sigmas))
        if start_diff < (- window_x // 2) + 1:
            start_diff = (- window_x // 2) + 1

        # get end offset from center 
        end_diff = np.around((fit_mean - gx_mean) + (fit_std * x_sigmas))
        if end_diff > window_x // 2 + 1:
            end_diff = window_x // 2 + 1

        # to calculate loss used during opt
        pred = gaussian(range(window_x), *popt)

    except RuntimeError:
        # couldn't fit curve, set start/end as center limits around peak
        start_diff = - center_x // 2 + 1
        end_diff = center_x // 2 + 1

        # to calculate loss used during opt
        pred = np.zeros(crop_1d.shape)

        # record
        converged = 0


    ## GAUSSIAN FIT EVALUATION
    # 1. scale raw data 0-1 
    # 2. scale pred data with same multiplier
    # 3. calculate MSE

    geval_min = np.min(crop_1d)
    geval_scale = np.max(crop_1d) - geval_min
    if geval_scale == 0:
        d['gauss_loss'] = 0
    else:
        geval_truth = (crop_1d - geval_min) / geval_scale
        geval_pred = (pred - geval_min) / geval_scale
        geval_mse = ((geval_truth - geval_pred) ** 2).mean()
        d['gauss_loss'] = geval_mse
    d['gauss_conv'] = converged


    ## PEAK X START/STOP EVALUATION
    # This metric is used for width and volume calculations
    start_diff = int(start_diff)
    end_diff = int(end_diff)

    start_time_idx = int(peak[1] + start_diff)
    if start_time_idx < 0:
        start_time_idx = 0
    elif start_time_idx >= exp.shape[1]:
        start_time_idx = exp.shape[1] - 1

    end_time_idx = int(peak[1] + end_diff)
    if end_time_idx >= exp.shape[1]:
        end_time_idx = exp.shape[1] - 1
    elif end_time_idx < 0:
        end_time_idx = 0

    d['start_time_idx'] = start_time_idx
    d['end_time_idx'] = end_time_idx


    ## GET PEAK BASE WIDTH 
    peak_base_width = end_time_idx - start_time_idx
    d['peak_base_width'] = peak_base_width

    ## STAND-IN FOR ANY CENTER-Y PROCESSING
    peak_center_y = window_y
    

    ########################################################################
    # THE FOLLOWING PROPERTIES REQUIRE RECROPPING WITH VAR WINDOW SIZE     # 
    ########################################################################

    ## MAKE CROP
    # window size with bg added
    if peak_center_x % 2 != 1:
        peak_center_x += 1
    if peak_center_y % 2 != 1:
        peak_center_y += 1
    # TODO: harcoded bg window size
    bg_side_size = 15
    peak_window_x = peak_center_x + bg_side_size * 2
    crop_center, crop_l, crop_r = make_crop(peak, exp, peak_window_x, peak_center_y, peak_center_x)

    ## BACKGROUND STATISTICS
    # peak_background_abs is the max of the medians of the two side windows
    # peak_background_std is the max of the stddevs of the two side windows
    # peak_background_diff is the diff of the medians of the two side windows
    # peak_background_ratio is the min of the ratios of the medians of the two side windows

    # Background, median for each mass axis
    peak_background_map = np.zeros(crop_center.shape)
    peak_background_side = np.zeros(crop_l.shape)
    for m in range(peak_center_y):
        bgval = np.max([np.median(crop_l[m]), np.median(crop_r[m])])
        peak_background_map[m] = bgval
        peak_background_side[m] = bgval

    # Background, median for the peak's mass axis 
    med_l = np.median(crop_l[peak_center_y // 2])
    med_r = np.median(crop_r[peak_center_y // 2])

    ## Peak background statistics
    # These have been redefined to only use the mass on which the max of the
    # peak sits. They are not used in the rest of the analysis.
    peak_background_abs = np.max([med_l, med_r])
    d['background_abs'] = peak_background_abs
    peak_background_std = np.max([np.std(crop_l), np.std(crop_r)])
    d['background_std'] = peak_background_std

    peak_background_diff = abs(med_r - med_l)
    d['background_diff'] = peak_background_diff
    peak_background_ratio = np.min([med_l / (med_r + 1e-4), med_r / (med_l + 1e-4)])
    d['background_ratio'] = peak_background_ratio

    ## SUBTRACT BACKGROUND FROM RAW DATA
    bgsub_center = crop_center - peak_background_map
    bgsub_l = crop_l - peak_background_side
    bgsub_r = crop_r - peak_background_side

    ## GET ABSOLUTE PEAK HEIGHT
    # note that this is the peak height count with the background subtracted
    peak_height = bgsub_center[peak_center_y//2, peak_center_x//2]
    d['height'] = peak_height

    ## GET PEAK ZSCORE
    # peak_height is already background subtracted so no mean offset
    # sigmas recalculated below using bg subtracted values
    sigmas = np.max([np.std(bgsub_l), np.std(bgsub_r)])
    peak_zscore = zscore(peak_height, 0, sigmas+1e-2)
    d['zscore'] = peak_zscore

    ## GET PEAK VOLUME
    # only calculate peak volume before the start and end times
    peak_volume = np.sum(bgsub_center)
    d['volume'] = peak_volume

    ## FILTER OUT EXP START PEAKS
    # metric to later remove artefacts that come from spikes right at the beginning of experiment
    on_edge = 0
    if (np.sum(crop_l) == 0) & (np.sum(crop_r) > 0):
        on_edge = 1
    d['on_edge'] = on_edge

    d['mass_idx'] = peak[0]
    d['time_idx'] = peak[1]

    return d

def get_peak_properties(label, peaks, exp, cores, outdir, config):
    '''Multiproc wrapper for calculating peak properties

    Parameters
    ----------
    label: string
        Name of experiment, mainly used for logging.
    peaks: list
        List of detected peak coordinates
    exp: ndarray
        Raw data from file
    cores: int
        Number of cores to multiproc
    outdir: string
        Path to output directory, used for gaussian fit debug plots 
    config: dict
        Configuration read in from YAML
    '''
    window_x = config.get('window_x')
    window_y = config.get('window_y')
    center_x = config.get('center_x')
    x_sigmas = config.get('x_sigmas')
    
    logging.info(f'Calculating peak properties from candidates.')
    with multiprocessing.Pool(cores) as pool:
        dicts = list(pool.map(partial(calc_peak_property, exp=exp, window_x=window_x, window_y=window_y, center_x=center_x, x_sigmas=x_sigmas), peaks))

    peak_properties = pd.DataFrame(data=dicts)
    return peak_properties

def down_select(label, peak_properties, min_peak_volume,
                noplots, mass_axis, exp_no_background, time_axis, file_id, outdir,
                savedata):
    '''Downselect found peak by their properties
    
    Parameters
    ----------
    label: string
        Name of experiment, mainly used for logging
    peak_properties: dataframe
        Peak properties with which to filter peaks
    min_peak_volume: int
        Minimum peak volume to allow
    noplots: bool
        Flag to disable debug plots
    mass_axis: list
        List of amu/z's for each mass axis, for debug plotting
    exp_no_background: ndarray
        Background-subtracted experiment data, for debug plotting
    time_axis: list
        List of minutes for each time axis, for debug plotting
    file_id: string
        Name of experiment, mainly used for logging
        TODO: Superceded by label, refactor
    outdir: string
        Output directory, for debug plotting
    savedata: bool
        Flag to enable heatmap data saving, for debug plotting
    
    Returns
    -------
    dataframe:
        downselected peak properties
    '''

    logging.info(f'Down-selecting peak candidates based on their properties.')
    logging.info(f'Starting with {len(peak_properties)} peaks')

    # filter non-converged
    passed_properties = peak_properties['gauss_conv'] == 1
    logging.info(f'{np.count_nonzero(passed_properties)} peaks after gaussian filter.')

    # threshold volume
    passed_properties = np.logical_and(passed_properties, peak_properties['volume'] >= min_peak_volume)
    logging.info(f'{np.count_nonzero(passed_properties)} peaks after volume filter.')

    # threshold migration time
    threshold_5min = find_nearest_index(time_axis, 5)
    passed_properties = np.logical_and(passed_properties, peak_properties['time_idx'] >= threshold_5min)
    logging.info(f'{np.count_nonzero(passed_properties)} peaks after time filter.')

    # edge filter
    passed_properties = np.logical_and(passed_properties, peak_properties['on_edge'] == 0)
    logging.info(f'{np.count_nonzero(passed_properties)} peaks after edge filter.')

    # Split Z Scores
    passed_5to10 = np.logical_and(passed_properties, peak_properties['zscore'] >= 5)
    passed_5to10 = np.logical_and(passed_5to10, peak_properties['zscore'] < 10)
    passed_5to10 = np.logical_and(passed_5to10, peak_properties['gauss_loss'] < 0.02)
    passed_5to10 = np.logical_and(passed_5to10, peak_properties['peak_base_width'] > 8)

    passed_10plus = np.logical_and(passed_properties, peak_properties['zscore'] >= 10)
    passed_10plus = np.logical_and(passed_10plus, peak_properties['peak_base_width'] > 5)
    passed_10plus = np.logical_and(passed_10plus, peak_properties['gauss_loss'] < 0.02)

    passed_properties = np.logical_or(passed_5to10, passed_10plus)
    logging.info(f'{np.count_nonzero(passed_properties)} peaks after z-score, width, loss filter.')
    filtered_properties = np.logical_not(passed_properties)

    passed_properties = peak_properties.loc[passed_properties]
    filtered_properties = peak_properties.loc[filtered_properties]

    if not noplots:
        axes = get_axes_ticks_and_labels(mass_axis, time_axis)
        peak_pos_df = {'x': passed_properties.time_idx, 'y': passed_properties.mass_idx}
        peak_pos_df = pd.DataFrame(data=peak_pos_df)
        n = 9  # plot ID
        plot_heatmap_with_peaks(exp_no_background, peak_pos_df, mass_axis, axes, 'Filtered Peaks',
                                '_' + str(n).zfill(2), file_id, outdir, savedata)
    
    return passed_properties, filtered_properties

def analyse_experiment(kwargs):
    '''Main program to analyze raw ACME data

    Parameters
    ----------
    kwargs:


    sigma: float
        standard deviation for 1st gaussian function difference of gaussian kernal
    sigma_ratio: float
        standard deviation for 2nd gaussian function for difference of gaussian kernal
    file: str
        file name to analyze
    filtered_threshold: float
        threshold to filter values after filter is applied
    min_SNR_conv: float
        threshold to filter peaks below a set SNR
    known_masses: bool
        if true: only peaks at known masses are kept
    masses_dist_max: float
        maximum distance between peak mass and known mass to be identified as known mass
    masses_file: str
        name of file that contains known masses
    priority_bin: int
        downlink priority bin of generated data products. If not defined, will be set to `0`.
    manifest_metadata: str
        Manifest metadata (YAML string) or `None`; takes precedence over file entries
    manifest_metadata_file: str
        Manifest metadata file (YAML)

    Returns
    -------

    '''
    # runtime profiling
    st = timeit.default_timer()
    gst = timeit.default_timer()

    filename = kwargs['file']
    label = kwargs['label']
    basedir = kwargs['basedir']
    cores = int(kwargs['cores'])
    priority_bin = int(kwargs.get('priority_bin', 0))
    metadata = load_manifest_metadata(
        kwargs.get('manifest_metadata_file', None),
        kwargs.get('manifest_metadata', None),
    )

    outdir = kwargs['outdir']

    ## Flags
    noplots = kwargs['noplots']
    noexcel = kwargs['noexcel']
    savedata = kwargs['saveheatmapdata']
    knowntraces = kwargs['knowntraces']
    debug_plot = kwargs.get('debug_plots')
    space_mode = kwargs.get('space_mode')
    if space_mode:
        noexcel = True
        noplots = True

    trace_window = 13

    # make sure trace window width is odd
    if not trace_window % 2 == 1:
        logging.warning(f'Malformed trace_window: {trace_window}')
        return

    file_id = kwargs['label']

    ext = Path(filename).suffix
    if ext == ".raw":
        # ThermoFisher MS .raw handling
        logging.info(f"Converting ThermoFisher raw file: {str(filename)}")
        filename = convert_file(str(filename), basedir, label)
        data = pickle.load(open(Path(filename), 'rb'))
    elif ext == ".pickle":
        # ThermoFisher MS .pickle handling
        logging.info(f"Loading ThermoFisher pickle file: {str(filename)}")
        data = pickle.load(open(Path(filename), 'rb'))
    elif ext == ".csv":
        # BaySpec MS .csv handling
        stem_comps = Path(filename).stem.split('_')
        if stem_comps[-1] != "00000" and stem_comps[0] != 'Spectra':
            logging.error(f"BaySpec CSV file {Path(filename).name} does not match Spectra_*_00000.csv")
            logging.error("Only glob Spectra_*_00000.csv, the rest will be found in the parent directory.")
            return
        logging.info(f"Loading BaySpec CSV file: {str(filename)}")
        data = read_csvs(filename)
    else:
        logging.error(f'Invalid file extension for file {Path(filename).name}')
        logging.error('Files should be either ".raw", ".pickle", or ".csv" format')
        return

    ## Reading data file
    time_axis = data['time_axis']
    mass_axis = data['mass_axis']
    exp = data['matrix']
    exp = exp.T  # transpose

    mean_time_diff = np.mean(np.diff(time_axis))

    if not noexcel:
        # Stores position where to place mass and time in excel sheet
        mass_pos_tr = dict()
        mass_pos_sp = dict()
        time_pos = dict()

        data_traces_counts = pd.DataFrame(time_axis, columns=['Time'])
        data_traces_basesub = pd.DataFrame(time_axis, columns=['Time'])
        data_spectra_counts = pd.DataFrame(mass_axis, columns=['Mass'])

    ## Load Compounds
    compounds = yaml.safe_load(open(kwargs['masses'], 'r'))

    ## load analyser settings
    args = yaml.safe_load(open(kwargs['params'], 'r'))

    min_peak_volume = args.get('min_peak_volume')
    masses_dist_max = args.get('masses_dist_max')

    window_x = args.get('window_x')
    window_y = args.get('window_y')
    center_x = args.get('center_x')

    # abort if experiment shape is too small
    if exp.shape[0] < window_y + 1 or exp.shape[1] < window_x:
        logging.error(f"{label} skipped, data shape {exp.shape} is too small")
        return
    
    logging.info("Finished {} step. Elapsed time = {time:.2f} s".format(
        'setup', time=timeit.default_timer() - st))
    st = timeit.default_timer()

    # find peaks in raw data
    peaks, background, exp_no_background = find_peaks(label, exp, time_axis, mass_axis, noplots, savedata, knowntraces, compounds, outdir, cores, args)
    
    logging.info("Finished {} step. Elapsed time = {time:.2f} s".format(
        'find_peaks', time=timeit.default_timer() - st))
    st = timeit.default_timer()

    # determine peak properties of found peaks
    peak_properties = get_peak_properties(label, peaks, exp, cores, outdir, args)
    
    logging.info("Finished {} step. Elapsed time = {time:.2f} s".format(
        'peak_properties', time=timeit.default_timer() - st))
    st = timeit.default_timer()

    # downselect peaks further based on peak_properties
    peak_properties, filtered_properties = down_select(label, peak_properties, min_peak_volume,
                                  noplots, mass_axis, exp_no_background,
                                  time_axis, file_id, outdir, savedata)
    
    logging.info("Finished {} step. Elapsed time = {time:.2f} s".format(
        'down_select', time=timeit.default_timer() - st))
    st = timeit.default_timer()

    # plot mugshots of peaks
    plot_mugshots(label, peak_properties, exp, time_axis, mass_axis, cores, outdir)

    logging.info("Finished {} step. Elapsed time = {time:.2f} s".format(
        'mugshot', time=timeit.default_timer() - st))
    st = timeit.default_timer()

    # write csv / excel
    if not space_mode:
        write_rawdata_csv(label, exp, time_axis, mass_axis, file_id, outdir, exp_no_background)
        
        logging.info("Finished {} step. Elapsed time = {time:.2f} s".format(
            'rawdata_csv', time=timeit.default_timer() - st))
        st = timeit.default_timer()

    if not noplots:
        # plot spectra
        plot_peak_vs_time(label, peak_properties, debug_plot, exp, mass_axis, time_axis, outdir, center_x, window_x, window_y, trace_window, knowntraces, compounds, cores)

        # plot spectra
        plot_peak_vs_mass(label, peak_properties, debug_plot, exp, mass_axis, time_axis, outdir, center_x, window_x, window_y, trace_window, exp_no_background, cores)

        if debug_plot:
            # additional debug plots
            plot_peak_vs_mass_time(label, peak_properties, exp_no_background, mass_axis, time_axis, outdir, center_x, window_x, window_y)
        
        logging.info("Finished {} step. Elapsed time = {time:.2f} s".format(
            'plotting', time=timeit.default_timer() - st))
        st = timeit.default_timer()
    
    #write peaks to csv
    peak_properties_exp = peak_properties.copy(deep=True)
    filtered_properties_exp = filtered_properties.copy(deep=True)
    data_peaks = write_peaks_csv(peak_properties_exp, mean_time_diff, mass_axis, time_axis, outdir, label, knowntraces, compounds)
    data_filtered = write_filtered_csv(filtered_properties_exp, mean_time_diff, mass_axis, time_axis, outdir, label, knowntraces, compounds)

    # write peaks to excel
    if not noexcel:
        write_excel(label, peak_properties, exp_no_background, exp, mass_axis, time_axis, knowntraces, compounds, file_id, basedir, outdir, data_peaks, mass_pos_tr, mass_pos_sp, time_pos, data_traces_counts, data_traces_basesub, data_spectra_counts)
    
    logging.info("Finished {} step. Elapsed time = {time:.2f} s".format(
        'csvs_excels', time=timeit.default_timer() - st))
    st = timeit.default_timer()

    # write background to bz2
    if not knowntraces:
        savepath = os.path.join(outdir, label + '_UM_peaks.csv')
        peaks = []
        with open(savepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                peaks.append(row)
        grid_summary = compress_background_smartgrid(exp, args, peaks, t_thresh_perc=98.5, m_thresh_perc=98.5)
        background_filepath = op.join(outdir, label+'_background.bz2')
        filesize_kB = write_pickle(grid_summary, background_filepath, True)
        logging.info(f"Saved Background, {filesize_kB:.2f} kB")
    else:
        logging.info(f"Skipped Background, knowntraces")
    
    logging.info("Finished {} step. Elapsed time = {time:.2f} s".format(
        'background', time=timeit.default_timer() - st))
    st = timeit.default_timer()

    # write Total Ion Count to csv
    tic = total_ion_count(exp, 1)
    tic_filepath = op.join(outdir, label+"_tic.csv")
    filesize_kB = write_tic(tic, tic_filepath)
    logging.info(f"Saved Total Ion Count, {filesize_kB:.2f} kB")

    # calculate Science Utility Estimate (SUE) and Diversity Descriptor (DD)
    mass_res = args.get('mass_resolution')
    mass_dd_res = args.get('dd_mass_resolution')
    time_res = args.get('time_resolution')
    SUE_filepath = op.join(outdir, label + '_SUE.csv')
    calc_SUE(label, peak_properties, kwargs['sue_weights'], compounds, mass_axis, mass_res, time_axis, time_res, masses_dist_max, SUE_filepath)
    DD_filepath = op.join(outdir, label + '_DD.csv')
    diversity_descriptor(label, peak_properties, kwargs['dd_weights'], compounds, mass_axis, mass_res, mass_dd_res, time_axis, time_res, masses_dist_max, DD_filepath)

    # write asdp manifest
    manifest = AsdpManifest('acme', priority_bin)
    manifest.add_metadata(**metadata)

    # background
    manifest.add_entry(
        'background_summary',
        'asdp',
        op.join(outdir, label+'_background.bz2'),
    )
    manifest.add_entry(
        'total_ion_count',
        'asdp',
        op.join(outdir, label+'_tic.csv'),
    )

    # peaks
    manifest.add_entry(
        'peak_properties',
        'asdp',
        op.join(outdir, label+'_UM_peaks.csv'),
    )
    manifest.add_entry(
        'peak_mugshots',
        'asdp',
        op.join(outdir, 'Mugshots'),
    )

    # sue/dd
    manifest.add_entry(
        'science_utility',
        'metadata',
        op.join(outdir, label+'_SUE.csv'),
    )

    manifest.add_entry(
        'diversity_descriptor',
        'metadata',
        op.join(outdir, label+'_DD.csv'),
    )

    manifest.write(op.join(outdir, label+'_manifest.json'))
    
    logging.info("Finished {} step. Elapsed time = {time:.2f} s".format(
        'asdp', time=timeit.default_timer() - st))

    # print execution time
    duration = timeit.default_timer() - gst
    
    logging.info(f'{label}: Finished processing file in ' + str(round(duration,1)) + ' seconds')
