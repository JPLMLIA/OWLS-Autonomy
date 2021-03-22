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
import time
import pickle
import csv
import scipy.signal

import numpy   as np
import pandas  as pd
import os.path as op
import matplotlib.pyplot as plt

from pathlib         import Path
from scipy           import ndimage
from PIL             import Image
from scipy.optimize  import curve_fit
from skimage.feature import peak_local_max

from acme_cems.lib.experiments  import convert_file
from acme_cems.lib              import background
from acme_cems.lib.plotting     import get_axes_ticks_and_labels, plot_heatmap, \
                                       plot_heatmap_with_peaks, plot_peak_vs_time, \
                                       plot_peak_vs_mass, plot_peak_vs_mass_time, \
                                       plot_mugshots 

from acme_cems.lib.utils        import make_crop, \
                                       find_nearest_index, write_rawdata_csv, \
                                       write_peaks_csv, write_excel, find_known_traces

from acme_cems.lib.background   import write_pickle, read_pickle, write_csv, read_csv, \
                                       write_tic, write_jpeg2000, read_jpeg2000, \
                                       compress_background_PCA, reconstruct_background_PCA, \
                                       compress_background_smartgrid, reconstruct_background_smartgrid, \
                                       remove_peaks, overlay_peaks, total_ion_count

from acme_cems.lib.JEWEL_in     import calc_SUE, diversity_descriptor

from utils.manifest import write_manifest

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

    size = int(3 * (sigma * ratio))
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


def find_peaks(label, exp, window_x, window_y, time_axis, mass_axis, noplots, file_id, outdir, sigma, sigma_ratio,
                   min_filtered_threshold, savedata, min_SNR_conv, center_x, denoise_x, masses_dist_max, compounds, knowntraces):
    '''finds peaks in 2D from raw ACME data

    Parameters
    ----------
    label: string
        Name of experiment, mainly used for logging
    exp: ndarray
        Raw experiment data
    window_x: int
        Maximum size of window on time axis to consider prior to gaussian fit
    window_y: int
        Size of window on mass axis to consider
    time_axis: list
        List of minutes for each time axis
    mass_axis: list
        List of amu/z's for each mass axis
    noplots: bool
        Flag to enable/disable debug plot generation
    file_id: string
        Name of experiment
        TODO: superceded by label, refactor
    outdir: string
        Output directory for debug plotting
    sigma: float
        standard deviation for 1st gaussian function difference of gaussian kernal
    sigma_ratio: float
        standard deviation for 2nd gaussian function for difference of gaussian kernal
    min_filtered_threshold: float
        threshold to filter values after filter is applied
    savedata: bool
        whether to save heatmap data with plots
    min_SNR_conv: float
        threshold to filter peaks below a set SNR
    center_x: int
        Default size of window on time axis if gaussian fit fails
    denoise_x: int
        Window size for median denoising
    masses_dist_max: float
        maximum distance between peak mass and known mass to be identified as known mass
    compounds: str
        name of file that contains known masses
    knowntraces: bool
        if true: only peaks at known masses are kept

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

    #DEBUG
    if not noplots:
        n = 1
        plot_heatmap(exp, mass_axis, axes, 'Raw Data', '_' + str(n).zfill(2), file_id, outdir)

    # make copy of experiment data
    roi = np.copy(exp)

    ## MEDIAN FILTER BACKGROUND DETECTION
    logging.info(f'{label}: Removing background with median filter.')
    background = ndimage.median_filter(roi, size=[1, denoise_x])
    roi -= background
    # copy var to return later
    exp_no_background = np.copy(roi)
    # zero out negative values
    roi[roi < 0] = 0

    #DEBUG
    if not noplots:
        n += 1
        plot_heatmap(background, mass_axis, axes, 'Background', '_' + str(n).zfill(2), file_id, outdir)
        n += 1
        plot_heatmap(roi, mass_axis, axes, 'Data with Background subtracted', '_' + str(n).zfill(2), file_id, outdir)


    ## GAUSSIAN FILTER
    logging.info(f'{label}: Applying difference of Gaussian filter.')
    # define filter
    blob_filter = diff_gauss(sigma, sigma_ratio)
    # convolve filter
    # NOTE: this function, by default, fills boundaries with zeros
    roi = scipy.signal.convolve2d(roi, blob_filter, mode='same')

    #DEBUG
    if not noplots:
        n += 1
        plot_heatmap(roi, mass_axis, axes, 'Convolved with Filter', '_' + str(n).zfill(2), file_id, outdir)


    ## ZERO OUT WITH THRESHOLD
    logging.info(f'{label}: Setting values < {str(min_filtered_threshold)} to zero.')
    roi[roi < min_filtered_threshold] = 0

    #DEBUG
    if not noplots:
        n += 1
        plot_heatmap(roi, mass_axis, axes, 'Convolved with Filter after thresholding', '_' + str(n).zfill(2), file_id, outdir)


    ## FIND PEAKS WITH NON MAX SUPRESSION
    logging.info(f'{label}: Localizing peak candidates with non-max supression.')

    # NOTE: These parameters should probably be reconsidered
    # min_distance: minimum distance between peaks
    # removing exclude_border after removing padding
    roi_peaks = peak_local_max(roi, min_distance=7)

    ## SHIFT PEAK CENTER TO MAX VALUE
    logging.info(f'{label}: Shifting peak centers to max value.')
    w_y = 5  # max peak shift in mass [mass index] (int: odd)
    c = 5  # max peak shift in time [time index] (int: odd)
    roi_peaks = get_peak_center(roi_peaks, exp_no_background, w_y, c)

    #DEBUG
    if not noplots:
        peak_pos_df = {'x': roi_peaks[:, 1], 'y': roi_peaks[:, 0]}
        peak_pos_df = pd.DataFrame(data=peak_pos_df)
        n += 1
        plot_heatmap_with_peaks(exp_no_background, peak_pos_df, mass_axis, axes,
                                     'All Peaks found', '_' + str(n).zfill(2), file_id, outdir, savedata)

    ## Z-SCORE IN CONVOLVED DATA
    logging.info(f'{label}: Removing peak candidates with z-score in convolved data  < {str(min_SNR_conv)}')
    before = len(roi_peaks)
    # min_SNR_conv here is a configuration
    roi_peaks = filter_by_SNR(roi_peaks, roi, min_SNR_conv, window_x, window_y, center_x)
    logging.info(f'{label}: Removed ' + str(before - len(roi_peaks)) + ' peak candidates')

    #DEBUG
    if not noplots:
        peak_pos_df = {'x': roi_peaks[:, 1], 'y': roi_peaks[:, 0]}
        peak_pos_df = pd.DataFrame(data=peak_pos_df)
        n += 1
        plot_heatmap_with_peaks(exp_no_background, peak_pos_df, mass_axis, axes, 'Pre-filtered Peaks',
                                     '_' + str(n).zfill(2), file_id, outdir, savedata)

    ## KNOWN MASSES HANDLING
    if knowntraces:
        before = len(roi_peaks)
        # get list of known compounds
        logging.info(f'{label}: Removing peak candidates that are not in --masses.')
        known_peaks_bool = find_known_traces(mass_axis[roi_peaks[:,0]], compounds, masses_dist_max)
        roi_peaks = roi_peaks[known_peaks_bool]
        logging.info(f'{label}: Removed ' + str(before - len(roi_peaks)) + ' peak candidates')

        #DEBUG
        if not noplots:
            n += 1
            peak_pos_df = {'x': roi_peaks[:, 1], 'y': roi_peaks[:, 0]}
            peak_pos_df = pd.DataFrame(data=peak_pos_df)
            plot_heatmap_with_peaks(exp_no_background, peak_pos_df, mass_axis, axes,
                                         'Known Masses Pre-filtered Peaks', '_' + str(n).zfill(2), file_id, outdir, savedata)

    if not noplots:
        logging.info(f'{label}: Done plotting Heatmaps')

    return roi_peaks, background, exp_no_background


def zscore(trace, means, sigmas):
    '''Returns trace zscore given means and sigmas'''
    return (trace - means) / sigmas

def gaussian(x, b, amp, mean, std):
    '''gaussian function'''
    return b + (amp * np.exp(-((np.array(x) - mean) ** 2) / (2 * (std ** 2))))

def get_peak_properties(label, peaks, window_x, window_y, center_x, x_sigmas, exp, outdir):
    '''calculates peak properties (mass, time, height, volume, start time, end time, ...)

    Parameters
    ----------
    label: string
        Name of experiment, mainly used for logging.
    peaks: list
        List of detected peak coordinates
    window_x: int
        Maximum size of window on time axis to consider prior to gaussian fit
    window_y: int
        Size of window on mass axis to consider
    center_x: int
        Default size of window on time axis if gaussian fit fails
    exp: ndarray
        Raw data from file
    outdir: string
        Path to output directory, used for gaussian fit debug plots 

    Returns
    -------
    df: pandas dataframe
         peak_mass, peak_time, peak_height, peak_volume, start_time, end_time
    '''
    logging.info(f'{label}: Calculating peak properties from candidates.')

    peak_height_list = []
    peak_zscore_list = []
    peak_volume_list = []
    start_time_idx_list = []
    end_time_idx_list = []
    peak_base_width_list = []
    peak_background_abs_list = []
    peak_background_std_list = []
    peak_background_ratio_list = []
    peak_background_diff_list = []
    gauss_loss_list = []
    gauss_conv_list = []

    on_edge_list = []   # custom metric to later remove artefacts that come from spikes right at the beginning of experiment

    for peak in peaks:
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

            # if start is later than peak or end is before peak, didn't converge
            # on correct peak
            if start_diff > 0 or end_diff < 0:
                raise RuntimeError

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

        geval_scale = np.max(crop_1d)
        geval_truth = crop_1d / geval_scale
        geval_pred = pred / geval_scale
        geval_mse = ((geval_truth - geval_pred) ** 2).mean()
        gauss_loss_list.append(geval_mse)

        gauss_conv_list.append(converged)


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

        start_time_idx_list.append(start_time_idx)
        end_time_idx_list.append(end_time_idx)


        ## GET PEAK BASE WIDTH 
        peak_base_width = end_time_idx - start_time_idx
        peak_base_width_list.append(peak_base_width)

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
        peak_background_abs_list.append(peak_background_abs)
        peak_background_std = np.max([np.std(crop_l), np.std(crop_r)])
        peak_background_std_list.append(peak_background_std)

        peak_background_diff = abs(med_r - med_l)
        peak_background_diff_list.append(peak_background_diff)
        peak_background_ratio = np.min([med_l / (med_r + 1e-4), med_r / (med_l + 1e-4)])
        peak_background_ratio_list.append(peak_background_ratio)

        ## SUBTRACT BACKGROUND FROM RAW DATA
        bgsub_center = crop_center - peak_background_map
        bgsub_l = crop_l - peak_background_side
        bgsub_r = crop_r - peak_background_side

        ## GET ABSOLUTE PEAK HEIGHT
        # note that this is the peak height count with the background subtracted
        peak_height = bgsub_center[peak_center_y//2, peak_center_x//2]
        peak_height_list.append(peak_height)

        ## GET PEAK ZSCORE
        # peak_height is already background subtracted so no mean offset
        # sigmas recalculated below using bg subtracted values
        sigmas = np.max([np.std(bgsub_l), np.std(bgsub_r)])
        peak_zscore = zscore(peak_height, 0, sigmas+1e-2)
        peak_zscore_list.append(peak_zscore)

        ## GET PEAK VOLUME
        # only calculate peak volume before the start and end times
        peak_volume = np.sum(bgsub_center)
        peak_volume_list.append(peak_volume)

        ## FILTER OUT EXP START PEAKS
        # metric to later remove artefacts that come from spikes right at the beginning of experiment
        on_edge = 0
        if (np.sum(crop_l) == 0) & (np.sum(crop_r) > 0):
            on_edge = 1
        on_edge_list.append(on_edge)

    # write peak properties to pandas dataframe
    d = {'height': peak_height_list,
         'zscore': peak_zscore_list,
         'volume': peak_volume_list,
         'start_time_idx': start_time_idx_list,
         'end_time_idx': end_time_idx_list,
         'peak_base_width': peak_base_width_list,
         'mass_idx': peaks[:, 0],
         'time_idx': peaks[:, 1],
         'background_abs': peak_background_abs_list,
         'background_std': peak_background_std_list,
         'background_ratio': peak_background_ratio_list,
         'background_diff': peak_background_diff_list,
         'gauss_loss': gauss_loss_list,
         'gauss_conv': gauss_conv_list,
         'on_edge': on_edge_list}

    peak_properties = pd.DataFrame(data=d)

    return peak_properties


def down_select(label, peak_properties, min_zscore, min_peak_volume,
                noplots, mass_axis, exp_no_background, time_axis, file_id, outdir,
                savedata):
    '''Downselect found peak by their properties
    
    Parameters
    ----------
    label: string
        Name of experiment, mainly used for logging
    peak_properties: dataframe
        Peak properties with which to filter peaks
    min_zscore: int
        Minimum z-score to allow
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

    logging.info(f'{label}: Down-selecting peak candidates based on their properties.')

    # z-score
    before = len(peak_properties)
    peak_properties = peak_properties.loc[peak_properties['zscore'] >= min_zscore]
    logging.info(f'{label}: Removed {before - len(peak_properties)} peak candidates with z-score < {min_zscore}')
    
    # filter out non-converged
    before = len(peak_properties)
    peak_properties = peak_properties.loc[peak_properties['gauss_conv'] == 1]
    logging.info(f'{label}: Removed {before - len(peak_properties)} peak candidates for failed gaussian convergence')

    # threshold volume
    before = len(peak_properties)
    peak_properties = peak_properties.loc[peak_properties['volume'] >= min_peak_volume]
    logging.info(f'{label}: Removed {before - len(peak_properties)} peak candidates with Volume < {min_peak_volume}')

    # threshold base width - 5 is around 2 seconds
    before = len(peak_properties)
    peak_properties = peak_properties.loc[peak_properties['peak_base_width'] > 5]
    logging.info(f"{label}: Removed {before - len(peak_properties)} peak candidates with Base Width <= 5")

    # threshold migration time
    before = len(peak_properties)
    threshold_5min = find_nearest_index(time_axis, 5)
    peak_properties = peak_properties.loc[peak_properties['time_idx'] >= threshold_5min]
    logging.info(f"{label}: Removed {before - len(peak_properties)} peak candidates with migration time < 5min")

    # threshold gaussian loss
    #before = len(peak_properties)
    #peak_properties = peak_properties.loc[peak_properties['gauss_loss'] < 0.014]
    #logging.info(f"{label}: Removed {before -len(peak_properties)} peak candidates with gauss_loss <= 0.014")

    # edge filter
    before = len(peak_properties)
    peak_properties = peak_properties.loc[peak_properties['on_edge'] == 0]
    logging.info(f'{label}: Removed ' + str(before - len(peak_properties)) + ' peak candidates on edge of experiment data')

    if not noplots:
        axes = get_axes_ticks_and_labels(mass_axis, time_axis)
        peak_pos_df = {'x': peak_properties.time_idx, 'y': peak_properties.mass_idx}
        peak_pos_df = pd.DataFrame(data=peak_pos_df)
        n = 9  # plot ID
        plot_heatmap_with_peaks(exp_no_background, peak_pos_df, mass_axis, axes, 'Filtered Peaks',
                                '_' + str(n).zfill(2), file_id, outdir, savedata)

    logging.info(f'{label}: {str(len(peak_properties))} peaks remaining')

    return peak_properties


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

    Returns
    -------

    '''
    # runtime profiling
    start_time = time.time()

    filename = kwargs['file']
    label = kwargs['label']
    basedir = kwargs['basedir']

    ext = os.path.basename(filename).split('.')[-1]
    if ext == "raw":
        logging.info(f'Converting raw file: {str(filename)}')
        filename = convert_file(str(filename), basedir, label)
    elif ext == "pickle":
        pass
    else:
        logging.error('Error: invalid file extension for file ' + os.path.basename(filename))
        logging.error('Files should be either ".raw" format or ".pickle" format')
        return

    outdir = kwargs['outdir']

    ## Flags
    noplots = kwargs['noplots']
    noexcel = kwargs['noexcel']
    savedata = kwargs['saveheatmapdata']
    knowntraces = kwargs['knowntraces']
    debug_plot = kwargs.get('debug_plots')
    field_mode = kwargs.get('field_mode')
    if field_mode:
        noexcel = True
        noplots = True

    trace_window = 13

    # make sure trace window width is odd
    if not trace_window % 2 == 1:
        logging.warning(f'Malformed trace_window: {trace_window}')
        return

    file_id = kwargs['label']

    ## Reading data file
    data = pickle.load(open(Path(filename), 'rb'))
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

    sigma = args.get('sigma')
    sigma_ratio = args.get('sigma_ratio')
    min_filtered_threshold = args.get('min_filtered_threshold')
    min_peak_volume = args.get('min_peak_volume')
    min_SNR_conv = args.get('min_SNR_conv')
    min_zscore = args.get('min_zscore')
    masses_dist_max = args.get('masses_dist_max')

    denoise_x = args.get('denoise_x')
    window_x = args.get('window_x')
    window_y = args.get('window_y')
    center_x = args.get('center_x')
    x_sigmas = args.get('x_sigmas')

    # abort if experiment shape is too small
    if exp.shape[0] < window_y + 1 or exp.shape[1] < window_x:
        logging.error(f"{label} skipped, data shape {exp.shape} is too small")
        return
    
    
    # find peaks in raw data
    peaks, background, exp_no_background = find_peaks(label, exp, window_x, window_y, time_axis, mass_axis, noplots,
                                                      file_id, outdir, sigma, sigma_ratio, min_filtered_threshold,
                                                      savedata, min_SNR_conv, center_x, denoise_x, masses_dist_max, 
                                                      compounds, knowntraces)

    # determine peak properties of found peaks
    peak_properties = get_peak_properties(label, peaks, window_x, window_y, center_x, x_sigmas, exp, outdir)
    # downselect peaks further based on peak_properties
    peak_properties = down_select(label, peak_properties, min_zscore, min_peak_volume,
                                  noplots, mass_axis, exp_no_background,
                                  time_axis, file_id, outdir, savedata)

    # plot mugshots of peaks
    plot_mugshots(label, peak_properties, exp, time_axis, mass_axis, outdir)


    # write csv / excel
    if not field_mode:
        write_rawdata_csv(label, exp, time_axis, mass_axis, file_id, outdir, exp_no_background)

    if not noplots:
        # plot spectra
        plot_peak_vs_time(label, peak_properties, debug_plot, exp, mass_axis, time_axis, outdir, center_x, window_x, window_y, trace_window, knowntraces, compounds)

        # plot spectra
        plot_peak_vs_mass(label, peak_properties, debug_plot, exp, mass_axis, time_axis, outdir, center_x, window_x, window_y, trace_window, exp_no_background)

        if debug_plot:
            # additional debug plots
            plot_peak_vs_mass_time(label, peak_properties, exp_no_background, mass_axis, time_axis, outdir, center_x, window_x, window_y)
    
    #write peaks to csv
    peak_properties_exp = peak_properties.copy(deep=True)
    data_peaks = write_peaks_csv(peak_properties_exp, mean_time_diff, mass_axis, time_axis, outdir, label, knowntraces, compounds)

    # write peaks to excel
    if not noexcel:
        write_excel(label, peak_properties, exp_no_background, exp, mass_axis, time_axis, knowntraces, compounds, file_id, basedir, outdir, data_peaks, mass_pos_tr, mass_pos_sp, time_pos, data_traces_counts, data_traces_basesub, data_spectra_counts)

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
        logging.info(f"{label}: Saved Background, {filesize_kB:.2f} kB")
    else:
        logging.info(f"{label}: Skipped Background, knowntraces")

    # write Total Ion Count to csv
    tic = total_ion_count(exp, 1)
    tic_filepath = op.join(outdir, label+"_tic.csv")
    filesize_kB = write_tic(tic, tic_filepath)
    logging.info(f"{label}: Saved Total Ion Count, {filesize_kB:.2f} kB")

    # calculate Science Utility Estimate (SUE) and Diversity Descriptor (DD)
    SUE_filepath = op.join(outdir, label + '_SUE.csv')
    calc_SUE(label, peak_properties, kwargs['sue_weights'], compounds, mass_axis, masses_dist_max, SUE_filepath)
    DD_filepath = op.join(outdir, label + '_DD.csv')
    diversity_descriptor(label, peak_properties, kwargs['dd_weights'], compounds, mass_axis, masses_dist_max, DD_filepath)

    # write asdp manifest
    asdp_list = []

    # background
    asdp_list.append([
        op.join(outdir, label+'_background.bz2'),
        'background_summary',
        'acme',
        'asdp'
    ])
    asdp_list.append([
        op.join(outdir, label+'_tic.csv'),
        'total_ion_count',
        'acme',
        'asdp'
    ])

    # peaks
    asdp_list.append([
        op.join(outdir, label+'_UM_peaks.csv'),
        'peak_properties',
        'acme',
        'asdp'
    ])
    asdp_list.append([
        op.join(outdir, 'Mugshots'),
        'peak_mugshots',
        'acme',
        'asdp'
    ])

    # sue/dd
    asdp_list.append([
        op.join(outdir, label+'_SUE.csv'),
        'science_utility',
        'acme',
        'metadata'
    ])

    asdp_list.append([
        op.join(outdir, label+'_DD.csv'),
        'diversity_descriptor',
        'acme',
        'metadata'
    ])

    write_manifest(asdp_list, op.join(outdir, label+'_manifest.csv'))

    # print execution time
    end_time = time.time()
    duration = end_time - start_time

    logging.info(f'{label}: Finished processing file in ' + str(round(duration,1)) + ' seconds')
