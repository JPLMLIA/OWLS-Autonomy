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
import os
import logging
import time
import pickle
import csv
import scipy.signal

import numpy   as np
import pandas  as pd
import os.path as op

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
                                       write_jpeg2000, read_jpeg2000, \
                                       compress_background_PCA, reconstruct_background_PCA, \
                                       compress_background_smartgrid, reconstruct_background_smartgrid, \
                                       remove_peaks, overlay_peaks, total_ion_count

from acme_cems.lib.JEWEL_in     import calc_SUE, diversity_descriptor


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


def add_padding(label, exp, window_x, window_y, time_axis, mass_axis):
    '''add zero padding to deal with boarders of dataset'''
    # will add self.window_x zeros to left and right boarder and self.window_y to top and bottom

    # make empty data matrix
    exp_0 = np.zeros((np.shape(exp)[0] + 2 * window_y, np.shape(exp)[1] + 2 * window_x))
    # place data in center
    exp_0[window_y:np.shape(exp)[0] + window_y, window_x:np.shape(exp)[1] + window_x] = exp
    exp = exp_0
    # zero pad time_axis
    time_axis = np.concatenate((np.zeros(window_x,), time_axis, np.zeros(window_x,)))
    # zero pad mass_axis
    mass_axis = np.concatenate((np.zeros(window_y,), mass_axis, np.zeros(window_y,)))

    #check that axis agree with new matrix shape
    if not (mass_axis.shape[0] == exp.shape[0]):
        logging.warning(f'{label}: Malformed mass axis: {mass_axis.shape[0]}')
        return exp, time_axis, mass_axis

    if not (time_axis.shape[0] == exp.shape[1]):
        logging.warning(f'{label}: Malformed time axis shape: {time_axis.shape[0]}')
        return exp, time_axis, mass_axis

    return exp, time_axis, mass_axis


def find_peaks(label, exp, window_x, window_y, time_axis, mass_axis, noplots, file_id, outdir, sigma, sigma_ratio,
                   min_filtered_threshold, savedata, min_SNR_conv, center_x, masses_dist_max, compounds, knowntraces):
    '''finds peaks in 2D from raw ACME data

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
    min_peak_height: float
        threshold to filter peaks below a set height
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
    roi_peaks: list
        found peaks
    background: ndarray
        raw background
    exp_no_backgroound: ndarray
        raw data with background removed
    '''
    axes = get_axes_ticks_and_labels(mass_axis, time_axis)
    
    if not noplots:
        n = 1
        plot_heatmap(exp, mass_axis, axes, 'Raw Data', '_' + str(n).zfill(2), file_id, outdir)

    ## find reagons of interest
    roi = np.copy(exp)

    # remove background with median filter
    logging.info(f'{label}: Removing background with median filter.')
    background = ndimage.median_filter(roi, size=[1, window_x])
    roi -= background
    exp_no_background = np.copy(roi)  # save variable for plotting
    
    if not noplots:
        n += 1
        plot_heatmap(background, mass_axis, axes, 'Background', '_' + str(n).zfill(2), file_id, outdir)
        n += 1
        plot_heatmap(roi, mass_axis, axes, 'Data with Background subtracted', '_' + str(n).zfill(2), file_id, outdir)

    # remove negative values
    roi[roi < 0] = 0
    
    ## enhance SNR of blobs
    logging.info(f'{label}: Applying difference of Gaussian filter.')
    blob_filter = diff_gauss(sigma, sigma_ratio)  # generate filter
    roi = scipy.signal.convolve2d(roi, blob_filter, mode='same')  # apply filter
    if not noplots:
        n += 1
        plot_heatmap(roi, mass_axis, axes, 'Convolved with Filter', '_' + str(n).zfill(2), file_id, outdir)

    # remove everything below a set threshold
    logging.info(f'{label}: Setting values < {str(min_filtered_threshold)} to zero.')
    roi[roi < min_filtered_threshold] = 0
    if not noplots:
        n += 1
        plot_heatmap(roi, mass_axis, axes, 'Convolved with Filter after thresholding', '_' + str(n).zfill(2), file_id, outdir)

    # find peak candidates with non max supression
    # exclude_boarder needs to be bigger than window_x//2
    logging.info(f'{label}: Localizing peak candidates with non-max supression.')
    roi_peaks = peak_local_max(roi, min_distance=7, exclude_border=40)
    
    ## do postprocessing of peak location on raw data
    logging.info(f'{label}: Shifting peak centers to max value.')
    # center found peaks so they are on the max value
    w_y = 5  # 2* max peak shift in mass [mass index] (int: odd)
    c = 5  # 2* max peak shift in time [time index] (int: odd)
    roi_peaks = get_peak_center(roi_peaks, exp_no_background, w_y, c)

    if not noplots:
        peak_pos_df = {'x': roi_peaks[:, 1], 'y': roi_peaks[:, 0]}
        peak_pos_df = pd.DataFrame(data=peak_pos_df)
        n += 1
        plot_heatmap_with_peaks(exp_no_background, peak_pos_df, mass_axis, axes,
                                     'All Peaks found', '_' + str(n).zfill(2), file_id, outdir, savedata)
    
    # filter peak candidates by surrounding standard deviation in convolved data
    logging.info(f'{label}: Removing peak candidates with z-score in convolved data  < {str(min_SNR_conv)}')
    before = len(roi_peaks)
    roi_peaks = filter_by_SNR(roi_peaks, roi, min_SNR_conv, window_x, window_y, center_x)
    logging.info(f'{label}: Removed ' + str(before - len(roi_peaks)) + ' peak candidates')

    if not noplots:
        peak_pos_df = {'x': roi_peaks[:, 1], 'y': roi_peaks[:, 0]}
        peak_pos_df = pd.DataFrame(data=peak_pos_df)
        n += 1
        plot_heatmap_with_peaks(exp_no_background, peak_pos_df, mass_axis, axes, 'Pre-filtered Peaks',
                                     '_' + str(n).zfill(2), file_id, outdir, savedata)

    # remove peaks to far seperated from known masses if known_masses = True
    if knowntraces:
        before = len(roi_peaks)
        # get list of known compounds
        logging.info(f'{label}: Removing peak candidates that are not in --masses.')
        known_peaks_bool = find_known_traces(mass_axis[roi_peaks[:,0]], compounds, masses_dist_max)
        roi_peaks = roi_peaks[known_peaks_bool]
        logging.info(f'{label}: Removed ' + str(before - len(roi_peaks)) + ' peak candidates')

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
    '''
        Returns trace zscore given means and sigmas
    '''
    return (trace - means) / sigmas


def get_peak_properties(label, peaks, window_x, window_y, center_x, exp):
    '''calculates peak properties (mass, time, height, volume, start time, end time, ...)

    Parameters
    ----------
    peaks: list
        found peaks
    exp: ndarray
        raw data from file
    mass_axis: ndarray
        mass axis
    time_axis: ndarray
        time axis

    Returns
    -------
    df: pandas dataframe
         peak_mass, peak_time, peak_height, peak_volume, start_time, end_time
    '''
    logging.info(f'{label}: Calculating peak properties from candidates.')

    peak_height_list = []
    peak_zscore_list = []
    peak_volume_list = []
    peak_volume_top_list = []
    start_time_idx_list = []
    end_time_idx_list = []
    peak_base_width_list = []
    peak_background_abs_list = []
    peak_background_std_list = []
    peak_background_ratio_list = []
    peak_background_diff_list = []
    volume_zscore_list =[]
    gauss_loss_list = []

    c1_flag_list = []   # custom metric to later remove artefacts that come from spikes right at the beginning of experiment

    for peak in peaks:
        ## MAKE CROP
        crop_center, crop_l, crop_r = make_crop(peak, exp, window_x, window_y, center_x)
        
        ## BACKGROUND STATISTICS
        med_l = np.median(crop_l)
        med_r = np.median(crop_r)
        peak_background_abs = np.max([med_l, med_r])
        peak_background_abs_list.append(peak_background_abs)
        peak_background_std = np.max([np.std(crop_l), np.std(crop_r)])
        peak_background_std_list.append(peak_background_std)

        peak_background_diff = abs(med_r - med_l)
        peak_background_diff_list.append(peak_background_diff)
        peak_background_ratio = np.min([med_l / (med_r + 1e-4), med_r / (med_l + 1e-4)])
        peak_background_ratio_list.append(peak_background_ratio)

        ## SUBTRACT BACKGROUND
        crop_center = np.copy(crop_center - peak_background_abs)

        ## GET ABSOLUTE PEAK HEIGHT
        peak_height = exp[peak[0], peak[1]] - peak_background_abs
        peak_height_list.append(peak_height)

        ## GET PEAK ZSCORE
        trace = exp[peak[0], peak[1]]
        means = peak_background_abs
        sigmas = peak_background_std
        peak_zscore = zscore(trace, means, sigmas)
        peak_zscore_list.append(peak_zscore)

        ## GET PEAK START AND END INDICES
        
        # get just center mass of time peak
        center1d = crop_center[window_y // 2]

        # initial guesses for gaussian curve optimization
        g_amp = peak_height
        g_mean = center_x // 2
        g_std = center_x // 4
        
        def gaussian(x, amp, mean, std):
            """Gaussian function"""
            return amp * np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))

        try:
            # fit curve onto spike
            popt, _ = curve_fit(gaussian, range(center_x), center1d, p0=[g_amp, g_mean, g_std], maxfev=100)
            fit_mean = popt[1]          # fit mean
            fit_std = abs(popt[2])      # fit std

            # width of deviations on each side
            sigmas = 2.5

            # get start offset from center
            start_diff = np.around((fit_mean - g_mean) - (fit_std * sigmas))
            if start_diff < (- center_x // 2) + 1:
                start_diff = (- center_x // 2) + 1

            # get end offset from center 
            end_diff = np.around((fit_mean - g_mean) + (fit_std * sigmas))
            if end_diff > center_x // 2 + 1:
                end_diff = center_x // 2 + 1

            # to calculate loss used during opt
            pred = gaussian(range(center_x), *popt)

        except RuntimeError:
            # couldn't fit curve, set start/end as window limits
            start_diff = - center_x // 2 + 1
            end_diff = center_x // 2 + 1

            # to calculate loss used during opt
            pred = np.zeros(center1d.shape)
        
        start_diff = int(start_diff)
        end_diff = int(end_diff)

        # calculate mse
        mse = ((center1d - pred) ** 2).mean()
        gauss_loss_list.append(mse)

        # get absolute timestamps and append, filtering for padding
        start_time_idx = int(peak[1] + start_diff)
        if start_time_idx < window_x:
            start_time_idx = window_x

        end_time_idx = int(peak[1] + end_diff)
        if end_time_idx >= exp.shape[1] - window_x:
            end_time_idx = exp.shape[1] - window_x - 1

        start_time_idx_list.append(start_time_idx)
        end_time_idx_list.append(end_time_idx)

        ## GET PEAK BASE WIDTH 
        peak_base_width = end_time_idx - start_time_idx
        peak_base_width_list.append(peak_base_width)

        ## GET PEAK VOLUME
        # only calculate peak volume before the start and end times
        peak_volume = np.sum(crop_center[:,(center_x//2 + start_diff):(center_x//2 + end_diff)])
        peak_volume_list.append(peak_volume)

        ## get volume of top 50% in center of center
        center_x_half = center_x // 2
        window_y_half = window_y // 2
        if not center_x_half % 2 == 1:  #ensure center width is odd
            center_x_half += 1
        if not window_y_half % 2 == 1:  #ensure window width in y is odd
            window_y_half += 1
        crop_crop_center, _, _ = make_crop(peak, exp, window_x, window_y_half, center_x_half)
        crop_crop_center = np.copy(crop_crop_center)
        crop_crop_center -= peak_background_abs
        peak_volume_top = np.sum(crop_crop_center[crop_crop_center > peak_height * 0.5])
        peak_volume_top_list.append(peak_volume_top)

        # get volume of top 50% and devide by surrounding std
        volume_zscore = peak_volume_top / peak_height
        volume_zscore_list.append(volume_zscore)


        ## FILTER OUT EXP START PEAKS
        # metric to later remove artefacts that come from spikes right at the beginning of experiment
        if (np.median(crop_l) == 0) & (np.median(crop_r) > 0):
            c1_flag_list.append(1)
        else:
            c1_flag_list.append(0)

    # write peak properties to pandas dataframe
    d = {'height': peak_height_list,
         'zscore': peak_zscore_list,
         'volume': peak_volume_list,
         'volume_top': peak_volume_top_list,
         'volume_zscore': volume_zscore_list,
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
         'c1_flag': c1_flag_list}

    peak_properties = pd.DataFrame(data=d)

    return peak_properties


def remove_padding(label, peak_properties, exp, time_axis, mass_axis, window_x, window_y, exp_no_background, background):
    '''removes previously added zero padding and adjust peak coordinates'''
    # remove padding from data matrix
    exp = exp[window_y:-window_y, window_x:-window_x]
    background = background[window_y:-window_y, window_x:-window_x]
    exp_no_background = exp_no_background[window_y:-window_y, window_x:-window_x]
    # remove padding from time_axis
    time_axis = time_axis[window_x:-window_x]
    # remove padding from mass axis
    mass_axis = mass_axis[window_y:-window_y]

    if not (mass_axis.shape[0] == exp.shape[0]):
        logging.warning(f'{label}: Malformed mass axis shape: {mass_axis.shape[0]}')
        return peak_properties, mass_axis, time_axis, exp_no_background, background, exp

    if not (time_axis.shape[0] == exp.shape[1]):
        logging.warning(f'{label}: Malformed time axis shape: {time_axis.shape[0]}')
        return peak_properties, mass_axis, time_axis, exp_no_background, background, exp

    # shift peak positions
    peak_properties['start_time_idx'] -= window_x
    peak_properties['end_time_idx'] -= window_x
    peak_properties['time_idx'] -= window_x
    peak_properties['mass_idx'] -= window_y

    return peak_properties, mass_axis, time_axis, exp_no_background, background, exp


def down_select(label, peak_properties, min_zscore, min_peak_volume, min_peak_volume_top, min_peak_volume_zscore,
                min_peak_height, noplots, mass_axis, exp_no_background, time_axis, file_id, outdir,
                savedata):
    '''Downselect found peak by their properties'''

    logging.info(f'{label}: Down-selecting peak candidates based on their properties.')

    # z-score
    before = len(peak_properties)
    peak_properties = peak_properties.loc[peak_properties['zscore'] > min_zscore]
    logging.info(f'{label}: Removed ' + str(before - len(peak_properties)) + ' peak candidates with z-score < '
                 + str(min_zscore))

    # threshold volume
    before = len(peak_properties)
    peak_properties = peak_properties.loc[peak_properties['volume'] > min_peak_volume]
    logging.info(f'{label}: Removed ' + str(before - len(peak_properties)) + ' peak candidates with Volume < '
                 + str(min_peak_volume))

    # threshold volume of top 50% (good to remove skinny peaks)
    before = len(peak_properties)
    peak_properties = peak_properties.loc[peak_properties['volume_top'] > min_peak_volume_top]
    logging.info(f'{label}: Removed ' + str(before - len(peak_properties)) + ' peak candidates with Volume_top < '
                 + str(min_peak_volume_top))

    # threshold volume of top 50% (good to remove skinny peaks)
    before = len(peak_properties)
    peak_properties = peak_properties.loc[peak_properties['volume_zscore'] > min_peak_volume_zscore]
    logging.info(f'{label}: Removed ' + str(before - len(peak_properties)) + ' peak candidates with Volume_zscore < '
                 + str(min_peak_volume_zscore))

    # threshold height
    before = len(peak_properties)
    peak_properties = peak_properties.loc[peak_properties['height'] > min_peak_height]
    logging.info(f'{label}: Removed ' + str(before - len(peak_properties)) + ' peak candidates with height < '
                 + str(min_peak_height))

    # use custom flags
    before = len(peak_properties)
    peak_properties = peak_properties.loc[peak_properties['c1_flag'] == 0]
    logging.info(f'{label}: Removed ' + str(before - len(peak_properties)) + ' peak candidates with c1 flag = 0')

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
    min_peak_height: float
        threshold to filter peaks below a set height
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
    start_time = time.time()

    file = kwargs['file']
    label = kwargs['label']
    basedir = kwargs['basedir']

    ext = os.path.basename(file).split('.')[-1]
    if ext == "raw":
        logging.info(f'Converting raw file: {str(file)}')
        file = convert_file(str(file), basedir, label)
    elif ext == "pickle":
        file = file
    else:
        logging.error('Error: invalid file extension for file ' + os.path.basename(file))
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
    file = pickle.load(open(Path(file), 'rb'))
    time_axis = file['time_axis']
    mass_axis = file['mass_axis']
    exp = file['matrix']
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
    min_peak_height = args.get('min_peak_height')
    min_peak_volume = args.get('min_peak_volume')
    min_peak_volume_top = args.get('min_peak_volume_top')
    min_peak_volume_zscore = args.get('min_peak_volume_zscore')
    min_SNR_conv = args.get('min_SNR_conv')
    min_zscore = args.get('min_zscore')
    masses_dist_max = args.get('masses_dist_max')

    window_x = args.get('window_x')
    window_y = args.get('window_y')
    center_x = args.get('center_x')

    # abort if experiment shape is too small
    if exp.shape[0] < window_y + 1 or exp.shape[1] < window_x:
        logging.error(f"{label} skipped, data shape {exp.shape} is too small")
        return
    
    # add zero padding
    exp, time_axis, mass_axis = add_padding(label, exp, window_x, window_y, time_axis, mass_axis)
    
    # find peaks in raw data
    peaks, background, exp_no_background = find_peaks(label, exp, window_x, window_y, time_axis, mass_axis, noplots,
                                                      file_id, outdir, sigma, sigma_ratio, min_filtered_threshold,
                                                      savedata, min_SNR_conv, center_x, masses_dist_max, compounds,
                                                      knowntraces)

    # determine peak properties of found peaks
    peak_properties = get_peak_properties(label, peaks, window_x, window_y, center_x, exp)
    # downselect peaks further based on peak_properties
    peak_properties = down_select(label, peak_properties, min_zscore, min_peak_volume, min_peak_volume_top,
                                  min_peak_volume_zscore, min_peak_height, noplots, mass_axis, exp_no_background,
                                  time_axis, file_id, outdir, savedata)

    # plot mugshots of peaks
    plot_mugshots(label, peak_properties, exp, time_axis, mass_axis, outdir)

    # remove zero padding and convert peak coordinates
    peak_properties, mass_axis, time_axis, exp_no_background, background, exp = remove_padding(label, peak_properties,
                                                                                               exp, time_axis,
                                                                                               mass_axis, window_x,
                                                                                               window_y,
                                                                                               exp_no_background,
                                                                                               background)

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
    filesize_kB = write_csv(tic, tic_filepath)
    logging.info(f"{label}: Saved Total Ion Count, {filesize_kB:.2f} kB")

    # calculate Science Utility Estimate (SUE) and Diversity Descriptor (DD)
    SUE_filepath = op.join(outdir, label + '_SUE.csv')
    calc_SUE(label, peak_properties, kwargs['sue_weights'], compounds, mass_axis, masses_dist_max, SUE_filepath)
    DD_filepath = op.join(outdir, label + '_DD.csv')
    diversity_descriptor(label, peak_properties, kwargs['dd_weights'], compounds, mass_axis, masses_dist_max, DD_filepath)

    # print execution time
    end_time = time.time()
    duration = end_time - start_time

    logging.info(f'{label}: Finished processing file in ' + str(round(duration,1)) + ' seconds')
