# Functions for ACME analyzer that are being reused elsewhere
# these are called from analyzer.py
import sys
import os

import logging

import numpy  as np
import pandas as pd

def make_crop(peak, roi, window_x, window_y, center_x):
    '''makes two crops: one of the peak and one of the surrounding area

    Parameters
    ----------
    peak: ndarray
        x and y coordinate of peak center
    roi: ndarray
        matrix to be cropped
    window_x: int
        total width of crop window. Must be odd
    window_y, int
        total height of crop window. Must be odd
    center_x: int
        center width that we from window. Must be odd

    Returns
    -------
    crop_center: ndarray
        center crop of crop (contains peak)
    crop_l: ndarray
        crop of window without center (contains left side of peak)
    crop_r: ndarray
        crop of window without center (contains right side of peak)
    '''
    # make sure all inputs are odd
    if not window_x % 2 == 1:
        logging.warning(f'Malformed window_x: {window_x}')
        return

    if not window_y % 2 == 1:
        logging.warning(f'Malformed window_y: {window_y}')
        return

    if not center_x % 2 == 1:
        logging.warning(f'Malformed center_x: {center_x}')
        return

    top = - (window_y // 2) + peak[0]
    bottom = window_y // 2 + peak[0] + 1
    left = - (window_x // 2) + peak[1]
    right = window_x // 2 + peak[1] + 1
    crop = roi[top: bottom, left: right]
    crop_center = crop[:, window_x // 2 - (center_x // 2): window_x // 2 + center_x // 2 + 1]
    crop_left = crop[:, : window_x // 2 - (center_x // 2)]
    crop_right = crop[:, window_x // 2 + center_x // 2 + 1:]
    # crop_lr = np.concatenate((crop_left, crop_right), 1)
    return crop_center, crop_left, crop_right

def find_nearest_index(array, value):
    '''
        Finds index of the value inside the array nearest to input 'value'
    '''
    if (value > np.max(array)):
        logging.warning('mass axis in heatmap plots: label mismatch')
        return len(array) - 1
    if (value < np.min(array)):
        logging.warning('mass axis in heatmap plots: label mismatch')
        return 0
    diff = np.subtract(array, value)
    return np.argmin(abs(diff))

def write_rawdata_csv(label, exp, time_axis, mass_axis, file_id, outdir, exp_no_background):
    '''export raw data as csv'''
    logging.info(f'{label}: Writing csv of Raw Data.')
    csv_data_w_time = np.vstack((time_axis, exp))
    mass_axis = list(mass_axis)
    mass_axis.insert(0, np.nan)
    mass_axis = np.array(mass_axis).reshape(-1, 1)
    csv_data_w_time_and_mass = np.concatenate([mass_axis, csv_data_w_time], axis=1)

    df = pd.DataFrame(data=csv_data_w_time_and_mass)
    df.to_csv(os.path.join(outdir, "Heat_Maps", file_id + '_Raw_Counts.csv'), sep=',', float_format='%.3f', index=False)

    logging.info(f'{label}: Writing csv of Background Removed Data.')
    csv_data_w_time = np.vstack((time_axis, exp_no_background))
    csv_data_w_time_and_mass = np.concatenate([mass_axis, csv_data_w_time], axis=1)

    df = pd.DataFrame(data=csv_data_w_time_and_mass)
    df.to_csv(os.path.join(outdir, "Heat_Maps", file_id + '_Background_Removed_Counts.csv'), sep=',', float_format='%.3f', index=False)

def write_peaks_csv(peak_properties, mean_time_diff, mass_axis, time_axis, outdir, label, knowntraces, compounds):
    '''Write pandas dataframe to csv file. This csv file is also used for performance evaluation'''

    logging.info(f'{label}: Writing csv of found peaks.')

    # rename variables
    peak_properties.rename(columns={'height': 'Peak Amplitude (Counts)', 'zscore': 'Peak Amplitude (ZScore)',
                                    'time_idx': 'Peak Central Time (idx)', 'mass_idx': 'Mass (idx)',
                                    'volume': 'Peak Volume (Counts)'}, inplace=True)

    # convert variables
    peak_properties['Peak Base Width (sec)'] = peak_properties['peak_base_width'] * mean_time_diff * 60  # from idx to sec

    # add mass and time in amu and minutes to improve human readability
    peak_properties['Mass (amu)'] = mass_axis[peak_properties['Mass (idx)']]
    peak_properties['Peak Central Time (Min)'] = time_axis[peak_properties['Peak Central Time (idx)']]
    peak_properties['Peak Left Time (Min)'] = time_axis[peak_properties['start_time_idx']]
    peak_properties['Peak Right Time (Min)'] = time_axis[peak_properties['end_time_idx']]

    data_peaks = peak_properties

    # reorder peak_properties for ACME scientists
    new_order = ['Peak Central Time (Min)', 'Mass (amu)', 'Peak Volume (Counts)', 'Peak Amplitude (Counts)',
     'Peak Amplitude (ZScore)','Peak Base Width (sec)', 'Peak Left Time (Min)', 'Peak Right Time (Min)',
     'volume_top', 'volume_zscore', 'start_time_idx', 'end_time_idx', 'peak_base_width',
     'Mass (idx)', 'Peak Central Time (idx)','background_abs', 'background_std', 'background_ratio', 
     'background_diff', 'gauss_loss']

    data_peaks = data_peaks[new_order]

    # add compound name
    compounds_name = []
    compounds_amu = np.array(list(compounds.keys()))
    if knowntraces:
        # find closest fit in compounds
        for peak_mass in data_peaks['Mass (amu)']:
            mass_dist = np.abs(peak_mass - compounds_amu)  # find distance between peak mass and known compounds mass
            best_fit = np.argmin(mass_dist)  # find distance to closest known compound mass
            compounds_name.append(compounds[compounds_amu[best_fit]])
        # add names to dataframe
        data_peaks.insert(0, 'Compounds', compounds_name)

    if len(data_peaks) > 0:
        # sort by Peak Central time
        data_peaks = data_peaks.groupby(['Peak Central Time (Min)']).apply(
            lambda _df: _df.sort_values(by=['Peak Central Time (Min)']))

        # round values
        data_peaks['Peak Central Time (Min)'] = np.round(data_peaks['Peak Central Time (Min)'],2)
        data_peaks['Mass (amu)'] = np.round(data_peaks['Mass (amu)'], 2)
        data_peaks['Peak Volume (Counts)'] = np.round(data_peaks['Peak Volume (Counts)'], 0)
        data_peaks['Peak Amplitude (Counts)'] = np.round(data_peaks['Peak Amplitude (Counts)'], 0)
        data_peaks['Peak Amplitude (ZScore)'] = np.round(data_peaks['Peak Amplitude (ZScore)'], 1)
        data_peaks['Peak Base Width (sec)'] = np.round(data_peaks['Peak Base Width (sec)'], 1)
        data_peaks['Peak Left Time (Min)'] = np.round(data_peaks['Peak Left Time (Min)'], 2)
        data_peaks['Peak Right Time (Min)'] = np.round(data_peaks['Peak Right Time (Min)'], 2)

    # rename excel file according to known or unknown traces
    if knowntraces:
        label = label + '_KM'
    else:
        label = label + '_UM'
    name = label + '_peaks'
    savepath = os.path.join(outdir, name + '.csv')

    data_peaks.to_csv(savepath, index=False, float_format='%.3f')
    return data_peaks

def write_excel(label, peak_properties, exp_no_background, exp, mass_axis, time_axis, knowntraces, compounds, file_id, basedir, outdir, data_peaks, mass_pos_tr, mass_pos_sp, time_pos, data_traces_counts, data_traces_basesub, data_spectra_counts):
    '''Writes found peaks to excel sheet'''

    # sort by peak central time
    if len(peak_properties) > 0:
        # sort by Peak Central time
        peak_properties = peak_properties.groupby(['time_idx']).apply(
            lambda _df: _df.sort_values(by=['time_idx']))

    ## preparing data for write_excel
    logging.info(f'{label}: Writing excel of found peaks ...')
    # make traces tabs
    i = 0
    # iterate over found peaks for counts vs time
    for peak in peak_properties.itertuples():
        i += 1
        mass = round(mass_axis[peak.mass_idx], 2)
        trace = exp[peak.mass_idx, :]  # raw data
        trace_basesub = exp_no_background[peak.mass_idx, :]  # raw data - background

        mass_pos_tr[i] = mass
        data_traces_counts.insert(i, 'Counts', trace, allow_duplicates=True)
        data_traces_basesub.insert(i, 'Counts', trace_basesub, allow_duplicates=True)

    i = 0
    # iterate over found peaks for counts vs mass
    for peak in peak_properties.itertuples():
        i += 1
        time = round(time_axis[peak.time_idx], 2)
        mass = round(mass_axis[peak.mass_idx], 2)
        spectra = exp[:, peak.time_idx]  # raw data

        time_pos[i] = time
        mass_pos_sp[i] = mass
        data_spectra_counts.insert(i, 'Counts', spectra, allow_duplicates=True)

    ## Create a Pandas Excel writer using XlsxWriter as the engine.
    # rename excel file according to known or unknown traces
    if knowntraces:
        file_id = file_id + '_KM'
    else:
        file_id = file_id + '_UM'

    writer = pd.ExcelWriter(os.path.join(outdir, file_id + '.xlsx'), engine='xlsxwriter')

    # remove unwanted variables from excel output
    data_peaks.drop(columns=['volume_top', 'volume_zscore',
                                  'start_time_idx', 'end_time_idx',
                                  'background_abs', 'background_std', 'peak_base_width',
                                  'background_ratio', 'background_diff', 'gauss_loss'], inplace=True)

    # rename variables as needed
    data_peaks.rename(columns = {'volume': 'Peak Volume (Counts)'}, inplace = True)

    # Write each dataframe to a different worksheet
    data_peaks.to_excel(writer, startrow=6, startcol=0, sheet_name='Peaks', index=False)
    data_traces_counts.to_excel(writer, startrow=7, startcol=0, sheet_name='Traces (Counts)', index=False)
    data_traces_basesub.to_excel(writer, startrow=7, startcol=0, sheet_name='Traces (Counts-Baseline)', index=False)
    data_spectra_counts.to_excel(writer, startrow=7, startcol=0, sheet_name='Spectra (Counts)', index=False)

    # Adding relevant info to each sheet
    peaks_sheet = writer.sheets['Peaks']
    peaks_sheet.write(0, 0, "Run Command: python -W ignore " + " ".join(sys.argv))
    peaks_sheet.write(2, 0, "Filename: ")
    peaks_sheet.write(2, 1, file_id)
    peaks_sheet.write(3, 0, "Folder Path: ")
    peaks_sheet.write(3, 1, basedir)

    traces_sheet = writer.sheets['Traces (Counts)']
    traces_sheet.write(0, 0, "Run Command: python -W ignore " + " ".join(sys.argv))
    traces_sheet.write(2, 0, "Filename: ")
    traces_sheet.write(2, 1, file_id)
    traces_sheet.write(3, 0, "Folder Path: ")
    traces_sheet.write(3, 1, basedir)
    traces_sheet.write(5, 0, "Time: ")
    traces_sheet.write(6, 0, "Mass: ")

    for m in mass_pos_tr.keys():
        traces_sheet.write(6, m, mass_pos_tr[m])
    for t_pos in time_pos.keys():
        traces_sheet.write(5, t_pos, round(time_pos[t_pos], 2))

    traces_sheet = writer.sheets['Traces (Counts-Baseline)']
    traces_sheet.write(0, 0, "Run Command: python -W ignore " + " ".join(sys.argv))
    traces_sheet.write(2, 0, "Filename: ")
    traces_sheet.write(2, 1, file_id)
    traces_sheet.write(3, 0, "Folder Path: ")
    traces_sheet.write(3, 1, basedir)
    traces_sheet.write(5, 0, "Time: ")
    traces_sheet.write(6, 0, "Mass: ")
    for m in mass_pos_tr.keys():
        traces_sheet.write(6, m, mass_pos_tr[m])
    for t_pos in time_pos.keys():
        traces_sheet.write(5, t_pos, round(time_pos[t_pos], 2))

    spectra_sheet = writer.sheets['Spectra (Counts)']
    spectra_sheet.write(0, 0, "Run Command: python -W ignore " + " ".join(sys.argv))
    spectra_sheet.write(2, 0, "Filename: ")
    spectra_sheet.write(2, 1, file_id)
    spectra_sheet.write(3, 0, "Folder Path: ")
    spectra_sheet.write(3, 1, basedir)
    spectra_sheet.write(5, 0, "Time: ")
    spectra_sheet.write(6, 0, "Mass: ")
    for m_pos in mass_pos_sp.keys():
        spectra_sheet.write(6, m_pos, mass_pos_sp[m_pos])
    for t_pos in time_pos.keys():
        spectra_sheet.write(5, t_pos, round(time_pos[t_pos], 2))

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

    logging.info(f'{label}: Done writing excel')

    return mass_pos_tr, data_traces_counts, data_traces_basesub, time_pos, mass_pos_sp, data_spectra_counts

def find_known_traces(peak_masses, compounds, masses_dist_max):
    compounds = np.array(list(compounds.keys()))
    min_dists = []
    for peak_mass in peak_masses:
        mass_dist = np.abs(peak_mass - compounds)  # find distance between peak mass and known compounds mass
        min_dists.append(np.min(mass_dist))  # append result to list
    min_dists = np.array(min_dists)
    known_peaks = min_dists < masses_dist_max  # make boolean for peaks that are known

    return known_peaks
