# functions that generate input for JEWEL
# SUE and Diversity Descriptor
# these are called from analyzer.py
import yaml
import logging

import numpy  as np
import pandas as pd

from acme_cems.lib.utils import find_known_traces

def calc_SUE(label, peak_properties, sue_weights_path, compounds, mass_axis, masses_dist_max,savepath):
    '''Cacluates the Science Utility Estimate for a given experiment

    Parameters
    ----------
    peak_properties: pd.Dataframe
        peak properties (mass, time, height, volume, start time, end time, ...)

    Returns
    -------
    SUE: float
        Science Utility Estimate
    '''
    logging.info(f'{label}: Calculating SUE.')

    # load SUE weights
    args = yaml.safe_load(open(sue_weights_path, 'r'))
    n_peaks_w = args.get('n_peaks_w')
    n_known_peaks_w = args.get('n_known_peaks_w')
    a_zscore_w = args.get('a_zscore_w')
    # combine all weights
    weights = np.array([n_peaks_w, n_known_peaks_w, a_zscore_w])
    weights /= np.sum(weights) # normalize to 1

    # load SUE max
    n_peaks_max = args.get('n_peaks_max')
    n_known_peaks_max = args.get('n_known_peaks_max')
    a_zscore_max = args.get('a_zscore_max')
    # combine max values
    feature_max = np.array([n_peaks_max, n_known_peaks_max, a_zscore_max])

    # check that we found at least one peak
    if len(peak_properties) == 0: #if we didnt find any peaks
        SUE = 0
    else:

        ## calc features

        # total number of peaks
        n_peak = len(peak_properties)

        # total number of peaks with known masses
        known_peaks_bool = find_known_traces(mass_axis[peak_properties.mass_idx], compounds, masses_dist_max)
        n_known_peaks = np.sum(known_peaks_bool)# count number of known peaks
        # average z-score
        a_zscore = peak_properties.zscore.mean()

        # combine all features
        features = np.array([n_peak, n_known_peaks, a_zscore])

        # transform with non-linear function
        exponent = 0.5
        features = np.abs(features)**exponent  # make sure everything is positive
        feature_max = feature_max**exponent

        # normalize by max feature
        features[features > feature_max] = feature_max[features > feature_max]
        features /= feature_max

        # weight features and calc final SUE
        SUE = np.round(np.sum(features * weights),3)

    logging.info(f'{label}: SUE = {str(SUE)}')

    # output to csv for JEWEL
    SUE_df = pd.DataFrame(data={'SUE': [SUE]}, index=[0])
    SUE_df.to_csv(savepath, index=False, float_format='%.3f')


def diversity_descriptor(label, peak_properties, dd_weights_path, compounds, mass_axis, masses_dist_max, savepath):
    '''Constructs the Diversity Descriptor for a given experiment

    Parameters
    ----------
    peak_properties: pd.Dataframe
        peak properties (mass, time, height, volume, start time, end time, ...)

    Returns
    -------
    Outputs Diversity descriptor to file
    '''

    # if True DD is not normalized and weighted. Output can be used tune weights and feature_max values in post processing
    tuning_mode_IO = False

    logging.info(f'{label}: Constructing Diversity Descriptor.')

    # load Diversity Descriptor weights and max values
    f = yaml.load(open(dd_weights_path, 'r'), Loader=yaml.FullLoader)

    # combine all weights and normalize to 1
    weights = np.array([f['n_peaks_w'],
                        f['n_known_peaks_w'],
                        f['a_zscore_w'],
                        f['s_zscore_w'],
                        f['a_height_w'],
                        f['s_height_w'],
                        f['a_volume_w'],
                        f['s_volume_w'],
                        f['a_width_w'],
                        f['s_width_w'],
                        f['a_background_abs_w'],
                        f['a_background_std_w'],
                        f['a_background_diff_w']])


    # get all maximum values for diversity features
    features_max = np.array([f['n_peaks_max'],
                            f['n_known_peaks_max'],
                            f['a_zscore_max'],
                            f['s_zscore_max'],
                            f['a_height_max'],
                            f['s_height_max'],
                            f['a_volume_max'],
                            f['s_volume_max'],
                            f['a_width_max'],
                            f['s_width_max'],
                            f['a_background_abs_max'],
                            f['a_background_std_max'],
                            f['a_background_diff_max']])

    # calculate features
    # check that we found at least two peak
    if len(peak_properties) < 2:
        d = {'no_peaks_found': [1]}
    else:
        DD = {}
        ## calc features
        DD['n_peaks'] = len(peak_properties)# number of peaks in sample

        # total number of peaks with known masses
        known_peaks_bool = find_known_traces(mass_axis[peak_properties.mass_idx], compounds, masses_dist_max)
        DD['n_known_peaks'] = np.sum(known_peaks_bool)  # count number of known peaks

        # average z-score of peaks
        DD['a_zscore'] = peak_properties.zscore.mean()
        # standard deviation z-score of peaks
        DD['s_zscore'] = peak_properties.zscore.std()
        # average height of peaks
        DD['a_height'] = peak_properties.height.mean()
        # standard deviation height of peaks
        DD['s_height'] = peak_properties.height.std()
        # average volume of peaks
        DD['a_volume'] = peak_properties.volume.mean()
        # standard deviation volume of peaks
        DD['s_volume'] = peak_properties.volume.std()
        # average width of peaks
        DD['a_width'] = peak_properties.peak_base_width.mean()
        # standard width volume of peaks
        DD['s_width'] = peak_properties.peak_base_width.std()

        # average background height
        DD['a_background_abs'] = peak_properties.background_abs.mean()
        # average standard deviation of background
        DD['a_background_std'] = peak_properties.background_std.mean()
        # average offset in left to right background
        DD['a_background_diff'] = peak_properties.background_diff.mean()

        # combine all features
        features = np.array([DD['n_peaks'],
                             DD['n_known_peaks'],
                             DD['a_zscore'],
                             DD['s_zscore'],
                             DD['a_height'],
                             DD['s_height'],
                             DD['a_volume'],
                             DD['s_volume'],
                             DD['a_width'],
                             DD['s_width'],
                             DD['a_background_abs'],
                             DD['a_background_std'],
                             DD['a_background_diff']])

        if not tuning_mode_IO:
            # normalize by feature_max
            features[features > features_max] = features_max[features > features_max]
            features /= features_max

            # weight features
            features *= weights

        # make dictionary for pandas
        d = {}
        for i in range(len(features)):
            d[list(DD.keys())[i]] = features[i]

    # output to csv for JEWEL
    DD_df = pd.DataFrame(data=d, index=[0])
    DD_df.to_csv(savepath, index=False, float_format='%.3f')


