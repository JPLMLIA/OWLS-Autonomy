# functions that generate input for JEWEL
# SUE and Diversity Descriptor
# these are called from analyzer.py
import yaml
import logging

import numpy  as np
import pandas as pd

from fsw.ACME.lib.utils import find_known_traces

def calc_SUE(label, peak_properties, sue_weights_path, compounds, mass_axis, mass_res, time_axis, time_res, masses_dist_max, savepath):
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

    # load SUE weights
    args = yaml.safe_load(open(sue_weights_path, 'r'))
    n_peaks_w = args.get('n_peaks_w')
    n_known_peaks_w = args.get('n_known_peaks_w')
    a_zscore_w = args.get('a_zscore_w')
    n_unique_times_w = args.get('n_unique_times_w')
    n_unique_masses_w = args.get('n_unique_masses_w')
    # combine all weights
    weights = np.array([n_peaks_w, n_known_peaks_w, a_zscore_w, n_unique_times_w, n_unique_masses_w])
    weights /= np.sum(weights) # normalize to 1

    # load SUE max
    n_peaks_max = args.get('n_peaks_max')
    n_known_peaks_max = args.get('n_known_peaks_max')
    a_zscore_max = args.get('a_zscore_max')
    n_unique_times_max = args.get('n_unique_times_max')
    n_unique_masses_max = args.get('n_unique_masses_max')
    # combine max values
    feature_max = np.array([n_peaks_max, n_known_peaks_max, a_zscore_max, n_unique_times_max, n_unique_masses_max])

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

        # number of unique masses and times
        n_unique_times = np.count_nonzero(np.histogram([time_axis[x] for x in peak_properties.time_idx], bins=np.arange(time_axis[0], time_axis[-1]+time_res, time_res))[0])
        n_unique_masses = np.count_nonzero(np.histogram([mass_axis[x] for x in peak_properties.mass_idx], bins=np.arange(mass_axis[0], mass_axis[-1]+mass_res, mass_res))[0])

        # combine all features
        features = np.array([n_peak, n_known_peaks, a_zscore, n_unique_times, n_unique_masses])

        # transform with non-linear function
        exponent = 0.5
        features = np.abs(features)**exponent  # make sure everything is positive
        feature_max = feature_max**exponent

        # normalize by max feature
        features[features > feature_max] = feature_max[features > feature_max]
        features /= feature_max

        # weight features and calc final SUE
        SUE = np.round(np.sum(features * weights),3)

    logging.info(f'SUE = {str(SUE)}')

    # output to csv for JEWEL
    SUE_df = pd.DataFrame(data={'SUE': [SUE]}, index=[0])
    SUE_df.to_csv(savepath, index=False, float_format='%.3f')


def diversity_descriptor(label, peak_properties, dd_weights_path, compounds, mass_axis, mass_res, mass_dd_res, time_axis, time_res, masses_dist_max, savepath):
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

    # load Diversity Descriptor weights and max values
    f = yaml.load(open(dd_weights_path, 'r'), Loader=yaml.FullLoader)

    # combine all weights and normalize to 1
    weights = np.array([f['a_background_abs_w'],
                        f['a_background_std_w'],
                        f['n_unique_times_w'],
                        f['n_unique_masses_w']])


    # get all maximum values for diversity features
    features_max = np.array([f['a_background_abs_max'],
                             f['a_background_std_max'],
                             f['n_unique_times_max'],
                             f['n_unique_masses_max']])

    # calculate features
    # check that we found at least two peak
    if len(peak_properties) < 2:
        d = {'no_peaks_found': [1]}
    else:
        DD = {}
        ## calc features
        # average background height
        DD['a_background_abs'] = peak_properties.background_abs.mean()
        # average standard deviation of background
        DD['a_background_std'] = peak_properties.background_std.mean()

        # number of unique masses and times
        DD['n_unique_times'] = np.count_nonzero(np.histogram([time_axis[x] for x in peak_properties.time_idx], bins=np.arange(time_axis[0], time_axis[-1]+time_res, time_res))[0])
        DD['n_unique_masses'] = np.count_nonzero(np.histogram([mass_axis[x] for x in peak_properties.mass_idx], bins=np.arange(mass_axis[0], mass_axis[-1]+mass_res, mass_res))[0])

        # combine all features
        features = np.array([DD['a_background_abs'],
                             DD['a_background_std'],
                             DD['n_unique_times'],
                             DD['n_unique_masses']])

        # vector of unique mass bins
        unique_vector, edges = np.histogram([mass_axis[x] for x in peak_properties.mass_idx], bins=np.arange(mass_axis[0], mass_axis[-1]+mass_dd_res, mass_dd_res))
        for i, value in enumerate(unique_vector):
            DD[f"mz_{int(edges[i])}-{int(edges[i+1])}"] = value
            features = np.append(features, value)
            weights = np.append(weights, f['v_unique_masses_w'])
            features_max = np.append(features_max, f['v_unique_masses_max'])

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

