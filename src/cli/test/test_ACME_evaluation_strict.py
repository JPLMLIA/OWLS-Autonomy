import pandas as pd
import pytest

from cli.ACME_evaluation_strict import *


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

@pytest.mark.skip
def test_get_precision_hand():
    # one label per peak
    output_peaks = pd.DataFrame(data={'mass_idx': [1, 10, 100], 'time_idx': [1, 10, 100], 'zscore': [1, 10, 100]})
    label_peaks = pd.DataFrame(data={'mass_idx': [1, 10, 100], 'time_idx': [1, 10, 100], 'zscore': [1, 10, 100]})
    mass_threshold = 30
    time_threshold = 30
    zscore_threshold = 5
    FP_df = output_peaks.iloc[0:1] * np.nan
    TP_df = output_peaks.iloc[0:1] * np.nan
    TP_df_expected = TP_df.append(output_peaks.iloc[1:3])

    n_TP, n_all, FP_df, TP_df = get_precision_hand(output_peaks, label_peaks, mass_threshold,time_threshold, zscore_threshold, FP_df, TP_df)

    assert n_all == 2
    assert n_TP == 2
    pd.testing.assert_frame_equal(TP_df, TP_df_expected)

    # multiple peaks in outputs match one peak in label
    output_peaks = pd.DataFrame(data={'mass_idx': [1, 10,11, 100], 'time_idx': [1, 10,11, 100], 'zscore': [1, 10,11, 100]})
    FP_df = output_peaks.iloc[0:1] * np.nan
    TP_df = output_peaks.iloc[0:1] * np.nan

    n_TP, n_all, FP_df, TP_df = get_precision_hand(output_peaks, label_peaks, mass_threshold, time_threshold,
                                                   zscore_threshold, FP_df, TP_df)

    assert n_all == 3
    assert n_TP == 3


    # multiple peaks in label match one peak in output
    output_peaks = pd.DataFrame(data={'mass_idx': [1, 10, 100], 'time_idx': [1, 10, 100], 'zscore': [1, 10, 100]})
    label_peaks = pd.DataFrame(data={'mass_idx': [1, 10,11, 100], 'time_idx': [1, 10,11, 100], 'zscore': [1, 10,11, 100]})
    FP_df = output_peaks.iloc[0:1] * np.nan
    TP_df = output_peaks.iloc[0:1] * np.nan

    n_TP, n_all, FP_df, TP_df = get_precision_hand(output_peaks, label_peaks, mass_threshold, time_threshold,
                                                   zscore_threshold, FP_df, TP_df)

    assert n_all == 2
    assert n_TP == 2

    # found peak is close to labeled peak and far away
    output_peaks = pd.DataFrame(data={'mass_idx': [1, 10, 100], 'time_idx': [1, 10, 100], 'zscore': [1, 10, 100]})
    label_peaks = pd.DataFrame(data={'mass_idx': [1, 20, 200], 'time_idx': [1, 20, 200], 'zscore': [1, 10, 100]})
    FP_df = output_peaks.iloc[0:1] * np.nan
    TP_df = output_peaks.iloc[0:1] * np.nan

    n_TP, n_all, FP_df, TP_df = get_precision_hand(output_peaks, label_peaks, mass_threshold, time_threshold,
                                                   zscore_threshold, FP_df, TP_df)

    assert n_all == 2
    assert n_TP == 1

@pytest.mark.skip
def test_get_recall_hand():
    # one label per peak
    output_peaks = pd.DataFrame(data={'mass_idx': [1, 10, 100], 'time_idx': [1, 10, 100], 'zscore': [1, 10, 100]})
    label_peaks = pd.DataFrame(data={'mass_idx': [1, 10, 100], 'time_idx': [1, 10, 100], 'zscore': [1, 10, 100], 'ambigious_flag':[0, 0, 0]})
    mass_threshold = 30
    time_threshold = 30
    zscore_threshold = 5
    FN_df = output_peaks.iloc[0:1] * np.nan

    n_TP, n_all, FN_df = get_recall_hand(output_peaks, label_peaks, mass_threshold,time_threshold, zscore_threshold, FN_df)

    assert n_all == 2
    assert n_TP == 2


    # multiple peaks in outputs match one peak in label
    output_peaks = pd.DataFrame(data={'mass_idx': [1, 10,11, 100], 'time_idx': [1, 10,11, 100], 'zscore': [1, 10,11, 100]})
    FN_df = output_peaks.iloc[0:1] * np.nan

    n_TP, n_all, FN_df = get_recall_hand(output_peaks, label_peaks, mass_threshold, time_threshold,
                                                   zscore_threshold, FN_df)

    assert n_all == 2
    assert n_TP == 2


    # output finds ambigious peak
    output_peaks = pd.DataFrame(data={'mass_idx': [1, 10, 100], 'time_idx': [1, 10, 100], 'zscore': [1, 10, 100]})
    label_peaks = pd.DataFrame(data={'mass_idx': [1, 10, 100], 'time_idx': [1, 10, 100], 'zscore': [1, 10, 100], 'ambigious_flag':[1, 1, 1]})
    FN_df = output_peaks.iloc[0:1] * np.nan

    n_TP, n_all, FN_df = get_recall_hand(output_peaks, label_peaks, mass_threshold, time_threshold,
                                                   zscore_threshold, FN_df)

    assert n_all == 0
    assert n_TP == 0

    # output misses ambigious peak
    output_peaks = pd.DataFrame(data={'mass_idx': [1, 10, 100], 'time_idx': [1, 10, 100], 'zscore': [1, 10, 100]})
    label_peaks = pd.DataFrame(data={'mass_idx': [1, 10, 100], 'time_idx': [500, 600, 700], 'zscore': [1, 10, 100], 'ambigious_flag':[1, 1, 1]})
    FN_df = output_peaks.iloc[0:1] * np.nan

    n_TP, n_all, FN_df = get_recall_hand(output_peaks, label_peaks, mass_threshold, time_threshold,
                                                   zscore_threshold, FN_df)

    assert n_all == 0
    assert n_TP == 0

@pytest.mark.skip
def test_get_TP():

    output_peaks = pd.DataFrame(data={'mass': [1, 10, 100],'time': [1, 2, 10], 'time_idx': [1, 10, 100], 'zscore': [1,10,10]})
    label_peaks = pd.DataFrame(data={'mass': [1, 10, 100],'time': [1, 2, 10], 'time_idx': [1, 10, 100], 'zscore': [1,10,10]})
    mass_threshold = 1
    time_threshold = 1
    zscore_threshold = 5
    FN_df = output_peaks.iloc[0:1] * np.nan
    TP_df = output_peaks.iloc[0:1] * np.nan

    n_TP, n_all_label, n_all_output, TP_df, FN_df = get_TP(output_peaks, label_peaks, mass_threshold, time_threshold,
                                                           zscore_threshold, TP_df, FN_df)

    assert n_TP == 2
    assert n_all_output == 2
    assert n_all_label == 2