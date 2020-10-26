'''
Command line interface to the ACME evaluation tool
'''
import sys
sys.path.append("../")

import numpy as np
import pandas as pd
import glob
import argparse
import matplotlib.pyplot as plt
import os

import logging

from utils import logger

logger.setup_logger(os.path.basename(__file__).rstrip(".py"), "output")
logger = logging.getLogger(__name__)


def get_precision_hand(output_peaks, label_peaks, mass_threshold, time_threshold, zscore_threshold, FP_df, TP_df):
    '''calculate true positive and false positive from hand labels

    Parameters
    ----------
    output_peaks: DataFrame
        output from analyser with mass, time, and time_idx columns of peaks
    label_peaks: DataFrame
        labels with mass, time, and time_idx columns of peaks
    mass_threshold: float
        Max difference in mass [amu] to be considered same peak
    time_threshold: float
        Max difference in time [time index or Min] to be considered same peak
    zscore_threshold: float
        Minimum z-score to be considered for calculation
    FP_df: DataFrame
        found false positive peaks
    TP_df: DataFrame
        found true positive peaks

    Returns
    -------
    n_TP: int
        Number of True Positive
    n_all: int
        Number of all peaks that are bigger than zscore_threshold
    FP_df: DataFrame
        found false positive peaks
    TP_df: DataFrame
        found true positive peaks
    '''
    n_all = 0     # initialize number of peaks considered

    # take first peak from outputs
    for i in range(len(output_peaks)):
        o = output_peaks.iloc[i]
        if o.zscore > zscore_threshold:
            n_all += 1
            # compare whether any of the labeld peaks match the labeled peaks
            peak_found = False
            for l in label_peaks.itertuples():
                if (np.abs(o.mass_idx - l.mass_idx) < mass_threshold) &\
                        (np.abs(o.time_idx - l.time_idx) < time_threshold):
                    peak_found = True
            if peak_found:
                TP_df = TP_df.append(o)
            else:
                # add peak to FP list
                FP_df = FP_df.append(o)

    n_TP = len(TP_df)-1
    if n_all != 0:
        logging.info('Precision: ' + str(np.round(n_TP / n_all,3)))

    return n_TP, n_all, FP_df, TP_df

def get_recall_hand(output_peaks, label_peaks, mass_threshold, time_threshold, zscore_threshold, FN_df):
    '''calculate recall from hand labels

    Parameters
    ----------
    output_peaks: DataFrame
        output from analyser with mass, time, and time_idx columns of peaks
    label_peaks: DataFraem
        labels with mass, time, and time_idx columns of peaks
    mass_threshold: float
        Max difference in mass [amu] to be considered same peak
    time_threshold: float
        Max difference in time [time index or Min] to be considered same peak
    zscore_threshold: float
        Minimum z-score to be considered for calculation

    Returns
    -------
    n_all: int
        Total number of Peaks that are not ambigious or below zscore_threshold
    n_TP: int
        Number of True Positive
    FN_df: DataFrame
        found False Negative peaks

    '''
    n_TP = 0        # number of TP peaks
    n_all = 0       # total number of peaks in labels

    # take first labeled peak
    for i in range(len(label_peaks)):
        l = label_peaks.iloc[i]
        # check that labeled peak is non-ambigeous and high enough
        if (l.zscore > zscore_threshold) & (l.ambigious_flag == 0):
            n_all += 1
            # compare whether any of the output peaks match the labeled peaks
            peak_found = False
            for o in output_peaks.itertuples():
                if (np.abs(o.mass_idx - l.mass_idx) < mass_threshold) &\
                        (np.abs(o.time_idx - l.time_idx) < time_threshold):
                    peak_found = True
            if peak_found:
                n_TP += 1
            else:
                FN_df = FN_df.append(l)
    if n_all != 0:
        logging.info('Recall: ' + str(np.round(n_TP / n_all,3)))
    return n_TP, n_all, FN_df

def get_TP(output_peaks, label_peaks, mass_threshold, time_threshold, zscore_threshold, TP_df, FN_df):
    '''calculate TP from labels

    Parameters
    ----------
    output_peaks: DataFrame
        output from analyser with mass, time, and time_idx columns of peaks
    label_peaks: DataFraem
        labels with mass, time, and time_idx columns of peaks
    mass_threshold: float
        Max difference in mass [amu] to be considered same peak
    time_threshold: float
        Max difference in time [time index or Min] to be considered same peak
    zscore_threshold: float
        Minimum z-score to be considered for calculation
    TP_df: DataFrame
        found true positive peaks
    FN_df: DataFrame
        found false negative peaks

    Returns
    -------
    n_all_label: int
        Total number of labels for peaks with z-score above zscore_threshold
    n_all_output: int
        Total number of analyser-foud peaks with z-score above zscore_threshold
    n_TP: int
        Number of True Positive
    TP_df: DataFrame
        found true positive peaks
    FN_df: DataFrame
        found false negative peaks
    '''

    n_all_label = 0
    n_all_output = 0

    for o in output_peaks.itertuples():
        if o.zscore > zscore_threshold:
            n_all_output += 1

    for i in range(len(label_peaks)):
        l = label_peaks.iloc[i]
        if l.zscore < zscore_threshold:
            continue
        else:
            n_all_label += 1

        peak_found = False
        for o in output_peaks.itertuples():
            if (np.abs(o.mass - l.mass) < mass_threshold) & (np.abs(o.time - l.time) < time_threshold):
                peak_found = True
        if peak_found:
            TP_df = TP_df.append(l)
        else:
            FN_df = FN_df.append(l)

    n_TP = len(TP_df)-1
    # calculate percentage of found peaks
    if n_all_label != 0:
        TP = n_TP / n_all_label
        logging.info('Recall: ' + str(np.round(TP,3)))
    if n_all_output != 0:
        TP = n_TP / n_all_output
        logging.info('Precision: ' + str(np.round(TP,3)))

    return n_TP, n_all_label, n_all_output, TP_df, FN_df

def get_performance(**kwargs):
    '''main program to compare 1D/2D results to labels from simulator'''

    # unpack parameters
    analyser_outputs = kwargs.get('analyser_outputs')
    path_labels = kwargs.get('path_labels')
    time_threshold = kwargs.get('time_threshold')
    mass_threshold = kwargs.get('mass_threshold')
    hand_label = kwargs.get('hand_labels')

    if kwargs['zscore']:
        zscore_threshold = kwargs.get('zscore_threshold')
    else:
        zscore_threshold = 5

    # convert mass_idx to amu and time_idx min if we are not working with hand labels
    if not hand_label:
        mass_idx_to_amu = 0.0833    # make sure that it agrees with ACME simulator settings (~L 295)
        time_idx_to_min = 0.0061    # make sure that it agrees with ACME simulator settings (~L 296)
        # convert time threshold to units of Min
        time_threshold *= time_idx_to_min  # varies between 0.005 to 0.0065
        # convert mass threshold to units of amu
        mass_threshold *= mass_idx_to_amu

    logging.info('Evaluating ' + analyser_outputs + '\n')

    # find outputs from ACME analyser
    outputs = glob.glob(analyser_outputs, recursive=True)
    output_names = []
    outputs_filtered = []
    for o in outputs:
        if 'heatmap' in o: #filter 1D hetmaps csv files
            continue
        outputs_filtered.append(o)
        o_name = o.split('/')[-1]
        o_name = o_name.rstrip('_peaks.csv')
        output_names.append(o_name)
    outputs = outputs_filtered

    if len(output_names) == 0:
        logging.info('Error: No outputs from ACME analyser found to process in directory ', analyser_outputs)
        exit()
    else:
        logging.info('Found ' + str(len(output_names)) + ' ACME outputs to process')

    # find labels
    labels = glob.glob(path_labels, recursive=True)
    label_names = []
    for l in labels:
        l_name = l.rstrip('_label.csv')
        l_name = l_name.split('/')[-1]
        label_names.append(l_name)

    if len(label_names) == 0:
        logging.info('Error: No ACME labels found to process in directory ', path_labels)
        exit()
    else:
        logging.info('Found ' + str(len(label_names)) + ' ACME labels')

    # compare
    n_TP_prec = 0
    n_all_prec = 0
    n_TP_rec = 0            # intitialize number of TP peaks
    n_all_rec = 0  # intitialize number of TP ambigeous peaks
    n_TP = 0
    n_comparisons = 0 # initialize number of comparisons

    # iterate over output files
    for i in range(len(outputs)):
        output_peaks = pd.read_csv(outputs[i])
        # rename output peaks
        output_peaks.rename(columns={'Mass (amu)': 'mass','Mass (idx)': 'mass_idx', 'Peak Central Time (Min)': 'time','Peak Central Time (idx)': 'time_idx', 'Peak Amplitude (ZScore)':'zscore'}, inplace=True)

        # iterate over labels to find one where the name matches our output file
        for j in range(len(label_names)):
            if output_names[i] == label_names[j]:
                n_comparisons += 1
                logging.info('Evaluating: ' + output_names[i])
                # read label
                label_peaks = pd.read_csv(labels[j])

                # rename labels
                if hand_label:
                    label_peaks.rename(columns={ 'Peak Amplitude (ZScore)': 'zscore'}, inplace=True)
                else:
                    label_peaks.rename(columns={'Mass (amu)': 'mass', 'Peak Central Time (Min)': 'time', 'Z-score': 'zscore'}, inplace=True)

                # initialize dataframe of TP FP FN peaks
                # there is probably a more elegant solution for this
                TP_df = output_peaks.iloc[0:1] * np.nan
                FP_df = output_peaks.iloc[0:1] * np.nan
                FN_df = label_peaks.iloc[0:1] * np.nan


                # calc TP from all labels
                if hand_label:
                    n_TP_prec_i, n_all_prec_i, FP_df, TP_df = get_precision_hand(output_peaks, label_peaks, mass_threshold, time_threshold, zscore_threshold, TP_df, FP_df)
                    n_TP_prec += n_TP_prec_i
                    n_all_prec += n_all_prec_i

                    n_TP_rec_i, n_all_rec_i, FN_df = get_recall_hand(output_peaks, label_peaks, mass_threshold, time_threshold, zscore_threshold, FN_df)
                    n_TP_rec += n_TP_rec_i
                    n_all_rec += n_all_rec_i
                else:
                    n_TP_i, n_all_rec_i, n_all_prec_i, TP_df, FN_df = get_TP(output_peaks, label_peaks, mass_threshold, time_threshold, zscore_threshold, TP_df, FN_df)
                    n_TP += n_TP_i
                    n_all_rec += n_all_rec_i
                    n_all_prec += n_all_prec_i

                # write peaks to csv that are FP, FN or TP
                path_name = os.path.dirname(outputs[i]) + '/eval/'
                if not os.path.exists(path_name):
                    os.makedirs(path_name)

                if hand_label:
                    FP_df.dropna(inplace=True)
                    FP_df.to_csv(path_name + output_names[i] + '_FP.csv', sep=',')
                TP_df.dropna(inplace=True); FN_df.dropna(inplace=True)
                TP_df.to_csv(path_name + output_names[i] + '_TP.csv', sep=',')
                FN_df.to_csv(path_name + output_names[i] + '_FN.csv', sep=',')

    if hand_label:
        Precision = np.round(n_TP_prec / n_all_prec,2)
        Recall = np.round(n_TP_rec / n_all_rec, 2)
        FP = int((n_all_prec - n_TP_prec) / n_comparisons)  # calculate number of FP per experiment
    else:
        Precision = np.round(n_TP / n_all_prec,2)
        Recall = np.round(n_TP / n_all_rec,2)
        FP = int((n_all_prec - n_TP) / n_comparisons)  # calculate number of FP per experiment
    F1 = np.round(2 * (Precision*Recall) / (Precision + Recall),2)



    logging.info('\nPrecission: ' + str(Precision) + '\nRecall: ' + str(Recall) + '\nF1: ' + str(F1) + '\nFP per dataset: ' + str(FP))

    return Precision, Recall, F1, FP

def main(**kwargs):
    '''Evaluates ACME performance with different z-score thresholds'''

    analyser_outputs = kwargs.get('analyser_outputs')

    if kwargs['zscore']:
        z_min = 5
        z_max = 20 #sqrt of z_max
        zscore_thresholds = np.arange(z_min, z_max + 1, 2)

        Precision = []
        Recall = []
        F1 = []
        FP = []

        # evaluate performance for different z-scores
        for z in zscore_thresholds:
            kwargs['zscore_threshold'] = z
            Precision_z, Recall_z, F1_z, FP_z = get_performance(**kwargs)

            Precision.append(Precision_z)
            Recall.append(Recall_z)
            F1.append(F1_z)
            FP.append(FP_z/100)

        ##plot result as a function of z-score
        plt.close('all')
        plt.figure(figsize=(10,6))
        plt.plot(zscore_thresholds, Precision,'r--', label='Precision')
        plt.plot(zscore_thresholds, Recall, 'bs-', label='Recall')
        plt.plot(zscore_thresholds, F1, 'k.-', label='F1')
        plt.plot(zscore_thresholds, FP, 'g*-', label='FP/dataset/100')
        if kwargs['hand_labels']:
            plt.ylim((0, 1.1))
        else:
            plt.ylim((0,1))
        plt.xlim((z_min, z_max))
        plt.title(analyser_outputs)
        plt.xlabel('Minimum Z-score considered [-]', fontsize=20)
        plt.ylabel('Performance [-]', fontsize=20)
        plt.legend( fontsize=20)
        #save figure
        save_path = analyser_outputs.split('*')[0]
        save_path_plot = save_path + 'Performance.png'
        plt.savefig(save_path_plot, dpi=200)
        plt.show()

        ##output results as csv
        performance = {'Z-score threshold': zscore_thresholds, 'Precission': Precision, 'Recall': Recall, 'F1':F1}
        df = pd.DataFrame(data=performance)
        save_path_csv = save_path + 'Performance.csv'
        df.to_csv(save_path_csv)

    else:
        Precision, Recall, F1, FP = get_performance(**kwargs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--analyser_outputs',
                        default=None,
                        help='Found peaks from analyzer -- Passed as globs')

    parser.add_argument('--path_labels',
                        default=None,
                        help='Labels to compare found peaks to -- Passed as globs')

    parser.add_argument('--hand_labels', action='store_true',
                        help='Expects hand labels in --path_labels')

    parser.add_argument('--zscore', action='store_true',
                        help='Will evaluate performance for different z-score thresholds. Currently not supported for hand labels')

    parser.add_argument('--mass_threshold', default=30,
                        help='How far can peaks be apart from each other in mass [mass index] to be considered the same peak'
                             '12 mass index correspond to 1 amu')

    parser.add_argument('--time_threshold', default=30,
                        help='How far can peaks be apart from each other in time [time index] to be considered the same peak'
                        '164 time index correspond to 1 Min')

    args = parser.parse_args()

    if args.analyser_outputs is None:
        logging.error('Error: Please specify directory with output from ACME analyser')
        print('e.g.:  --analyser_outputs .../Silver_Dataset_V1/**/*_peaks.csv')
        exit()

    if args.path_labels is None:
        logging.error('Error: Please specify directory with labels from ACME simulator')
        print('e.g.:  --path_labels .../Silver_Dataset_V1/**/*_label.csv')
        exit()

    if args.zscore is True & args.hand_labels is True:
        logging.warning('Uncertainties in the z-score of the labels can lead to precision / Recall > 1')

    main(**vars(args))
    logging.info("======= Done =======")
