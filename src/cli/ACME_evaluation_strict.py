
'''
Command line interface to the ACME evaluation tool (strict version)

This evaluation script enforces a one-to-one match between labeled and output
peaks, and therefore is able to calculate a consistent precision, recall, and f1.
'''
import sys, os
import os.path as op
import glob
import argparse
import csv
from pathlib import Path
import logging

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import logger

def calc_tp(o, l, mass_t=5, time_t=5):
    otp = 0
    ltp = 0
    seen_labels = set()
    for opeak in o:
        # see which label peaks are within threshold of output peaks
        thresh_mask = (np.abs(l[:,0] - opeak[0]) <= mass_t) & (np.abs(l[:,1] - opeak[1]) <= time_t)
        if np.sum(thresh_mask) >= 1:
            otp += 1
            # output matches to more than one label
            for lpeak_idx in thresh_mask.nonzero()[0]:
                if lpeak_idx not in seen_labels:
                    seen_labels.add(lpeak_idx)

    ltp = len(seen_labels)
    return otp, ltp

def calc_tp_strict(o, l, mass_t=5, time_t=5):
    seen_outputs = set()
    for lpeak in l:
        # see which output peaks are within threshold of labeled peak
        thresh_mask = (np.abs(o[:,0] - lpeak[0]) <= mass_t) & (np.abs(o[:,1] - lpeak[1]) <= time_t)
        thresh_peaks = o[thresh_mask]
        if len(thresh_peaks) > 0:
            thresh_dists = [np.sqrt((lpeak[0]-tpeak[0])**2 + (lpeak[1]-tpeak[1])**2) for tpeak in thresh_peaks]
            closest_peak = thresh_peaks[np.argmin(thresh_dists)]
            seen_outputs.add(tuple(closest_peak))
    return len(seen_outputs)


def load_detected_peaks(acme_detection_file):
    """Helper to load peaks detected by ACME"""

    output_peaks = []
    with open(acme_detection_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            output_peaks.append([row['Mass (idx)'],
                                 row['Peak Central Time (idx)'],
                                 row['Peak Amplitude (ZScore)']])

    return output_peaks


def load_labeled_peaks(peak_labels_file, hand_labels, ambiguous):
    """Helper to load peaks labeled in mass-spec data"""

    label_peaks = []
    with open(peak_labels_file, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            if hand_labels:
                if ambiguous or not float(row['ambiguous_flag']):
                    # only count peak if not ambiguous or flagged
                    label_peaks.append([row['mass_idx'], row['time_idx'], row['Peak Amplitude (ZScore)']])
            else:
                label_peaks.append([row['mass_idx'], row['time_idx'], row['ZScore']])

    return label_peaks


def eval_z_score(z_thresh, label_sets, mass_t, time_t):
    """Helper to evaluate labels at a specific z-score"""

    output_results = []
    output_results_verbose = []
    z_tp = 0
    z_oshape = 0
    z_lshape = 0

    num_detected_peaks = len(label_sets[0][0])
    if num_detected_peaks == 0:
        logging.warning("No peaks found!")
        return output_results, output_results_verbose

    # Per-experiment
    # TODO: convert over to prebuilt precision/recall functions in sklearn.metrics
    #    and take advantage of user's ability to determine what should happen
    #    for undefined values
    for o, l, exp in label_sets:
        o = o[o[:, 2] >= z_thresh]

        # calculate true positives
        tp = calc_tp_strict(o[:, :2], l[:, :2], mass_t, time_t)

        # precision and recall
        if len(o):
            precision = tp / o.shape[0]
        else:
            precision = 0
            logging.warning(f"Labeled set for experiment {exp} did not have any"
                            " labeled peaks meeting z-thresh in precision "
                            "calc. Setting precision to 0")
        recall = tp / l.shape[0]

        # f1
        if (precision + recall) > 0:
            f1 = 2 * ((precision * recall) / (precision + recall))
        else:
            f1 = 0
            logging.warning(f"F1 not computable as (precision+recall) = 0. Setting F1 to 0")

        # save
        output_results_verbose.append([z_thresh, exp, o.shape[0], l.shape[0], tp,
                                       (o.shape[0]-tp), precision, recall, f1])

        z_tp += tp
        z_oshape += o.shape[0]
        z_lshape += l.shape[0]

    # Global precision and recall
    if z_oshape:
        z_precision = z_tp / z_oshape
    else:
        z_precision = 0
        logging.warning(f"Global precision calc for {len(label_sets)} label sets undefined. Setting to 0")

    # Global F1
    if z_lshape > 0:
        z_recall = z_tp / z_lshape
    else:
        z_recall = 0
        logging.warning(f"No labels available across {len(label_sets)} for "
                        "global recall calculation. Setting to 0")

    # Global F1
    if (z_precision + z_recall) > 0:
        z_f1 = 2 * ((z_precision * z_recall) / (z_precision + z_recall))
    else:
        z_f1 = 0
        logging.warning("Global recall and precision are zero, so cannot "
                        "calculate global F1. Setting to 0.")

    output_results.append([z_thresh, z_precision, z_recall, z_f1,
                           (z_oshape-z_tp) / len(label_sets)])

    return output_results, output_results_verbose


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('acme_outputs',         help='Found peaks from ACME analyzer -- Passed as globs (including quotes)')

    parser.add_argument('acme_labels',          help='Labels to compare found peaks to -- Passed as globs (including quotes)')

    parser.add_argument('--hand_labels',        action='store_true',
                                                help='Expects hand labels in --path_labels')

    parser.add_argument('--mass_threshold',     default=5,
                                                help='How far can peaks be apart from each other in mass [mass index] '
                                                     'to be considered the same peak 12 mass index correspond to 1 amu')

    parser.add_argument('--time_threshold',     default=5,
                                                help='How far can peaks be apart from each other in time [time index] '
                                                     'to be considered the same peak 164 time index correspond to 1 Min')

    parser.add_argument('--ambiguous',          action='store_true',
                                                help='Some peaks are labeled as ambiguous by SMEs. Call this flag to include '
                                                     'them as true peak labels.')

    parser.add_argument('--log_name',           default="ACME_evaluation.log",
                                                help="Filename for the pipeline log. Default is ACME_evaluation.log")

    parser.add_argument('--log_folder',         default=op.join(op.abspath(op.dirname(__file__)), "logs"),
                                                help="Folder path to store logs. Default is cli/logs")

    args = parser.parse_args()

    logger.setup_logger(args.log_name, args.log_folder)

    # parse args
    outputs = sorted(glob.glob(args.acme_outputs))
    if not len(outputs):
        logging.error(f"No ACME outputs found at {args.acme_outputs}")
        sys.exit(1)

    labels = sorted(glob.glob(args.acme_labels))
    if not len(labels):
        logging.error(f"No ACME labels found at {args.acme_labels}")
        sys.exit(1)

    if len(outputs) != len(labels):
        logging.warning(f"{len(outputs)} outputs but {len(labels)} labels")

    mass_t = float(args.mass_threshold)
    time_t = float(args.time_threshold)
    ambiguous = args.ambiguous
    hand_labels = args.hand_labels

    # pair up outputs and labels
    out_label_pairs = []
    label_stems = [Path(p).stem for p in labels]
    for output in outputs:
        # find if output has corresponding label
        exp = Path(output).stem.split('_UM_peaks')[0]
        if exp+"_label" in label_stems:
            label_idx = label_stems.index(exp+"_label")
            out_label_pairs.append((output, labels[label_idx], exp))
        else:
            logging.warning(f"Label not found for output {output}")

    # read and store peaks
    exp_label_peaks = []
    for detected_f, label_f, exp in out_label_pairs:
        # Load detected/labeled peaks
        output_peaks = np.array(load_detected_peaks(detected_f)).astype(np.float)
        label_peaks = np.array(load_labeled_peaks(label_f, hand_labels, ambiguous)).astype(np.float)

        exp_label_peaks.append((output_peaks, label_peaks, exp))

    ## Absolute performance
    abs_output, abs_output_verbose = eval_z_score(0, exp_label_peaks, mass_t, time_t)

    ## Sweep across z-scores
    zscores = list(range(5,16))
    output_array = []
    output_verbose_array = []
    for z in tqdm(zscores, desc='z-scores'):
        temp_output, temp_output_verbose = eval_z_score(z, exp_label_peaks,
                                                        mass_t, time_t)
        output_array.extend(temp_output)
        output_verbose_array.extend(temp_output_verbose)

    output_array = np.array(output_array)
    output_verbose_array = np.array(output_verbose_array)

    ## Plotting (only if peaks are found)
    if len(output_array) > 0:
        fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(zscores, output_array[:,1], 'r^--', label='Precision')
        ax.plot(zscores, output_array[:,2], 'bs-', label='Recall')
        #ax.plot(zscores, output_array[:,3], 'md-.', label='F1')
        ax.set_ylim(0, 1)
        ax.set_xlim(min(zscores), max(zscores))
        ax.set_xlabel('Minimum Z-score Considered')
        ax.set_ylabel('Performance')
        #ax.set_title(args.acme_outputs, fontsize=8)
        plt.grid(axis='both')

        ax2 = ax.twinx()
        ax2.plot(zscores, output_array[:,4], 'g*--', label='Average FPs')
        ax2.set_ylim(bottom=0)
        ax2.set_ylabel('Number of FP peaks')
        ax2.tick_params(axis='y', labelcolor='g')

        ax.legend(loc='lower left')
        ax2.legend(loc='lower right')
        plt.tight_layout()
        logging.info('Saving acme_eval_strict.png')
        fig.savefig(op.join(args.log_folder,'acme_eval_strict.png'), dpi=150)

    ## CSV Output

    logging.info('Saving acme_eval.csv')
    with open(op.join(args.log_folder,'acme_eval.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['z-score', 'precision', 'recall', 'f1', 'mean FP'])
        if len(output_array) > 0:
            writer.writerow(abs_output[0])

    logging.info('Saving acme_eval_verbose.csv')
    with open(op.join(args.log_folder,'acme_eval_verbose.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['z-score', 'experiment', 'pred N', 'label N', 'true positive', 'false positive', 'precision', 'recall', 'f1'])
        if len(output_array) > 0:
            writer.writerows(abs_output_verbose)

    logging.info('Saving acme_eval_sweep.csv')
    with open(op.join(args.log_folder,'acme_eval_sweep.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['z-score', 'precision', 'recall', 'f1', 'mean FP'])
        if len(output_array) > 0:
            writer.writerows(output_array)

    logging.info('Saving acme_eval_sweep_verbose.csv')
    with open(op.join(args.log_folder,'acme_eval_sweep_verbose.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['z-score', 'experiment', 'pred N', 'label N', 'true positive', 'false positive', 'precision', 'recall', 'f1'])
        if len(output_array) > 0:
            writer.writerows(output_verbose_array)

if __name__ == "__main__":
    main()
