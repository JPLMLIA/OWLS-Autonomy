
'''
Command line interface to the ACME evaluation tool

The way precision and recall is calculated is not straightforward.
This is because the detector may detect multiple peaks that match to a single
labeled peak.

In traditional object detection, any additional detections past the first is 
considered a false positive. In our case (for now), we only care about detections
that aren't near any peaks at all.

Therefore, there is an OUTPUT PRECISION and LABEL RECALL.

output precision:
    Of all the predicted peaks, how many are matched to labels?
label recall:
    Of all the labeled peaks, how many are matched to predictions?

The F1 score does not make sense under these circumstances since the precision
and recall are calculated on different components of the data.
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

def calc_tp(o, l, mass_t=30, time_t=30):
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
                seen_labels.add(lpeak_idx)

    ltp = len(seen_labels)
    return otp, ltp

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('acme_outputs',         help='Found peaks from analyzer -- Passed as globs')

    parser.add_argument('acme_labels',          help='Labels to compare found peaks to -- Passed as globs')

    parser.add_argument('--hand_labels',        action='store_true',
                                                help='Expects hand labels in --path_labels')

    parser.add_argument('--mass_threshold',     default=30,
                                                help='How far can peaks be apart from each other in mass [mass index] '
                                                     'to be considered the same peak 12 mass index correspond to 1 amu')
    
    parser.add_argument('--time_threshold',     default=30,
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
        logging.warning(f"No ACME outputs found at {args.acme_outputs}")

    labels = sorted(glob.glob(args.acme_labels))
    if not len(labels):
        logging.warning(f"No ACME labels found at {args.acme_labels}")

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
    for out_f, label_f, exp in out_label_pairs:
        output_peaks = []
        with open(out_f, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                output_peaks.append([float(row['Mass (idx)']), float(row['Peak Central Time (idx)']), float(row['Peak Amplitude (ZScore)'])])
        
        label_peaks = []
        with open(label_f, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                if hand_labels:
                    if ambiguous or not float(row['ambigious_flag']):
                        # only count peak if not ambiguous or flagged
                        label_peaks.append([float(row['mass_idx']), float(row['time_idx']), float(row['Peak Amplitude (ZScore)'])])
                else:
                    label_peaks.append([float(row['mass_idx']), float(row['time_idx']), float(row['Z-Score'])])

        output_peaks = np.array(output_peaks).astype(np.float)
        label_peaks = np.array(label_peaks).astype(np.float)

        exp_label_peaks.append((output_peaks, label_peaks, exp))

    ## Sweep across z-scores
    zscores = list(range(5,16))
    output_array = []
    output_verbose_array = []
    for z in tqdm(zscores, desc='z-scores'):
        # Global statistics per Z-Score
        z_otp = 0
        z_oshape = 0
        z_ltp = 0
        z_lshape = 0

        # Per-experiment
        for o, l, exp in exp_label_peaks:
            o = o[o[:,2]>=z]
            l = l[l[:,2]>=z]
            # calculate output true positive and label true positive
            otp, ltp = calc_tp(o[:,:2], l[:,:2], mass_t, time_t)
            # output precision
            precision = otp / o.shape[0]
            # label recall
            recall = ltp / l.shape[0]
            # save
            output_verbose_array.append([z, exp, otp, o.shape[0], precision, ltp, l.shape[0], recall, (o.shape[0]-otp)])
            
            z_otp += otp
            z_oshape += o.shape[0]
            z_ltp += ltp
            z_lshape += l.shape[0]

        # Global precision and recall
        z_precision = z_otp / z_oshape
        z_recall = z_ltp / z_lshape

        output_array.append([z, z_precision, z_recall, (z_oshape-z_otp)/len(exp_label_peaks)])
    
    output_array = np.array(output_array)
    output_verbose_array = np.array(output_verbose_array)

    ## Plotting

    fig, ax = plt.subplots()
    ax.plot(zscores, output_array[:,1], 'r^--', label='Pred Precision')
    ax.plot(zscores, output_array[:,2], 'bs-', label='Label Recall')
    ax.set_ylim(0, 1)
    ax.set_xlim(min(zscores), max(zscores))
    ax.set_xlabel('Minimum Z-score Considered')
    ax.set_ylabel('Performance')
    ax.set_title(args.acme_outputs, fontsize=8)
    plt.grid(axis='both')

    ax2 = ax.twinx()
    ax2.plot(zscores, output_array[:,3], 'g*--', label='Average FPs')
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel('Number of FP peaks')
    ax2.tick_params(axis='y', labelcolor='g')

    ax.legend(loc='lower left')
    ax2.legend(loc='lower right')
    plt.tight_layout()
    logging.info('Saving acme_eval.png')
    fig.savefig(op.join(args.log_folder,'acme_eval.png'), dpi=400)

    ## CSV Output

    logging.info('Saving acme_eval.csv')
    with open(op.join(args.log_folder,'acme_eval.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['z-score', 'pred precision', 'label recall', 'mean FP'])
        writer.writerows(output_array)
    
    logging.info('Saving acme_eval.csv')
    with open(op.join(args.log_folder,'acme_eval_verbose.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['z-score', 'experiment', 'pred TP', 'pred N', 'pred precision', 'label TP', 'label N', 'label recall', 'pred FP'])
        writer.writerows(output_verbose_array)

if __name__ == "__main__":
    main()