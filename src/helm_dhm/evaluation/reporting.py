"""
Collection of tools for plotting and reporting performance
"""
import statistics
import json
from pathlib import Path
import os.path as op

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, fbeta_score
import matplotlib.pyplot as plt
import logging
from collections import OrderedDict

def point_score_report(y_true, y_pred, json_save_fpath=None):
    """Calculate a set of standard metrics from binary detection results.

    Parameters
    ----------
    y_true: np.ndarray
        Binary array containing the ground truth binary labels.
    y_pred: np.ndarray
        Binary array matching shape and order of `y_true`. For matching track
        locations, each entry represents whether or not a single point correctly
        matched the corresponding entry in `y_true`.
    json_save_fpath: str or None
        If provided, will attempt to save to save results as a json to this
        location. String should use the `.json` extension.

    Returns
    -------
    score_dict: dict
        Dictionary containing multiple score indices.
    """

    # Simple error checks on input
    if y_true.shape != y_pred.shape:
        raise ValueError('Shape of y_true and y_pred must match.')
    if json_save_fpath and Path(json_save_fpath).suffix != '.json':
        raise ValueError('If providing `json_save_fpath`, it must end in `.json`.')

    # Calculate relevant scores
    score_dict = {}
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred,
                                                               average='binary')
    score_dict['precision'] = precision
    score_dict['recall'] = recall
    score_dict['f_1'] = f1

    score_dict['f_0.5'] = fbeta_score(y_true, y_pred, beta=0.5)
    score_dict['f_0.25'] = fbeta_score(y_true, y_pred, beta=0.25)

    # Save to json if desired
    if json_save_fpath:
        with open(json_save_fpath, 'w') as json_file:
            json.dump(score_dict, json_file)

    return score_dict


def extended_point_report(y_pred, track_numbers, json_save_fpath=None):
    '''Computes per track recall'''
    track_dict = {}
    c_track_num = track_numbers[0]
    start = 0
    matched = 0.0
    for i, t in enumerate(track_numbers):
        if t != c_track_num:
            # End of track: compute recall
            total = i - start
            track_dict[c_track_num] = {'recall': matched / total, 
                                       'track_size': total}
            # Reset for next track
            matched = 0.0
            start = i
            c_track_num = t     
        # Sum matches for this track
        if y_pred[i]:
            matched += 1 

    # Save to json if desired
    if json_save_fpath:
        with open(json_save_fpath, 'w') as json_file:
            json.dump(track_dict, json_file, indent=4)

    return track_dict

def plot_labels(true_track_list, exp_name, plot_output_directory,
                win_size=(1024, 1024)):
    """Plot traces for all tracks on a dark background

    Parameters
    ----------
    track_fpaths: list of str
        Full filepaths to each track to be plotted
    exp_name: str
        Experiment name
    plot_output_directory: str
        Directory for saving the track plot
    win_size: iterable
        Number of pixels in row and column dimensions, respectively.
    """

    # Create plot and use dark background
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot each track
    for label in true_track_list:
        plt.plot(label[:, 1], label[:, 0])

    # Set up title and axis labels
    ax.set_title('Particle tracks labeled in experiment\n' + exp_name)
    ax.invert_yaxis()
    ax.axis('equal')  # Force a square axis
    ax.set_xlim(0, win_size[1])
    ax.set_ylim(win_size[0], 0)

    plt.savefig(op.join(plot_output_directory, exp_name + "_label_plots.png"),
                dpi=150)
    plt.close()

def track_score_report(matches, true_track_list, pred_track_list, n_pred_tracks,
                       coverage_thresh, json_save_fpath, n_frames):
    """Save accuracy stats for a track report.

    Parameters
    ----------
    matches: dict
        Keys: matches of form (true index, pred index)
        Values: number of matched points
    true_track_list: list
        Labeled track points
    pred_track_list: list
        Predicted track points
    n_true_tracks: int
        Number of labeled tracks
    n_pred_tracks: int
        Number of predicted tracks
    json_save_fpath: str or None
        If provided, will attempt to save to save results as a json to this
        location. String should use the `.json` extension.
    n_frames: int
        Number of experiment frames
    """

    num_matches = len(matches.keys())
    n_true_tracks = len(true_track_list)

    # Calculate the number of predicted tracks that had a match
    if num_matches == 0:
        n_matched_true_tracks = 0
        n_matched_pred_tracks = 0
    else:
        matched_pred_tracks = np.unique([m[1] for m in matches.keys()])
        n_matched_pred_tracks = len(matched_pred_tracks)

    if n_matched_pred_tracks > n_pred_tracks:
        logging.warning("More matched pred tracks than actual pred tracks")

    # Calculate the number of true tracks with sufficient coverage
    summed_matches = [0] * n_true_tracks
    for (match, n) in matches.items():
        ti = match[0]
        summed_matches[ti] += n

    matched_true_tracks = [i for i in range(n_true_tracks) 
                           if summed_matches[i] / len(true_track_list[i]) > coverage_thresh]
    n_matched_true_tracks = len(matched_true_tracks)

    # Calculate the number of points in unmatched predicted tracks

    # Get indices of unmatched predicted tracks
    unmatched_pred = set(range(n_pred_tracks))
    for (match, n) in matches.items():
        unmatched_pred.remove(match[1])
    
    # Get number of points in these unmatched predicted tracks
    unmatched_points = 0
    for pi in unmatched_pred:
        unmatched_points += len(pred_track_list[pi])
    
    # False Track Points per Frame
    FTPF = float(unmatched_points) / n_frames

    if n_matched_true_tracks > n_true_tracks:
        logging.warning("More matched true tracks than actual true tracks")

    # Store the proportion of true/pred tracks that had a corresponding match
    score_dict = {}

    recall = n_matched_true_tracks / n_true_tracks if n_true_tracks > 0 else 0
    precision = n_matched_pred_tracks / n_pred_tracks if n_pred_tracks > 0  else 0

    score_dict['n_matched_true_tracks'] = n_matched_true_tracks
    score_dict['n_true_tracks'] = n_true_tracks
    score_dict['n_matched_pred_tracks'] = n_matched_pred_tracks
    score_dict['n_pred_tracks'] = n_pred_tracks

    score_dict['n_false_positives'] = n_pred_tracks - n_matched_pred_tracks
    score_dict['n_unmatched_points'] = unmatched_points
    score_dict['n_frames'] = n_frames
    score_dict['false_track_points_per_frame'] = FTPF

    score_dict['prop_true_tracks_matched'] = recall
    score_dict['prop_pred_tracks_matched'] = precision

    # Store track fragmentation
    if n_true_tracks == 0:
        score_dict['pred_over_true_ratio'] = np.inf
    else:
        score_dict['pred_over_true_ratio'] = n_pred_tracks / n_true_tracks

    if n_pred_tracks == 0:
        score_dict['true_over_pred_ratio'] = np.inf
    else:
        score_dict['true_over_pred_ratio'] = n_true_tracks / n_pred_tracks

    # Save to json if desired
    if json_save_fpath:
        with open(json_save_fpath, 'w') as json_file:
            json.dump(score_dict, json_file)

    return score_dict


def aggregate_statistics(data, metrics, n_bins, outdir, micro_metric_path=None,
                            macro_metric_path=None, metrics_raw_path=None):
    '''Create a histogram plot for each metric. Optionally write out aggregated stats as json.

    Parameters
    ----------
    data : list
        list of experiment names + dictionaries containing metric scores.
    metrics : list
        the metrics to plot
    n_bins : int
        the number of histogram bins to use
    outdir : str
        directory to write plots
    micro_metric_path: str
        path to output file containing micro average performance metrics
    macro_metric_path: str
        path to output file containing macro average performance metrics
    metrics_raw_path: str
        path to output file containing raw distributions for each metric

    Returns
    -------
    success : bool
        whether data was plotted
    '''

    if not data or not metrics:
        logging.warning('No evalution plots generated.')
        return

    means = {}
    dists = {}

    exps = [d[0] for d in data]
    values = [d[1] for d in data]

    # Histograms
    success = False
    for metric in metrics:
        if metric not in values[0]:
            logging.warning('Metric {} does not match any batch statistic'
                            .format(metric))
            continue
        fig, ax = plt.subplots()
        ax.set_xlabel(metric.capitalize())
        ax.set_ylabel('Frequency')
        ax.set_title(metric.capitalize() + ' Distribution')
        ax.set_xlim((0, 1))

        # drop nonfinite metrics and count how many were dropped
        x = [d[metric] for d in values if np.isfinite(d[metric])]
        nonfinite = len(data) - len(x)

        if nonfinite:
            logging.warning("{} nonfinite track metrics skipped for metric histogram.".format(nonfinite))

        ax.hist(x, bins=n_bins, range=(0., 1.))

        if len(x) > 1:
            mean = statistics.mean(x)
            stdev = statistics.stdev(x)
            lbl = f'Mean = {mean:0.4f}\nStd. Dev. = {stdev:0.4f}'
            ax.text(0.05, 0.95, lbl, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top')
        elif len(x) > 0:
            mean = x[0]

        means[metric] = mean

        dict_temp = {}
        for d in data:
            dict_temp[d[0]] = d[1][metric]
        dists[metric] = OrderedDict(sorted(dict_temp.items(), key=lambda t: t[1]))
        
        # Save plot
        fig.tight_layout()
        out_path = op.join(outdir, metric + ".png")
        fig.savefig(out_path)
        success = True

    if not success:
        logging.warning('No histogram evalution plots generated.')

    if macro_metric_path:
        with open(op.join(outdir, macro_metric_path), 'w') as metrics_f:
            json.dump(means, metrics_f)

    if metrics_raw_path:
        with open(op.join(outdir, metrics_raw_path), 'w') as raws_f:
            json.dump(dists, raws_f, indent=4)

    if micro_metric_path and 'n_true_tracks' in values[0].keys():
        micro_n_matched_true_tracks = np.sum([d['n_matched_true_tracks'] for d in values])
        micro_n_matched_pred_tracks = np.sum([d['n_matched_pred_tracks'] for d in values])
        micro_n_true_tracks = np.sum([d['n_true_tracks'] for d in values])
        micro_n_pred_tracks = np.sum([d['n_pred_tracks'] for d in values])
        micro_n_unmatched_points = np.sum([d['n_unmatched_points'] for d in values])
        micro_n_frames = np.sum([d['n_frames'] for d in values])

        micro_recall = micro_n_matched_true_tracks / micro_n_true_tracks if micro_n_true_tracks > 0 else 0
        micro_precision = micro_n_matched_pred_tracks / micro_n_pred_tracks if micro_n_pred_tracks > 0 else 0
        micro_mean_fp = (micro_n_pred_tracks - micro_n_matched_pred_tracks) / len(values)
        micro_ftpf = micro_n_unmatched_points / micro_n_frames

        metric_micro_dict = {
            "micro_recall": micro_recall,
            "micro_precision": micro_precision,
            "micro_mean_fp": micro_mean_fp,
            "micro_false_tracks_per_frame": micro_ftpf
        }

        with open(op.join(outdir, micro_metric_path), 'w') as micro_f:
            json.dump(metric_micro_dict, micro_f, indent=4)