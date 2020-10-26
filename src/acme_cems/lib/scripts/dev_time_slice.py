""" Development script for just testing background algorithms """

import sys
import os
import os.path as op
import argparse
import pickle
import yaml
import logging
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

sys.path.append("../../..")
from utils                           import logger
from acme_cems.lib.background import write_pickle, read_pickle, write_csv, read_csv, \
                        compress_background_smartgrid, reconstruct_background_smartgrid, \
                        reconstruct_stats_smartgrid, remove_peaks, overlay_peaks, \
                        get_n_regions, get_filesize_est, get_background_error
from acme_cems.lib.utils import find_nearest_index

logger.setup_logger(os.path.basename(__file__).rstrip(".py"), "output")
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data',               default=None,
                                                help='Experiment to be processed')
    
    parser.add_argument('--params',             default='../../../cli/configs/acme_config.yml',
                                                help='Path to config file for Analyser')

    parser.add_argument('--peak_csv',           default=None,
                                                help='Peak CSV for experiment')

    parser.add_argument('--mugshot_dir',        default=None,
                                                help='Directory of mugshots for experiment')
    
    parser.add_argument('--time',               default=None,
                                                help='Time to slice at')
    
    parser.add_argument('--mass_max',           default=None,
                                                help='Max time for mass traces')
    
    ### FILE I/O

    # load configurations 
    args = parser.parse_args()

    with open(args.params, 'r') as f:
        config = yaml.safe_load(f)

    with open(args.data, 'rb') as f:
        exp_file = pickle.load(f)
    raw_size = op.getsize(args.data) / 1024

    peaks = []
    if args.peak_csv is not None:
        with open(args.peak_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                peaks.append(row)
    peaks_size = op.getsize(args.peak_csv) / 1024

    # parse .pickle
    exp = exp_file['matrix'].T
    time_axis = exp_file['time_axis']
    mass_axis = exp_file['mass_axis']

    ### PARAMETER SETUP

    # aspect ratio for better plotting
    xres = exp.shape[1]
    yres = exp.shape[0]
    aspect = (np.ptp(time_axis) / np.ptp(mass_axis)) / (xres / yres)

    # take slice at specified time
    slice_time = float(args.time)
    slice_index = find_nearest_index(time_axis, slice_time)

    # check time max
    if args.mass_max is None:
        mass_max = yres
    else:
        mass_max = find_nearest_index(mass_axis, float(args.mass_max))

    # example run for later
    # python dev_time_slice.py --data /data/MLIA_active_data/data_OWLS/ACME/lab_data/2020_CESI_Data/200904/pickle/001_Mix27_10uM.pickle --peak_csv /data/MLIA_active_data/data_OWLS/WORK_DIR/jake/outdir_acme/10161631_bg/001_Mix27_10uM/Unknown_Masses/001_Mix27_10uM_UM_peaks.csv --mugshot_dir /data/MLIA_active_data/data_OWLS/WORK_DIR/jake/outdir_acme/10161631_bg/001_Mix27_10uM/Unknown_Masses/Mugshots --time 14.59

    ### COMPRESSION AND RECONSTRUCTION

    # compress and export
    exp_nopeak = remove_peaks(exp, peaks, config)
    grid_summary = compress_background_smartgrid(exp_nopeak, config, peaks, t_thresh_perc=98.5, m_thresh_perc=98.5)
    filesize = write_pickle(grid_summary, 'grid_background.bz2', True)

    # reconstruct with and without sampling
    # sampling
    exp_nopeak_recon = reconstruct_background_smartgrid(grid_summary, orig_exp=exp_nopeak, eval=True)
    # non-sampling
    exp_nopeak_means, exp_nopeak_stds = reconstruct_stats_smartgrid(grid_summary)

    # overlay peaks on sampled reconstruction
    exp_recon, mugshot_size = overlay_peaks(exp_nopeak_recon, peaks, args.mugshot_dir)
    # overlay peaks on None-array
    exp_mugshot_only, _ = overlay_peaks(np.empty(exp.shape, dtype=object), peaks, args.mugshot_dir)

    ### PLOTTING

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(nrows=4, ncols=4, figure=fig)

    ax_cm1 = fig.add_subplot(gs[:, 0])
    ax_cm2 = fig.add_subplot(gs[:, 1])
    ax_tr1 = fig.add_subplot(gs[0, 2:])
    ax_tr2 = fig.add_subplot(gs[1, 2:])
    ax_tr3 = fig.add_subplot(gs[2, 2:])
    ax_tr4 = fig.add_subplot(gs[3, 2:])

    # Global setup
    cmap_max = np.percentile(exp, 99)
    cmap_extent = [time_axis[0], time_axis[-1], mass_axis[-1], mass_axis[0]]

    # Colormap original experiment
    ax_cm1.imshow(exp, cmap='viridis', vmin=0, vmax=cmap_max, extent=cmap_extent, aspect=aspect)
    ax_cm1.set_title(f"Raw Data ({raw_size:.2f}kB)")
    ax_cm1.set_ylabel("Mass (amu/z)")
    ax_cm1.set_xlabel("Time (min)")

    # Colormap reconstructed experiment
    ax_cm2.imshow(exp_recon, cmap='viridis', vmin=0, vmax=cmap_max, extent=cmap_extent, aspect=aspect)
    ax_cm2.set_title(f"Reconstructed Data from ASDPs\nPeaks List ({peaks_size:.2f}kB)\nMugshots ({mugshot_size:.2f}kB)\nBackground ({filesize:.2f}kB)")
    ax_cm2.set_ylabel("Mass (amu/z)")
    ax_cm2.set_xlabel("Time (min)")


    tr_max = np.max(exp[:, slice_index])
    # Timetrace original data
    ax_tr1.plot(mass_axis[:mass_max], exp[:mass_max, slice_index], color='tab:blue', label="Original", linewidth=0.5)
    ax_tr1.set_title(f"Mass Trace @ {slice_time:.2f} min")
    ax_tr1.set_ylabel("Ion Count")
    ax_tr1.set_xlabel("Mass (amu/z)")
    ax_tr1.set_ylim(0, tr_max)
    ax_tr1.legend()

    # Timetrace reconstruction statistics
    ax_tr2.plot(mass_axis[:mass_max], exp_mugshot_only[:mass_max, slice_index], color='tab:orange', label="Mugshots", linewidth=0.5)
    ax_tr2.plot(mass_axis[:mass_max], exp_nopeak_means[:mass_max, slice_index], color='tab:green', label="Summarized", linewidth=0.5)
    _std_min = exp_nopeak_means[:mass_max, slice_index] - exp_nopeak_stds[:mass_max, slice_index]
    _std_max = exp_nopeak_means[:mass_max, slice_index] + exp_nopeak_stds[:mass_max, slice_index]
    ax_tr2.fill_between(mass_axis[:mass_max], _std_min, _std_max, color='tab:green', alpha=.5, label="±1σ")
    ax_tr2.set_ylabel("Ion Count")
    ax_tr2.set_xlabel("Mass (amu/z)")
    ax_tr2.set_ylim(0, tr_max)
    ax_tr2.legend()

    # Timetrace reconstruction sampled
    ax_tr3.plot(mass_axis[:mass_max], exp_recon[:mass_max, slice_index], color='tab:purple', label="Sampled Data", linewidth=0.5)
    ax_tr3.set_ylabel("Ion Count")
    ax_tr3.set_xlabel("Mass (amu/z)")
    ax_tr3.set_ylim(0, tr_max)
    ax_tr3.legend()

    # Timetrace comparison
    ax_tr4.plot(mass_axis[:mass_max], exp[:mass_max, slice_index], color='tab:blue', label="Original", linewidth=0.5)
    ax_tr4.plot(mass_axis[:mass_max], exp_mugshot_only[:mass_max, slice_index], color='tab:orange', label="Mugshots", linewidth=0.5)
    ax_tr4.plot(mass_axis[:mass_max], exp_nopeak_means[:mass_max, slice_index], color='tab:green', label="Summarized Mean", linewidth=0.5)
    ax_tr4.fill_between(mass_axis[:mass_max], _std_min, _std_max, color='tab:green', alpha=.5, label="Summarized stddev")
    ax_tr4.set_ylabel("Ion Count")
    ax_tr4.set_xlabel("Mass (amu/z)")
    ax_tr4.set_ylim(0, tr_max)
    ax_tr4.legend()

    plt.tight_layout()

    fig.savefig("bg_time_slice.png", dpi=800)