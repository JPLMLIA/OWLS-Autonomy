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

sys.path.append("../../..")
from utils                           import logger
from acme_cems.lib.background import write_pickle, read_pickle, write_csv, read_csv, \
                        write_jpeg2000, read_jpeg2000, \
                        compress_background_PCA, reconstruct_background_PCA, \
                        compress_background_smartgrid, reconstruct_background_smartgrid, \
                        remove_peaks, overlay_peaks

logger.setup_logger(os.path.basename(__file__).rstrip(".py"), "output")
logger = logging.getLogger(__name__)


def plot_background_pipeline(orig_exp, peak_sub, peak_sub_recon, orig_recon, filesize, method, outpath="background.png", grid=None):

    cmap_max = np.percentile(orig_exp, 99)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(14, 6))
    
    ax1.imshow(orig_exp, cmap='viridis', vmin=0, vmax=cmap_max)
    ax1.set_title("Original Experiment")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Mass")

    ax2.imshow(peak_sub, cmap='viridis', vmin=0, vmax=cmap_max)
    ax2.set_title("Peaks Removed")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Mass")

    ax3.imshow(peak_sub_recon, cmap='viridis', vmin=0, vmax=cmap_max)
    ax3.set_title(f"Summarized with\n{method} ({filesize:.2f} kB)")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Mass")

    if grid is not None:
        grid.pop('shape')
        for row in grid.keys():
            for region in grid[row]:
                rect = patches.Rectangle(
                    (region[0], row[0]),
                    region[1] - region[0],
                    row[1] - row[0],
                    linewidth=0.01,
                    edgecolor='r',
                    facecolor='none'
                )
                ax3.add_patch(rect)

    ax4.imshow(orig_recon, cmap='viridis', vmin=0, vmax=cmap_max)
    ax4.set_title("Reconstruction with\nPeaks Overlaid")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Mass")

    plt.tight_layout()
    fig.savefig(outpath, dpi=800)


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
    
    args = parser.parse_args()

    with open(args.params, 'r') as f:
        config = yaml.safe_load(f)

    with open(args.data, 'rb') as f:
        exp_file = pickle.load(f)

    peaks = []
    if args.peak_csv is not None:
        with open(args.peak_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                peaks.append(row)

    exp = exp_file['matrix'].T

    ### PCA Background Summarization with Peaks

    # python dev_background.py --data /data/MLIA_active_data/data_OWLS/ACME/lab_data/2020_CESI_Data/200904/pickle/001_Mix27_10uM.pickle --peak_csv /data/MLIA_active_data/data_OWLS/WORK_DIR/jake/outdir_acme/10151333/001_Mix27_10uM/Unknown_Masses/001_Mix27_10uM_UM_peaks.csv --mugshot_dir /data/MLIA_active_data/data_OWLS/WORK_DIR/jake/outdir_acme/10151333/001_Mix27_10uM/Unknown_Masses/Mugshots/
    # python dev_background.py --data /data/MLIA_active_data/data_OWLS/ACME/lab_data/2020_CESI_Data/200710/pickle/001_Mix27_10uM.pickle --peak_csv /data/MLIA_active_data/data_OWLS/WORK_DIR/jake/outdir_acme/10151355/001_Mix27_10uM/Unknown_Masses/001_Mix27_10uM_UM_peaks.csv --mugshot_dir /data/MLIA_active_data/data_OWLS/WORK_DIR/jake/outdir_acme/10151355/001_Mix27_10uM/Unknown_Masses/Mugshots/
    """
    exp_nopeak = remove_peaks(exp, peaks, config)
    PCA_summary = compress_background_PCA(exp_nopeak, config, n_comp=1)
    filesize = write_pickle(PCA_summary, 'PCA_background.bz2', True)
    exp_nopeak_recon = reconstruct_background_PCA(*PCA_summary, orig_exp=exp, eval=True)
    exp_recon = overlay_peaks(exp_nopeak_recon, peaks, args.mugshot_dir, config)
    plot_background_pipeline(exp, exp_nopeak, exp_nopeak_recon, exp_recon, filesize, "PCA1", outpath="pca_background.png")
    
    exp_nopeak = remove_peaks(exp, peaks, config)
    filesize = write_jpeg2000(exp_nopeak, c_ratio=200)
    exp_nopeak_recon = read_jpeg2000("jpeg2k_background_*.jp2")
    exp_recon = overlay_peaks(exp_nopeak_recon, peaks, args.mugshot_dir, config)
    plot_background_pipeline(exp, exp_nopeak, exp_nopeak_recon, exp_recon, filesize, "jpg2k_c150", outpath="jpg2k_background.png")
    """

    exp_nopeak = remove_peaks(exp, peaks, config)
    grid_summary = compress_background_smartgrid(exp_nopeak, config, peaks, t_thresh_perc=98.5, m_thresh_perc=98.5)
    filesize = write_pickle(grid_summary, 'grid_background.bz2', True)
    exp_nopeak_recon = reconstruct_background_smartgrid(grid_summary, orig_exp=exp_nopeak, eval=True)
    exp_recon = overlay_peaks(exp_nopeak_recon, peaks, args.mugshot_dir)
    plot_background_pipeline(exp, exp_nopeak, exp_nopeak_recon, exp_recon, filesize, "grid_t98.5_m98.5", outpath="grid_background.png", grid=None)