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
                        compress_background_smartgrid, reconstruct_background_smartgrid, \
                        remove_peaks, overlay_peaks, \
                        get_n_regions, get_filesize_est, get_background_error

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

    # python dev_bg_sizeopt.py --data /data/MLIA_active_data/data_OWLS/ACME/lab_data/2020_CESI_Data/200904/pickle/001_Mix27_10uM.pickle --peak_csv /data/MLIA_active_data/data_OWLS/WORK_DIR/jake/outdir_acme/10051608/001_Mix27_10uM/Unknown_Masses/001_Mix27_10uM_UM_peaks.csv --mugshot_dir /data/MLIA_active_data/data_OWLS/WORK_DIR/jake/outdir_acme/10051608/001_Mix27_10uM/Unknown_Masses/Mugshots

    exp_nopeak = remove_peaks(exp, peaks, config)

    ### SWEEP THROUGH AND ESTIMATE OUTPUT FILESIZE
    """
    sizes=[]
    for i in range(100):
        m = int(np.random.random() * 40 + 40)
        t = int(np.random.random() * 10 + 90)

        summary = compress_background_smartgrid(exp_nopeak, config, t_thresh_perc=t, m_thresh=m)
        nreg = get_n_regions(summary)
        filesize = write_pickle(summary, 'grid_background.bz2', True)
        est_filesize = get_filesize_est(nreg)

        logrow = [nreg, filesize, est_filesize]
        print(logrow)
        sizes.append(logrow)
    
    with open('sizes.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerows(sizes)
    """

    KBCAP = 80

    log = []
    t_thresh = 98
    m_thresh = 98

    summary = compress_background_smartgrid(exp_nopeak, config, peaks, t_thresh_perc=t_thresh, m_thresh_perc=m_thresh)
    nreg = get_n_regions(summary)
    est = get_filesize_est(nreg)

    while est > KBCAP:
        temp_t = t_thresh + 0.25
        temp_m = m_thresh + 0.25

        summaryA = compress_background_smartgrid(exp_nopeak, config, peaks, t_thresh_perc=temp_t, m_thresh_perc=m_thresh)
        reconA = reconstruct_background_smartgrid(summaryA)
        nregA = get_n_regions(summaryA)
        estA = get_filesize_est(nregA)
        errA = get_background_error(exp_nopeak, reconA)

        summaryB = compress_background_smartgrid(exp_nopeak, config, peaks, t_thresh_perc=t_thresh, m_thresh_perc=temp_m)
        reconB = reconstruct_background_smartgrid(summaryB)
        nregB = get_n_regions(summaryB)
        estB = get_filesize_est(nregB)
        errB = get_background_error(exp_nopeak, reconB)

        if errA > errB:
            m_thresh = temp_m
            est = estB
            print(t_thresh, m_thresh, est, errB)
        else:
            t_thresh = temp_t
            est = estA
            print(t_thresh, m_thresh, est, errA)
