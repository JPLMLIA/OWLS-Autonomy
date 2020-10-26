'''
Command line interface for the ACME data simulator
'''
import sys
sys.path.append("../")

import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import os
import sys
import logging

from acme_cems.lib.analyzer          import make_crop
from utils                           import logger

logger.setup_logger(os.path.basename(__file__).rstrip(".py"), "output")
logger = logging.getLogger(__name__)

def plot_exp(exp, save = False, save_path = None):
    '''make plots of raw data in a similar style than the ACME heat maps

    Parameters
    ----------
    exp: ndarray
        data matrix
    save: bool
        should the plot be saved?
    save_path: str
        path to where the file should be saved

    Returns
    -------
    plot to screen or as .png to disk

    '''
    plt.figure(figsize=(20, 10))
    max = np.std(exp) * 3 + np.mean(exp)
    min = 0
    plt.imshow(exp, vmin= min, vmax=max, cmap='inferno')
    plt.xlabel('Time [idx]')
    plt.ylabel('Mass [idx]')
    plt.colorbar(label='Ion Counts clipped at 3 std)')
    if save:
        plt.savefig(save_path + '.png', dpi = 200)
        plt.close()
    else:
        plt.show()


def add_peak(exp, peak):
    '''adds gaussian peaks on data matrix

    Parameters
    ----------
    exp: ndarray
        data matrix
    peak: DataFrame
        peak properties (mass,time,width,height,...)

    Returns
    -------
    exp: ndarray
        data matrix with added peaks
    volume: float
        volume of added peak (total number of ion counts for event that caused peak)
    '''
    size = int(peak.mass_width_idx) * 2     # size of matrix to calculate gaussian peak for
    size -= 1   #make size odd
    sigma_x = peak.time_width_idx/2 / 3  # convert to 1 sigma
    sigma_y = peak.mass_width_idx/2 / 3  # convert to 1 sigma
    height = peak.height            # peak height

    x, y = np.meshgrid(np.linspace(-(size // 2), size // 2, size), np.linspace(-(size // 2), (size // 2), size))
    g = np.exp(-(x**2 / (2*sigma_x**2) + y**2 / (2*sigma_y**2)))*height

    # calculate volume of peak (ion count of peak integrated over time and mass, approximated as a sum)
    volume = np.sum(g)

    # add to matrix
    x_pos = peak.time_idx
    y_pos = peak.mass_idx
    exp[y_pos - (size // 2): y_pos + (size // 2) + 1, x_pos - (size // 2): x_pos + (size // 2) + 1] += g

    return exp, volume


def add_stripe(exp, stripe, cliffs):
    '''adds salt stripes to data matrix

    Parameters
    ----------
    exp: ndarray
        data matrix
    stripe: DataFrame
        stripe properties (mass_idx,width,height,...)
    cliffs: ndarray
        time_idx of cliffs

    Returns
    -------
    exp: ndarray
        data matrix with stripes
    '''

    width = stripe.stripe_width
    # make stripe_width odd
    if width%2 == 0:
        width += 1

    # make empty stripe
    stripe_mat = np.zeros((width, exp.shape[1]))

    # add offset
    for i in range(1,len(cliffs)):
        stripe_mat[:,cliffs[i-1]:cliffs[i]] += stripe.stripe_offset * np.random.randint(low=0, high=2)

    # smooth transitions
    smooth = 10
    for i in range(len(stripe_mat)):
        stripe_mat[i,:] = np.convolve(stripe_mat[i,:], np.ones((smooth,))/smooth, mode='same')

    # add noise on stripe
    constrained_noise = np.random.randn(stripe_mat.shape[0], stripe_mat.shape[1])
    constrained_noise[np.abs(constrained_noise) > 3] = 3
    stripe_mat += constrained_noise * stripe.stripe_noise

    # add to matrix
    y_pos = stripe.stripe_mass_idx
    y_len = width
    exp[y_pos - (y_len // 2): y_pos + (y_len // 2) + 1, :] += stripe_mat

    return exp


def add_background_offset(exp, background_offsets, cliffs):
    '''

    Parameters
    ----------
    exp: ndarray
        data matrix
    background_offsets: ndarray
        offset to be added to background
    cliffs: ndarray
        time_idx of cliffs

    Returns
    -------
    exp: ndarray
        data matrix with background offsets added

    '''
    background_mat = np.zeros_like(exp)
    for i in range(1, len(cliffs)):
        background_mat[:, cliffs[i - 1]:cliffs[i]] += background_offsets[i - 1]

    # smooth transitions
    smooth = 10
    for i in range(len(background_mat)):
        background_mat[i, :] = np.convolve(background_mat[i, :], np.ones((smooth,)) / smooth, mode='same')

    exp += background_mat
    return exp


def acme_sim(args):
    '''Maine program to simulate ACME data

    Parameters
    ----------
    args
        params: Path to config file for Simulator
        save_path: Path to save output of Simulator
        n_runs: number of simulation runs

    Returns
    -------

    '''
    params = args.get('params')
    save_path = args.get('out_dir')
    n_runs = args.get('n_runs')
    DEBUG = False

    # make parent outdir if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # make case name from parameter file
    case_name = params.split('.')[-2]
    case_name = case_name.split('/')[-1]

    params = yaml.safe_load(open(params, 'r'))

    # read parameter
    n_peaks = params['n_peaks']                                     # number of peaks
    height_min = params['height_min']                               # peak height in counts
    height_max = params['height_max']
    mass_width_min = params['mass_width_min']                       # 2 * 3 sigma of peak in mass dimension [idx]
    mass_width_max = params['mass_width_max']
    time_mass_width_ratio_min = params['time_mass_width_ratio_min'] # ratio of peak width in mass vs time
    time_mass_width_ratio_max = params['time_mass_width_ratio_max']
    peak_min_dist = params['peak_min_dist']                         # minimum distance of peak to each oterh in [idx]

    background_noise = params['backgound_noise']                    # 1 sigma of background noise
    background_offset_min = params['background_offset_min']         # offset
    background_offset_max = params['background_offset_max']         # offset
    n_stripes = params['n_stripes']                                 # number of horizontal stripes
    stripes_noise_min = params['stripes_noise_min']                 # added Noise on stripes
    stripes_noise_max = params['stripes_noise_max']
    stripes_offset_min = params['stripes_offset_min']               # offset of stripes
    stripes_offset_max = params['stripes_offset_max']
    stripes_width_min = params['stripes_width_min']                 # width of stripes
    stripes_width_max = params['stripes_width_max']
    n_cliffs = params['n_cliffs']                                   # number of vertical cliffs (abrupt changes of stripes)

    peaks_on_stripes = params['peaks_on_stripes']                   # allow for peaks to fall on stripes

    # conversion from pixel to z/amu and min
    mass_idx_to_amu = 0.0833    # from 190411010_Mix25_50uM_NaCl_1M.raw.pickle'
    time_idx_to_min = 0.0061

    # iterate over number of simulations we want to perform
    n = 0
    while n < n_runs:
        n += 1

        # make folder for data
        case_num = str(n).zfill(2)
        outdir = os.path.join(save_path, case_num)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # make empty data
        mass_axis = np.arange(70.0833,400, mass_idx_to_amu)
        time_axis = np.arange(0.001,39.988,time_idx_to_min)
        exp = np.zeros((len(time_axis), len(mass_axis)))

        # transpose so x is time and y is mass
        exp = exp.transpose()

        n_peaks_init = 2 * n_peaks  # make more peaks initially so that the final number of peaks equals n_peaks

        # generate time_idx of peaks, avoid being to close to boarders
        time_idx = np.random.randint(low=mass_width_max * time_mass_width_ratio_max,
                                     high=len(time_axis) - mass_width_max * time_mass_width_ratio_max,
                                     size=(n_peaks_init))
        # generate mass_idx of peaks, avoid being to close to boarders
        mass_idx = np.random.randint(low=mass_width_max, high=len(mass_axis)- mass_width_max, size=(n_peaks_init))
        # generate peak height
        height = np.random.uniform(low=height_min, high=height_max, size=(n_peaks_init))
        # generate peak width [sigma]
        mass_width_idx = np.random.uniform(low=mass_width_min, high=mass_width_max, size=(n_peaks_init))
        time_mass_ratio = np.random.uniform(low=time_mass_width_ratio_min, high=time_mass_width_ratio_max,
                                            size=(n_peaks_init))
        time_width_idx = mass_width_idx * time_mass_ratio

        # convert mass and time to amu and Min
        mass_width = mass_width_idx * mass_idx_to_amu
        time_width = time_width_idx * time_idx_to_min

        # calc peak time and mass
        time = time_axis[time_idx]
        mass = mass_axis[mass_idx]

        # calculate peak start time and end time
        start_time_idx = (time_idx - time_width_idx/2).astype(int)
        end_time_idx = (time_idx + time_width_idx/2).astype(int)
        start_time = time - time_width/2
        end_time = time + time_width/2

        # put all variables in DataFrame
        peaks = {'time_idx':time_idx, 'time':time , 'mass_idx': mass_idx, 'mass':mass, 'mass_width_idx': mass_width_idx,
                 'mass_width': mass_width,'time_width_idx': time_width_idx, 'time_width': time_width,'height': height,
                 'start_time_idx':start_time_idx, 'start_time': start_time,
                 'end_time_idx': end_time_idx, 'end_time': end_time}
        peaks = pd.DataFrame(data=peaks)

        # remove peaks that are to close to each other
        too_close=[]
        for peak in peaks.itertuples():
            dist = ((peak.mass_idx-peaks.mass_idx)**2 + (peak.time_idx-peaks.time_idx)**2)**0.5 #calc distance of peak to every other peak
            dist[dist == 0] = dist.max()
            if dist.min() < peak_min_dist:
                too_close.append(peak.Index)
        peaks.drop(too_close, inplace=True)

        # limit number of peaks to n_peaks
        peaks = peaks.iloc[0:n_peaks]

        ## add peaks to empty data field
        volumes = []
        for peak in peaks.itertuples():
            exp, volume = add_peak(exp, peak)      # make peak matrix and add to exp
            volumes.append(volume)

        peaks['volume'] = volumes

        if DEBUG:
            plot_exp(exp)

        # add 'cliffs'
        cliffs = []
        while len(cliffs) < n_cliffs:
            c = np.random.randint(low=10, high=len(time_axis)-10)   # generate location of cliffs
            # make sure that there are no peaks on cliffs
            min_dist = np.min(np.abs(c - peaks.time_idx))
            if min_dist > (mass_width_max * time_mass_width_ratio_max) / 2 + 5:
                cliffs.append(c)
        # make numpy array, sort, add start and end point
        cliffs = np.array(cliffs)
        cliffs = np.sort(cliffs)
        cliffs = np.concatenate(([0], cliffs, [len(time_axis)]))

        # add 'stripes'
        stripe_mass_idx = []
        while len(stripe_mass_idx) < n_stripes:
            s = np.random.randint(low=stripes_width_max, high=len(mass_axis) - stripes_width_max)
            min_dist = np.min(np.abs(s - peaks.mass_idx))
            # make sure that there are no peaks on stripes if peaks_on_stripes == False
            if not peaks_on_stripes:
                if min_dist > mass_width_max / 2:
                    stripe_mass_idx.append(s)
            else:
                stripe_mass_idx.append(s)

        stripe_noise = np.random.uniform(low=stripes_noise_min, high=stripes_noise_max, size=(n_stripes))
        stripe_offset = np.random.uniform(low=stripes_offset_min, high=stripes_offset_max, size=(n_stripes))
        stripe_width = np.random.randint(low=stripes_width_min, high=stripes_width_max, size=(n_stripes))

        stripes = {'stripe_noise': stripe_noise, 'stripe_offset': stripe_offset, 'stripe_width': stripe_width,
                   'stripe_mass_idx': stripe_mass_idx}
        stripes = pd.DataFrame(data=stripes)

        for stripe in stripes.itertuples():
            exp = add_stripe(exp, stripe, cliffs)

        if DEBUG:
            plot_exp(exp)

        # add random noise to background
        constrained_noise = np.random.randn(np.shape(exp)[0],np.shape(exp)[1])
        # remove everything above and below 3 std to ensure we don't have noise that looks like peaks
        constrained_noise[np.abs(constrained_noise) > 3] = 3
        exp += constrained_noise * background_noise

        # add background offset
        background_offsets = np.random.uniform(low=background_offset_min, high=background_offset_max, size=(n_cliffs + 1))
        exp = add_background_offset(exp, background_offsets, cliffs)

        # remove negative values
        exp[exp < 0] = 0

        if DEBUG:
            plot_exp(exp)

        out_path = os.path.join(outdir, case_name + '_' + case_num)
        plot_exp(exp, save=True, save_path=out_path)


        # make hand label file
        # add total peak height with background offsets and noise as "Counts: Raw"
        Counts_Raw = exp[peaks.mass_idx, peaks.time_idx]
        peaks['Counts: Raw'] = Counts_Raw

        # add 'Z-score' = peak height / std(background)
        # add "Counts: Baseline" and "Sigma"
        Counts_Baseline = []
        Zscores = []
        Sigma =[]
        for peak in peaks.itertuples():
            window_x = int(peak.time_width_idx) + 10
            if window_x%2==0: # make sure its odd
                window_x +=1
            window_y = int(peak.mass_width_idx)
            if window_y%2==0:
                window_y +=1
            center = int(peak.time_width_idx)
            if center%2==0:
                center +=1
            peak_xy = np.array([peak.mass_idx, peak.time_idx])
            crop_center, crop_left, crop_right = make_crop(peak_xy,exp,window_x,window_y,center)
            background_std = (np.std(crop_right) + np.std(crop_left))/2
            # calc and append values to lists
            Zscores.append(peak.height / background_std)
            Counts_Baseline.append((np.median(crop_right) + np.median(crop_left))/2)
            Sigma.append(background_std)
        peaks['Z-score'] = Zscores
        peaks['Counts: Baseline'] = Counts_Baseline
        peaks['Sigma'] = Sigma

        # rename peak height to 'Counts: Baseline removed'
        peaks.rename(columns={'height': 'Counts: Baseline removed'}, inplace=True)

        # make sure that we dont have ambiguous peaks
        if np.min(np.array(Zscores)) < 5:
            logging.info('repeating run ' + str(n) + ' because z-score of < 5 found')
            n -= 1
            continue

        # write hand labels to disc
        save_path_label = out_path + '_label.csv'
        peaks.to_csv(save_path_label, sep=',', index=False)

        # transpose back to original format
        exp = exp.transpose()

        # save data to pickle file
        save_path_pickle = os.path.join(out_path + '.raw.pickle')
        data = {"matrix": exp, "mass_axis": mass_axis, "time_axis": time_axis}
        pickle.dump(data, open(save_path_pickle, "wb"))

    logging.info('>>> Done')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--params',      default='configs/acme_sim_params.yml',
                                         help='Path to config file for Simulator')

    parser.add_argument('--out_dir',     default=None,
                                         help='Path to save output of Simulator')

    parser.add_argument('--n_runs',      default=10,
                                         help='Number of simulation runs to perform')

    args = parser.parse_args()

    acme_sim(vars(args))
    logging.info("======= Done =======")
