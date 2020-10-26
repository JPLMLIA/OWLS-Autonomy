# code to visualize output from ACME_evaluation to better understand FN FP TP peaks
# need to run ACME_evaluation.py before using this script
#
# Steffen Mauceri
# June 2020

import glob
import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from acme_cems.lib.analyser2D_obj_v4 import make_crop


#Location of original ACME data -- Passed as glob
ACME_data = '/Users/smauceri/Projects/local/data_ACME/Data_Files_v4_all/*/*.raw.pickle'
#Location of FN output from ACME evaluaiton -- Passed as globs
ACME_eval_output = '/Users/smauceri/Projects/local/data_ACME/Data_Files_v4_all/*/*/*/eval/*_FN.csv'

def plot_time_mass(exp, peak_df, save_dir, title):
    ''' plots peaks vs mass and time and saves them as png'''
    # size for plotting
    window_x = 101
    window_y = 33
    center = 99
    # size as specified in params.yml
    center_x_param = 41
    window_y_param = 13
    window_x_param = 61

    for p in peak_df.itertuples():
        peak = [int(p.mass_idx), int(p.time_idx)]
        crop,_,_ = make_crop(peak, exp, window_x, window_y, center)

        plot_name = 'TimeMass_' + str(peak[1]) + '_' + str(peak[0])
        savepath = save_dir + plot_name + '.png'

        exp_mean = np.mean(crop)
        exp_std = np.std(crop)
        matrix_clip = np.clip(crop, 0, exp_mean + 3 * exp_std)

        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 4))

        # prepare values for rectangle to show area considered as background

        # plot peak
        ax.plot(center // 2, window_y // 2, linestyle='none', markersize=5, marker="+", fillstyle='none',
                c='r', label='Peak')

        # prepare values for rectangle to show area considered as background
        # make right rectangle
        
        r_top_left = (window_x // 2 + center_x_param / 2, window_y // 2 - window_y_param / 2)
        r_height = window_y_param
        r_width = (window_x_param - center_x_param) / 2
        rectangle_r = plt.Rectangle(r_top_left, r_width, r_height, ec='green', fill=False, label='Background')
        # make left rectangle
        r_top_left = (window_x // 2 - center_x_param / 2 - r_width, window_y // 2 - window_y_param / 2)
        rectangle_l = plt.Rectangle(r_top_left, r_width, r_height, ec='green', fill=False, label='Background')
        fig.gca().add_patch(rectangle_r)
        fig.gca().add_patch(rectangle_l)
        # make rectangle to show center
        r_top_left = (window_x // 2 - center_x_param / 2, window_y // 2 - window_y_param / 2)
        r_width = center_x_param
        rectangle_l = plt.Rectangle(r_top_left, r_width, r_height, ec='red', ls='--', fill=False, label='Center')
        fig.gca().add_patch(rectangle_r)
        fig.gca().add_patch(rectangle_l)

        
        cmap = ax.imshow(matrix_clip, cmap='inferno')
        fig.colorbar(cmap, ax=ax, label='Ion Counts (clipped at 3 std)')

        # make x/y legends
        time_labels = np.arange(0, matrix_clip.shape[1], 5)
        time_labels -= matrix_clip.shape[1] // 2
        mass_labels = np.arange(0, matrix_clip.shape[0], 5)
        mass_labels -= matrix_clip.shape[0] // 2
        ax.set_xticks(np.arange(0, matrix_clip.shape[1], 5))
        ax.set_xticklabels(time_labels)
        ax.set_yticks(np.arange(0, matrix_clip.shape[0], 5))
        ax.set_yticklabels(mass_labels)

        plt.ylabel("Mass [mass idx relative to peak]")
        plt.xlabel("Time [time idx relative to peak]")
        plt.title(title + str(peak[1]) + '_' + str(peak[0]) + ' Raw Data with identified Peak')

        # handle background label being shown twice
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        #plt.show()
        plt.savefig(savepath, dpi=200)

def plot_time(exp, peak_df, save_dir, title):
    '''plots peak vs time and saves as png'''
    # size for plotting
    window_x = 101
    window_y = 5
    # size as specified in params.yml
    center_x_param = 41
    window_x_param = 61

    for p in peak_df.itertuples():
        peak = [int(p.mass_idx), int(p.time_idx)]
        plot_name = 'Time_' + str(peak[1]) + '_' + str(peak[0])
        savepath = save_dir + plot_name + '.png'

        # get trace that goes through peak
        trace = np.mean(exp[peak[0] - window_y//2 : peak[0] + window_y//2, :],0)

        background_range = [p.time_idx - (center_x_param // 2),
                            p.time_idx + (center_x_param // 2) + 1,
                            p.time_idx - (window_x_param // 2),
                            p.time_idx + (window_x_param // 2) + 1]

        plt.close('all')
        plt.figure(figsize=(10, 5))

        for b in background_range:
            plt.axvline(b, alpha=0.3, c='k', label='Background Time Range')

        # plot raw data slice
        plt.plot(trace, '.-r', alpha=0.5, label='Raw Data')

        # plot found peak, start and end time
        plt.plot(p.time_idx, trace[peak[1]], '*g', alpha=0.5, markersize=10,label='Peak')

        plt.xlabel("Time (Min)")
        plt.ylabel("Ion Counts")
        plt.title(title + str(peak[1]) + '_' + str(peak[0]) + 'Raw Data with identified Peak +-3 mass_idx')
        x1 = np.max([0, p.time_idx - window_x // 2])
        x2 = np.min([len(trace) - 1, p.time_idx + window_x // 2])
        plt.xlim(x1, x2)

        # make sure that every label is only plotted once
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        #plt.show()
        plt.savefig(savepath, dpi=200)

def plot_features(FP, TP, save_dir, title):
    '''plots peak features of FP and TP'''

    features = FP.columns
    features = features[1:] # remove counter column

    for feature in features:
        FP_feature = FP[feature]
        TP_feature = TP[feature]

        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 4))
        max_bin = np.max(np.array([FP_feature.max(), TP_feature.max()]))
        min_bin = np.min(np.array([FP_feature.min(), TP_feature.min()]))
        plt.hist(FP_feature, histtype='step', bins=50, range=(min_bin,max_bin), label='FP')
        plt.hist(TP_feature, histtype='step', bins=50, range=(min_bin,max_bin), label='TP')
        plt.legend()
        plt.xlabel(feature)
        plt.ylabel('occurrence')

        plot_name = 'TP_FP_' + feature
        savepath = save_dir + plot_name + '.png'
        #plt.show()
        plt.savefig(savepath, dpi=200)

    pass

# find output from ACME_evaluation
outputs = glob.glob(ACME_eval_output, recursive=False)
output_names = []
for o in outputs:
    o_name = o.split('/')[-1]
    o_name = o_name.split('_FN.csv')[0]
    output_names.append(o_name)

if len(output_names) == 0:
    print('Error: No outputs from ACME evaluation found to process in directory ', ACME_eval_output)
    exit()
else:
    print('Found ' + str(len(output_names)) + ' ACME evaluation outputs to process')

# find ACME data
data_path = glob.glob(ACME_data, recursive=False)
data_names = []
for d in data_path:
    d_name = d.split('/')[-1]
    d_name = d_name.split('.raw.pickle')[0]
    data_names.append(d_name)

if len(data_names) == 0:
    print('Error: No ACME data found to process in directory ', ACME_data)
    exit()
else:
    print('Found ' + str(len(data_names)) + ' ACME data')


# find a match and load the data
for i in range(len(outputs)):
    o = outputs[i]
    FP = pd.read_csv(o.split('FN.csv')[0] + 'FP.csv')
    FN = pd.read_csv(o)
    TP = pd.read_csv(o.split('FN.csv')[0] + 'TP.csv')

    for j in range(len(data_path)):
        if data_names[j] == output_names[i]:
            d = data_path[j]

            save_dir = os.path.dirname(o)
            # Reading data file
            file = pickle.load(open(d, 'rb'))
            exp = file['matrix']
            exp = exp.T  # transpose

            # make plot vs time and mass as matrix
            plot_time_mass(exp, FP, save_dir + '/FP', 'FP: ')
            plot_time_mass(exp, FN, save_dir + '/FN', 'FN: ')

            # make plot vs time as time-series
            plot_time(exp, FP, save_dir + '/FP', 'FP: ')
            plot_time(exp, FN, save_dir + '/FN', 'FN: ')

            # make plots of FP vs TP features
            if (len(FP) > 0) & (len(TP) > 0):
                plot_features(FP,TP, save_dir + '/FPvsTP', 'FPvsTP:')
