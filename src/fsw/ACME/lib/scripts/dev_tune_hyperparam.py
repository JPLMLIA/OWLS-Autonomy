# Optimize Hyperparameters for ACME analyzer
# This is a Monte Carlo optimization. Parameters are randomly chosen and if they improve the performance we keep the
# new set of parameters.
#
#
# input:    ()
#
# output:   peak detection performance for input variables
#
# Steffen Mauceri
# Apr 2020
# Last updated: Oct 2020

import os
import numpy as np
import yaml
import matplotlib.pyplot as plt

from cli.ACME_pipeline import analyse_all_data  # analyser
from cli.ACME_evaluation import get_performance  # performance evaluation

# necessary to call with kwargs
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def run_optimizer():
    '''Main Program to optimize Hyperparameters'''

    ## make changes ***************************
    n_iter = 10         # number of iterations
    opt_n = 'name_'     # name of optimization run
    ## stop make changes **********************

    # initialize variables
    Precision = np.zeros((n_iter, 2))
    Recall = np.zeros((n_iter, 2))
    F1 = np.zeros((n_iter, 2))
    FP = np.zeros((n_iter, 2))

    # start iterating over different hyperparam.
    for i in range(n_iter):
        # read initial config file
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..', 'lib', 'params.yml')) as file:
            params = yaml.full_load(file)


        # *******   comment all variables out that you don't want to optimize!    *****************

        # change variables
        #window_y = int(np.random.uniform(low=5, high=20))  # window size in mass direction
        #if window_y%2 == 0: #make it odd
        #    window_y += 1
        #params['window_y'] = window_y
        #params['sigma'] = float(np.random.uniform(low=3.5, high=5))  # standard deviation of gausian filter
        #params['sigma_ratio'] = float(np.random.uniform(low=2.5, high=3))  # ratio of standard deviations for filter
        # params['min_peak_height'] = float(
        #     np.random.uniform(low=10, high=30))  # minimum peak height
        # params['min_filtered_threshold'] = float(np.random.uniform(low=150, high=300))  # threshold after convolutional filter is applied
        # params['min_peak_height'] = float(np.random.uniform(low=10, high=20))  # minimum height of a peak to be detected
        #params['min_peak_volume'] = float(
        #    np.random.uniform(low=200, high=5000))  # minimum volume (Ion counts) of a peak to be detected
        #params['min_peak_volume_top'] = float(np.random.uniform(low=100, high=params['min_peak_volume']))  # minimum volume (Ion counts) of top 50% of peak to be detected
        #params['min_peak_volume_zscore'] = float(
        #    np.random.uniform(low=5, high=30))  # minimum volume (Ion counts) of top 50% of peak to be detected
        # their SNR after background subtraction and convolution
        params['min_SNR_conv'] = float(np.random.uniform(low=10, high=30))  # threshold for filtering found peaks by
        # params['min_peak_base_width'] = float(np.random.uniform(low=1, high=8))  # minimum volume (Ion counts) of top 50% of peak to be detected
        # params['max_peak_roughness'] = float(
        #     np.random.uniform(low=50, high=120))  # minimum volume (Ion counts) of top 50% of peak to be detected

        #***********************************

        # save as yml file for evaluation
        with open(r'params_opt_' + opt_n + '_ID' + str(i) + '.yml', 'w') as file:
            documents = yaml.dump(params, file)

        # run ACME analyser with new parameters
        args = Namespace(data='/Users/smauceri/Projects/local/data_ACME/Data_Files_Hand_analyzed_opt' + opt_n + '/*/*.pickle',
                         masses='../lib/compoundsJune20.yml', params = 'params_opt_' + opt_n + '_ID' + str(i) + '.yml',
                         noplots=True, noexcel=True, debug_plots=None,
                         saveheatmapdata=None, knowntraces=None)
        analyse_all_data(vars(args))

        # evaluate analyser on Hand Labels
        args = Namespace(analyser_outputs='/Users/smauceri/Projects/local/data_ACME/Data_Files_Hand_analyzed_opt' + opt_n + '/*/*/*/*_peaks.csv',
                         path_labels='/Users/smauceri/Projects/local/data_ACME/Hand_Labels/*_label.csv', hand_labels=True,
                         mass_threshold=30, time_threshold=30, zscore=None)
        Precision_i, Recall_i, F1_i, FP_i = get_performance(**vars(args))
        # save performance and variable
        Precision[i, 0] = Precision_i
        Recall[i, 0] = Recall_i
        F1[i, 0] = F1_i
        FP[i, 0] = FP_i

        # plot result
        plot_name = 'Parameter study for all parameters ' + opt_n
        savepath = plot_name + '.png'
        plt.close('all')
        fig, ax = plt.subplots(figsize=(30, 5))

        ax.plot(range(n_iter), Precision[:, 0], linestyle='--', markersize=5, marker="+", c='k', label='Precision')
        ax.plot(range(n_iter), Recall[:, 0], linestyle='--', markersize=5, marker="x", c='b', label='Recall')
        ax.plot(range(n_iter), F1[:, 0], linestyle='--', markersize=5, marker="o", c='c', label='F1')
        ax.plot(range(n_iter), FP[:, 0]/100, linestyle='--', markersize=5, marker="d", c='g', label='FP/dataset/100')

        plt.ylim(0, 1)

        plt.ylabel("Performance")
        plt.xlabel('Run ID')
        plt.title('Parameter study ' + opt_n)
        plt.legend()
        plt.savefig(savepath, dpi=200)

if __name__ == '__main__':
    run_optimizer()