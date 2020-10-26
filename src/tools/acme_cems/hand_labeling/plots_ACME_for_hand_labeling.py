# reads in a folder with ACME samples and plots the 2D matrix as slices with time on x-axis.
# Each plot is made for a fixed mass bin
# Afterwards, these plots can then be used for hand labeling peaks (currently done in MATLAB)
#
# Steffen Mauceri
# May 2020
# updated: Oct 2020


import glob
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy import ndimage


def plot_slice(slice, name, title, id):
    '''helper function for individual subplots'''
    plt.subplot(n_rows, n_columns, id)
    plt.plot(slice)
    plt.title(name + title)
    plt.xlabel('time_idx')
    plt.ylabel('counts')
    plt.xticks(range(0,np.shape(exp)[1],100))
    plt.grid(b=True)

    # ensure that plot is always at same horizontal position in image
    locs, labels = plt.yticks()
    labels = []
    for l in locs:
        labels.append(str(int(l)).zfill(6))
    plt.yticks(locs, labels, fontsize=12)


# Make Changes
path_labels = '/Users/smauceri/Projects/local/data_ACME/Data_Files/**/*.raw.pickle'
mass_bin = 0.5 #in amu
# Stop making changes

# convert to mass index
mass_idx_to_amu = 0.0833
mass_bin = int(mass_bin//mass_idx_to_amu)
# make odd
if mass_bin%2 == 0:
    mass_bin += 1
print('mass bin is ' + str(mass_bin) + ' [idx] and ' + str(mass_bin * mass_idx_to_amu) + ' [amu]')


# find datasets
datasets = glob.glob(path_labels, recursive=True)

if len(datasets) == 0:
    print('Error: No ACME datasets found to process in directory ', path_labels)
    exit()
else:
    print('Found ' + str(len(datasets)) + ' ACME datasets')


## iterate over datasets
for dataset in datasets:
    # read data
    file = pickle.load(open(dataset, 'rb'))
    name = dataset.split('/')[-1]
    name = name.split('.')[0]
    exp = file['matrix']
    # transpose in exp to bring it in the format we are used to (Time on x, Mass on y)
    exp = np.transpose(exp)
    # make background subtracted exp
    exp_bkr = exp - ndimage.median_filter(exp, size=[1, 50])

    ## iterate over mass axis of exp
    for m in range(0,np.shape(exp)[0]-mass_bin,mass_bin):
        slice_raw = np.sum(exp[m:m+mass_bin,:],0)
        slice_bkr = np.sum(exp_bkr[m:m+mass_bin,:],0)
        slice_bkr_root = np.copy(slice_bkr)
        slice_bkr_root[slice_bkr_root < 0] = 0
        slice_bkr_root = np.sqrt(slice_bkr_root)

        center_mass = m + mass_bin//2 +1

        ## plot slice
        title = name + '_' + str(center_mass).zfill(4)
        print('plotting ' + title)

        # set up the plot
        n_columns = 1
        n_rows = 3
        plt.subplots(n_rows, n_columns, dpi=100, figsize=(40, 15))
        #plt.subplots_adjust(left=0.1, bottom=0.1, right=39, top=14, wspace=0.2, hspace=0.2)

        # plot raw data
        plot_name = 'Raw Counts: '
        plot_slice(slice_raw,plot_name,title,1)

        # plot background subtracted data
        plot_name = 'Counts - Background: '
        plot_slice(slice_bkr,plot_name,title,2)

        # plot log of background subtracted data
        plot_name = 'sqrt(Counts - Background): '
        plot_slice(slice_bkr_root, plot_name, title, 3)


        plt.savefig('./Hand_Label_Plots/' + title + '.png',bbox_inches='tight',pad_inches = 0.2, dpi = 100)
        plt.close('all')
        #plt.show()
