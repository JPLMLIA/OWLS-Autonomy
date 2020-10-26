# part of the ACME hand label generation Pipeline
# last step and adds a z-score to every hand labeled peak

# Steffen Mauceri
# June 2020
# updated: Oct 2020


import glob
import pandas as pd
import pickle
import numpy as np

# these hyperparameters should correspond to what is used in the actual ACME analyzer.
def make_crop(peak, exp, window_x = 61, window_y = 13, center = 41):
    '''makes two crops: one of the peak and one of the surrounding area

    Parameters
    ----------
    peak: ndarray
        x and y coordinate of peak center
    roi: ndarray
        matrix to be cropped
    window_x: int
        total width of crop window. Must be odd
    window_y, int
        total height of crop window. Must be odd
    center: int
        center width that we be cropped from cropped again from window. Must be odd

    Returns
    -------
    crop_center: ndarray
        center crop of crop (contains peak)
    crop_l: ndarray
        crop of window without center (contains left side of peak)
    crop_r: ndarray
        crop of window without center (contains right side of peak)

    Examples
    -------
    >>> peak = np.array([10, 10])
    >>> roi = np.eye(100)
    >>> window_x = 9
    >>> window_y = 3
    >>> center = 3
    >>> crop_center, crop_l, crop_r = make_crop(peak,roi,window_x,window_y,center)
    >>> crop_center
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    >>> np.mean(crop_l)
    0.0
    >>> np.shape(crop_l)
    (3, 3)
    >>> center = 4
    >>> crop_center, crop_l, crop_r = make_crop(peak,roi,window_x,window_y,center) # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
        assert center%2 == 1
    AssertionError
    '''
    # make sure all inputs are odd
    assert window_x % 2 == 1
    assert window_y % 2 == 1
    assert center % 2 == 1

    top = - (window_y // 2) + peak[0]
    bottom = window_y // 2 + peak[0] + 1
    left = - (window_x // 2) + peak[1]
    right = window_x // 2 + peak[1] + 1
    crop = exp[top: bottom, left: right]
    crop_center = crop[:, window_x // 2 - (center // 2): window_x // 2 + center // 2 + 1]
    crop_left = crop[:, : window_x // 2 - (center // 2)]
    crop_right = crop[:, window_x // 2 + center // 2 + 1:]

    return crop_center, crop_left, crop_right

# glob path to one or multiple ACME datasets
path_data = '/Users/smauceri/Projects/local/data_ACME/Data_Files/*/*.raw.pickle'
# path to hand labels that correspond to data
path_labels = '/Users/smauceri/Projects/local/data_ACME/Hand_Labels/*_label.csv'

data = glob.glob(path_data)
labels = glob.glob(path_labels)

for l in labels:
    # get name label
    l_name = l.rstrip('_label.csv')
    l_name = l_name.split('/')[-1]

    for d in data:
        #get name of data
        d_name = d.split('/')[-1]
        d_name = d_name.split('.')[0]


        if l_name == d_name:
            print(l_name)
            file = pickle.load(open(d, 'rb'))
            exp = file['matrix']
            # transpose in exp to bring it in the format we are used to (Time on x, Mass on y)
            exp = np.transpose(exp)

            label_peaks = pd.read_csv(l)
            z_scores = []
            for p in label_peaks.itertuples():
                # make crop
                peak = np.array([p.mass_idx, p.time_idx])
                crop_center, crop_left, crop_right = make_crop(peak,exp)
                ## calc z-score
                # use range since label is often not exactly on peak top
                peak_height = np.max(exp[p.mass_idx-2:p.mass_idx+2, p.time_idx-10: p.time_idx+10])
                background = np.max([np.median(crop_left),np.median(crop_right)])
                # std of left or right, what ever is higher
                noise = np.max([np.std(crop_left), np.std(crop_right)])
                e = 0.0001
                z = (peak_height - background) / (noise + e)
                z_scores.append(z)

            # add z-score to hand labels
            label_peaks['Peak Amplitude (ZScore)'] = z_scores
            # save new hand labels
            label_peaks.to_csv(l, sep=",", index=False)