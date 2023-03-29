'''
HRFI target detection.

Calling it tracker for self-consistency with other modules, 
however time is not currently factored into tracking
'''
import json
import os
import logging

import numpy              as np

from skimage.filters      import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure      import label, regionprops
from skimage.morphology   import closing, square

from utils.file_manipulation import read_image

def hrfi_tracker(hrfi_filepath, experiment, config):

    rgb = read_image(hrfi_filepath, config["raw_hologram_resolution"])
    
    # Verify band weights in config
    sum_band_w = np.round(config["instrument"]["red_band_weight"] + \
                          config["instrument"]["green_band_weight"] + \
                          config["instrument"]["blue_band_weight"], 3)

    if sum_band_w != 1.0:
        logging.warning(f"Instrument band weights don't sum to 1 ({sum_band_w}), normalizing.")
        config["instrument"]['red_band_weight'] /= sum_band_w
        config["instrument"]['green_band_weight'] /= sum_band_w
        config["instrument"]['blue_band_weight'] /= sum_band_w

    # Reduce 3 banded image to 1, with band weights being set by detector calibration
    gray = ((config["instrument"]["red_band_weight"]   * rgb[:,:,0]) + 
            (config["instrument"]["green_band_weight"] * rgb[:,:,1]) + 
            (config["instrument"]["blue_band_weight"]  * rgb[:,:,2]))

    # Scale data 0 to 1
    gray = np.clip(gray, 0, config["instrument"]["max_dn"])
    gray = gray / config["instrument"]["max_dn"]

    # Determine threshold
    t1 = threshold_otsu(gray)
    t2 = np.percentile(gray, config["instrument"]["min_perc"])
    t = max(t1, t2)

    # Determine connected regions of whitespace and mark them as potential critters
    image = np.zeros((rgb.shape[0],rgb.shape[1]))
    image[gray > t] = 1
    bw = closing(image==1, square(3))
    cleared = clear_border(bw)

    label_image = label(cleared)

    # Filter to only save organisms larger than min_organism_size pixels
    bboxes = []
    [bboxes.append(region.bbox) for region in regionprops(label_image) if region.area >= config["tracker"]["min_bbox_area"] and region.area <= config["tracker"]["max_bbox_area"]]

    track_folder = os.path.join(experiment, config['experiment_dirs']['track_dir'])
    if not os.path.exists(track_folder):
        os.makedirs(track_folder)

    file_tag = hrfi_filepath.split("/")[-1].split(".")[0]
    with open(os.path.join(track_folder, f"{file_tag}_bboxes.json"), 'w') as f:
        json.dump(bboxes, f, indent=2)
    
