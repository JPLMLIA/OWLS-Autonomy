'''
HRFI target detection.

Calling it tracker for self-consistency with other modules, 
however time is not currently factored into tracking
'''
import json
import os

import numpy              as np

from skimage.filters      import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure      import label, regionprops
from skimage.morphology   import closing, square
from skimage              import color

from utils.file_manipulation import tiff_read

def hrfi_tracker(hrfi_filepath, experiment, config):

    rgb = tiff_read(hrfi_filepath)
    gray = color.rgb2gray(rgb)

    # Convert to grayscale and apply threshold to binary
    image = np.zeros((rgb.shape[0],rgb.shape[1]))
    image[gray > config["binary_threshold"]] = 1

    # Determine connected regions of whitespace and mark them as potential critters
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(3))
    cleared = clear_border(bw)

    label_image = label(cleared)

    # Filter to only save organisms larger than min_organism_size pixels
    bboxes = []
    [bboxes.append(region.bbox) for region in regionprops(label_image) if region.area >= config["min_organism_size"]]

    track_folder = os.path.join(experiment, config['experiment_dirs']['track_dir'])
    if not os.path.exists(track_folder):
        os.makedirs(track_folder)

    file_tag = hrfi_filepath.split("/")[-1].split(".")[0]
    with open(os.path.join(track_folder, f"{file_tag}_bboxes.json"), 'w') as f:
        json.dump(bboxes, f, indent=2)
    
