import os
import os.path as op
import cv2
import csv
import json
import logging
import numpy              as np
import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches

from skimage.color        import label2rgb
from PIL                  import Image

from skimage.io._plugins.pil_plugin import ndarray_to_pil
from utils.file_manipulation        import tiff_read

def mugshots(hrfi_filepath, experiment, config):

    padding = config["mugshot_padding"]
    rgb = tiff_read(hrfi_filepath)

    # Reduce 3 banded image to 1, with band weights being set by detector calibration
    gray = ((config["instrument"]["red_band_weight"]   * rgb[:,:,0]) + 
            (config["instrument"]["green_band_weight"] * rgb[:,:,1]) + 
            (config["instrument"]["blue_band_weight"]  * rgb[:,:,2]))

    # Scale data 0 to 1
    gray = gray / config["instrument"]["max_dn"]
    asdp_dir = os.path.join(experiment, config['experiment_dirs']['asdp_dir'])
    mugshot_dir = os.path.join(asdp_dir, "mugshots")
    if not os.path.exists(mugshot_dir):
        os.makedirs(mugshot_dir)

    track_folder = os.path.join(experiment, config['experiment_dirs']['track_dir'])
    file_tag = hrfi_filepath.split("/")[-1].split(".")[0]

    with open(os.path.join(track_folder, f"{file_tag}_bboxes.json"), 'r') as f:
        label_image = json.load(f)

    mugshot_count = 0
    for region in label_image:
        minr = max(region[0] - padding, 0)
        minc = max(region[1] - padding, 0)
        maxr = region[2] + padding
        maxc = region[3] + padding
        mugshot = rgb[minr:maxr,minc:maxc,:]
        im = Image.fromarray(mugshot)
        im.save(os.path.join(mugshot_dir, f"{file_tag}_{mugshot_count}.tif"))

        ### Binary ASDP ###
        
        # Make a subdir called binary_mugshots to store binary mask mugshot ASDPs
        binary_mugshot_dir = os.path.join(asdp_dir, "binary_mugshots")
        if not os.path.exists(binary_mugshot_dir):
            os.makedirs(binary_mugshot_dir)

        # Separate each band
        red_band = mugshot[:,:,0]
        green_band = mugshot[:,:,1]
        blue_band = mugshot[:,:,2]

        # Threshold with each band activiation DN
        red_band_mask = red_band > config["asdp"]["r_activation_thresh"]
        green_band_mask = green_band > config["asdp"]["g_activation_thresh"]
        blue_band_mask = blue_band > config["asdp"]["b_activation_thresh"]

        # Combine binary masks
        binary_mugshot = np.stack((red_band_mask, green_band_mask, blue_band_mask),axis=2)
        
        # TODO - still stored in a regular bit-depth image for convienence, can write binary version later
        binary_im = ndarray_to_pil(binary_mugshot).convert("1")
        binary_im.save(os.path.join(binary_mugshot_dir, f"{file_tag}_{mugshot_count}_binary.tif"))


        ### Ellipse ASDP ####

        # Make subdir called ellipses to store ellipse ASDPs
        ellipse_dir = os.path.join(asdp_dir, "ellipses")
        if not os.path.exists(ellipse_dir):
            os.makedirs(ellipse_dir)

        gray = cv2.cvtColor(mugshot, cv2.COLOR_BGR2GRAY)

        output = cv2.bitwise_and(mugshot, mugshot)

        ret, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        output_ellipses = []
        if len(contours) != 0:

            c = max(contours, key = cv2.contourArea)
            if len(c) > 5:
                ellipse = cv2.fitEllipse(c)
                output_ellipses.append(ellipse)
                cv2.ellipse(output, ellipse, (0,0,255))

        # Save graphic of ellipse overlaid on mugshot (TODO - add field mode and exclude)
        frame1 = plt.imshow(output)
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        plt.savefig(os.path.join(ellipse_dir,f"{file_tag}_{mugshot_count}_hull.png"))
        plt.close()

        # Stored as a list of ellipse tuples
        with open(op.join(ellipse_dir, f"{file_tag}_{mugshot_count}_hull.json"), 'w') as f:
            json.dump(output_ellipses, f)

        ### Contours / Hull ASDP ###

        # Make a subdir called contours to store contour / hull ASDPs
        contour_dir = os.path.join(asdp_dir, "contours")
        if not os.path.exists(contour_dir):
            os.makedirs(contour_dir)

        gray = cv2.cvtColor(mugshot, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hulls = []
        for i in range(len(contours)):
            hulls.append(cv2.convexHull(contours[i], False))

        drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
        for i in range(len(contours)):
            color_contours = (0, 255, 0) # green - color for contours
            draw_color = (255, 0, 0) # blue - color for convex hull
            cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
            cv2.drawContours(drawing, hulls, i, draw_color, 1, 8)

        # Save graphic of hulls and contours on mugshot (TODO - add field mode and exclude)
        frame1 = plt.imshow(drawing)
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        plt.savefig(os.path.join(contour_dir,f"{file_tag}_{mugshot_count}_contour.png"))
        plt.close()

        # Stored as a dictionary with hull and contour lists
        output_dictionary = {"contours":[c.tolist() for c in contours], 
                             "hulls":[h.tolist() for h in hulls]}
        with open(op.join(contour_dir, f"{file_tag}_{mugshot_count}_contour.json"), "w") as f:
            json.dump(output_dictionary, f)

        mugshot_count += 1

    ### Mugshot Context Image ###

    if not config['_field_mode']:
        
        context_dir = os.path.join(asdp_dir, "context")
        if not os.path.exists(context_dir):
            os.makedirs(context_dir)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(rgb)
        for region in label_image:
            minr = region[0] - padding
            minc = region[1] - padding
            maxr = region[2] + padding
            maxc = region[3] + padding

            mugshot = rgb[minr:maxr,minc:maxc,:]

            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)

        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(os.path.join(context_dir,f"{file_tag}_mugshot_context.png"))
        plt.close()

def generate_SUEs_DDs(asdp_dir, mugshot_id, sue_config, dd_config):
    """ Create and save the science utility estimate and diversity descriptor 
    for a HiRAILS mugshot

    Parameters
    ----------
    asdp_dir: str
        Path to ASDP directory where outputs will be saved
    track_fpaths: list
        List of an experiment's files to compute DD for
    sue_config: dict
        Subset of HiRAILS config parameters relevant to the SUE caslculation.
        Used to pull the desired weights and extrema.
    dd_config: dict
        Subset of HiRAILS config parameters relevant to the DD calculation.
        Used to pull the desired weights and extrema.

    Returns
    -------
    dd: float
        Diversity Desciptor
    """
    ## ASDP products to read
    # Get RGB mugshot filepath
    mugshot_fp = op.join(asdp_dir, 'mugshots', f'{mugshot_id}.tif')
    # Get binary mugshot filepath
    binary_fp = op.join(asdp_dir, 'binary_mugshots', f'{mugshot_id}_binary.tif')
    # Get contour filepath
    contour_fp = op.join(asdp_dir, 'contours', f'{mugshot_id}_contour.json')
    # Get ellipse filepath
    ellipse_fp = op.join(asdp_dir, 'ellipses', f'{mugshot_id}_hull.json')
    
    ## FILL PERC AND PERIMETER RATIO
    # Open binary mugshot
    binary = np.array(Image.open(binary_fp)).astype(bool)
    # Open contour
    with open(contour_fp, 'rb') as f:
        data = json.load(f)
        contours = [np.array(c, dtype=np.int32) for c in data['contours']]
        hulls = [np.array(h, dtype=np.int32) for h in data['hulls']]

    if len(contours) != 0:
        # Keep largest contour
        contour_areas = [cv2.contourArea(c) for c in contours]
        contour = contours[np.argmax(contour_areas)]
        hull = hulls[np.argmax(contour_areas)]

        if cv2.arcLength(hull, True) != 0:
            perim_ratio = cv2.arcLength(contour,True) / cv2.arcLength(hull,True)
        else:
            perim_ratio = 0

        contour_area = np.max(contour_areas)
        
        # Get only pixels in largest contour
        mask = np.zeros(binary.shape, np.uint8)
        cv2.drawContours(mask, [contour], 0, 1, -1)
        # Calculate fill % of pixels within contour
        pix_incontour = np.sum(binary * mask)
        fill_perc = pix_incontour / contour_area
    else:
        perim_ratio = 0
        fill_perc = 0
    
    ## ECCENTRICITY
    # Open ellipse
    with open(ellipse_fp, 'rb') as g:
        ellipse = json.load(g)
    
    if len(ellipse) != 0:
        ellipse = ellipse[0]
        # Semimajor and semiminor axes for ecc calc
        semimajor = ellipse[1][1] / 2
        semiminor = ellipse[1][0] / 2
        eccentricity = np.sqrt(1 - (semiminor**2)/(semimajor**2))
    else:
        eccentricity = 0

    ## CHANNEL BRIGHTNESS
    # Open mugshot
    mugshot = np.array(Image.open(mugshot_fp))
    # (row, col, RGB)
    red_99 = np.percentile(mugshot[:,:,0], 99)
    green_99 = np.percentile(mugshot[:,:,1], 99)
    blue_99 = np.percentile(mugshot[:,:,0], 99)

    ## CALCULATE SUE
    # Generate SUE vector and pull weights from config
    sue_vec = np.array([
        fill_perc / sue_config['extrema']['fill_perc'],
        perim_ratio / sue_config['extrema']['perim_ratio'],
        eccentricity / sue_config['extrema']['eccentricity'],
        red_99 / sue_config['extrema']['red_99th'],
        blue_99 / sue_config['extrema']['blue_99th'],
        green_99 / sue_config['extrema']['green_99th']
    ])
    sue_weights = np.array([
        sue_config['weights']['fill_perc'],
        sue_config['weights']['perim_ratio'],
        sue_config['weights']['eccentricity'],
        sue_config['weights']['red_99th'],
        sue_config['weights']['blue_99th'],
        sue_config['weights']['green_99th']
    ])

    if np.sum(sue_weights) != 1.:
        logging.warning('Sum of SUE weights != 1. May lead to SUE that does not lie on interval [0, 1]')

    # Clip SUE vector between 0 and 1, and compute weighted average
    sue_clipped = np.clip(sue_vec, 0, 1)
    sue = np.round(np.dot(sue_clipped, sue_weights), 3)

    # Write SUE to CSV and return value
    output_dir = op.join(asdp_dir, 'SUEs')
    if not op.exists(output_dir):
        os.mkdir(output_dir)
    sue_csv_fpath = op.join(output_dir, f'{mugshot_id}_sue.csv')

    with open(sue_csv_fpath, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['SUE'])
        writer.writeheader()
        writer.writerow({'SUE': sue})


    ## DIVERSITY DESCRIPTOR
    raw_dd = {
        'fill_perc': fill_perc,
        'perim_ratio': perim_ratio,
        'eccentricity': eccentricity,
        'red_99th': red_99,
        'blue_99th': blue_99,
        'green_99th': green_99
    }
    dd_vals = {}

    weight_sum = 0 
    for key in raw_dd:
        # Compute DD on [0, 1] interval
        clipped_dd_val = np.clip((raw_dd[key] / dd_config['extrema'][key]), 0, 1)
        # Weight DD and store
        dd_vals[key] = np.round(clipped_dd_val * dd_config['weights'][key], 3)

        # Sum each weight component
        weight_sum += dd_config['weights'][key]

    if np.round(weight_sum, 3) != 1.:
        logging.warning('Sum of DD weights != 1. May lead to DD that does not lie on interval [0, 1]')

    output_dir = op.join(asdp_dir, 'DDs')
    if not op.exists(output_dir):
        os.mkdir(output_dir)
    dd_csv_fpath = op.join(output_dir, f'{mugshot_id}_dd.csv')

    with open(dd_csv_fpath, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dd_vals.keys())
        writer.writeheader()
        writer.writerow(dd_vals)

    return sue, dd_vals