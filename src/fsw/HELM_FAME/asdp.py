'''
Autonomous Science Data Product (ASDP) generation library
'''
import os
import csv
import logging
import glob
import yaml
import shutil
import json

import os.path           as op
import numpy             as np
import matplotlib.pyplot as plt

from pathlib       import Path
from PIL           import Image
from PIL           import ImageDraw
from math          import ceil, floor
from tqdm          import tqdm

from utils.track_loaders     import load_json
from utils.file_manipulation import tiff_read
from utils.dir_helper        import get_exp_subdir


def rehydrate_mugshots(experiment, config):
    """Place mugshots on a black, or downlinked, background based on track data"""

    mugshot_dir = op.join(get_exp_subdir('asdp_dir', experiment, config), 'mugshots')
    rehydrated_dir = op.join(get_exp_subdir('asdp_dir', experiment, config), 'rehydrated')
    mugshot_list = list(glob.glob(op.join(mugshot_dir, '*.png')))
    tracks = list(glob.glob(os.path.join(experiment, config['experiment_dirs']['predict_dir'],"*.json")))

    # Read total frames and frame resolution from validate step processing report
    #   TODO - consider moving processing report to EVRs and centralize flight/ground knowledge transfer through EVRs
    report_path = list(glob.glob(os.path.join(experiment, config['experiment_dirs']['validate_dir'],"*processing_report.txt")))
    with open(report_path[0]) as fp:
        report_lines = fp.readlines()
    total_frames = int(report_lines[0].split()[-1])
    recon_row = int(report_lines[1].split()[-1])
    recon_col = int(report_lines[1].split()[-3])

    if not op.exists(rehydrated_dir):
        os.makedirs(rehydrated_dir)

    files = mugshot_list
    files = [x.split("/")[-1] for x in files]

    if files:
        params = [x.rstrip(".png").split("_") for x in files]
        params = np.stack(params)
        params = params.astype(float)
        num_tracks = int(np.max(params[:,6]))

    for x in tqdm(range(0, total_frames)):

        try:
            background_image_path = list(glob.glob(os.path.join(experiment, config['experiment_dirs']['validate_dir'],"*_median_image.tif")))[0]
        except:
            background_image_path = None

        if background_image_path is not None:
            base = Image.open(background_image_path)
            base = base.resize((recon_row, recon_col))
            base = np.asarray(base)
            base_rows, base_cols = base.shape
        else:
            base = np.zeros((recon_row, recon_col), dtype=np.uint8)
            base_rows = recon_row
            base_cols = recon_col

        base = np.repeat(base[:, :, np.newaxis], 3, axis=2)
        img = Image.fromarray(base)
        
        # Draw the frame number in the top left
        draw = ImageDraw.Draw(img)
        draw.text((25, 25), f"Frame {x}", fill='white')
        base = np.array(img)
        img = Image.fromarray(base)

        for track in tracks:

            with open(track, 'r') as f:
                d = json.load(f)

                times = np.asarray(d["Times"])
                if x in times:

                    track_index = int(track.split("/")[-1].rstrip(".json"))
                    tmp = list(glob.glob(f'{mugshot_dir}/*_*_*_*_*_*_*_{track_index}_*_*_*_*.png'))
                    tmp = [x.split("/")[-1] for x in tmp]
                    params = [x.rstrip(".png").split("_") for x in tmp]

                    if params:

                        params = np.stack(params)
                        params = params.astype(float)
                        distance = abs(params[:,2] - x)
                        f = tmp[np.argmin(distance)]

                        index = np.where(times == x)[0][0]

                        scale = np.array((params[0][10],params[0][11]))
                        position = np.asarray(d["Particles_Estimated_Position"] * scale)
                        pos = position[index]

                        row = int(round(pos[0]))
                        col = int(round(pos[1]))

                        snapshot = Image.open(os.path.join(mugshot_dir,f)) 
                        snapshot = np.array(snapshot)
                        if snapshot.ndim == 2:
                            rows, cols = snapshot.shape
                            bands = 1
                        else:
                            rows, cols, bands = snapshot.shape
                        
                        row_min = ceil(row - (rows/2))
                        row_max = ceil(row + (rows/2))
                        col_min = ceil(col - (cols/2))
                        col_max = ceil(col + (cols/2))

                        if row_min < 0:
                            row_min = 0
                        if col_min < 0:
                            col_min = 0

                        if row_max > base_rows:
                            row_max = base_rows
                        if col_max > base_cols:
                            col_max = base_cols

                        row_size = row_max - row_min
                        col_size = col_max - col_min
                        
                        if bands == 1:
                            base[row_min:row_max, col_min:col_max, 0] = snapshot[:row_size, :col_size]
                            base[row_min:row_max, col_min:col_max, 1] = snapshot[:row_size, :col_size]
                            base[row_min:row_max, col_min:col_max, 2] = snapshot[:row_size, :col_size]
                        else:
                            base[row_min:row_max, col_min:col_max, :] = snapshot[:row_size, :col_size,:]
                        

                        img = Image.fromarray(base)
                        draw = ImageDraw.Draw(img)
                        draw.rectangle([(col_min, row_min), (col_max, row_max)], outline='#00ffff')
                        draw.text((col_min+2, row_min+2), f"{track_index}", fill='white')
                        base = np.array(img)

        img.save(os.path.join(rehydrated_dir,'{x:05}.png'.format(x=x)))  

    return total_frames

def get_track_classification_counts(track_fpaths):
    """Tally up number of motile, non-motile, and other tracks"""

    tracks = [load_json(fpath) for fpath in track_fpaths]

    n_motile = 0
    n_non_motile = 0
    other = 0

    # Calculate number of motile/non_motile tracks
    for track in tracks:
        if track['classification'] == "motile":
            n_motile += 1
        elif track['classification'] == "non-motile":
            n_non_motile += 1
        elif track['classification'] in ["ambiguous", "other"]:
            other += 1
        else:
            logging.warning('Motility type not understood. Got: "%s"',
                            track['classification'])

    return n_motile, n_non_motile, other

def calculate_mugshot_crop_bounds(bbox, mugshot_width, mugshot_radius, padding, positions, size):
    """ Determine particle bounding box based on cropping/padding configuration

    Parameters
    ----------
    bbox: numpy array
        Tracker calculated particle bounding box
    mugshot_width: int
        Configuration key fixed_width. Declares fixed radius if >0, 
        if 0 the box is dynamic based on tracker bbox
    mugshot_radius: int
        Radius of the box
    padding: int
        Configuration key padding.  Extra radius padding when mugshot_width == 0 (dynamic)
    positions: numpy array
        Position of the bounding box in the frame
    size: int
        Volume of the particle, as calculated by the tracker

    Returns
    -------
    row_min: int
        Minimum row for the particle bounding box
    col_min: int
        Minimum column for the particle bounding box
    row_max: int
        Maximum row for the particle bounding box
    col_max: int
        Maximum column for the particle bounding box

    """

    # Define scale from tracker bbox to original resolution
    mugshot_bbox = np.array(bbox)
    if mugshot_width == 0 and size:
        row_min = int(np.round(mugshot_bbox[0,0]) - padding)
        row_max = int(np.round((mugshot_bbox[0,0] + mugshot_bbox[1,0])) + padding)
        col_min = int(np.round(mugshot_bbox[0,1]) - padding)
        col_max = int(np.round((mugshot_bbox[0,1] + mugshot_bbox[1,1])) + padding)
    else:
        pos = np.array(positions)
        row = int(np.round(pos[0]))
        col = int(np.round(pos[1]))
        row_min = row - mugshot_radius
        row_max = row + mugshot_radius
        col_min = col - mugshot_radius
        col_max = col + mugshot_radius

    return row_min, col_min, row_max, col_max

def mugshots(experiment, holograms, name, output_dir, config):
    """ Create and save experiment mugshot crops

    Parameters
    ----------
    experiment: str
        Path to experiment directory
    holograms: list
        List of hologram file paths
    name: str
        Experiment name
    output_dir: str
        Directory to save results in
    config: dict
        Pipeline config dictionary

    Returns
    -------
    None
    """

    # width of a mugshot crop if enforced
    mugshot_width = config['mugshot']['fixed_width']
    mugshot_radius = mugshot_width // 2
    padding = config['mugshot']['padding']

    # shape of track, probably downsized
    track_shape = np.array(config['preproc_resolution'])

    # original resolution of hologram, enforced in validate
    holo_shape = np.array(config['raw_hologram_resolution'])

    # scaling factor for mugshot coordinates, ignore channel #
    mugshot_scale = holo_shape[:2] / track_shape[:2]

    # set maximum number of mugshots per track
    if config['mugshot']['max_pertrack'] == 0:
        # if 0, no maximum
        max_pertrack = np.inf
    else:
        max_pertrack = config['mugshot']['max_pertrack'] - 1

    # output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # tracks and predictions
    predict_jsons = glob.glob(os.path.join(experiment, config.get("experiment_dirs").get('predict_dir'), "*.json"))
    
    unique_count = 0
    for track_file in predict_jsons:

        saved_mugshots = 0

        track = load_json(track_file)

        sizes = track["Particles_Size"]
        bbox = [x * mugshot_scale if type(x) is list else None for x in track["Particles_Bbox"]]
        classification = track["classification"]
        positions = [x * mugshot_scale for x in track["Particles_Estimated_Position"]]
        times = track["Times"]
        track_id = track["Track_ID"]
        num_samples = len(sizes)

        # Sort smallest to largest particle size, ignoring NaNs
        particle_sizes = track["Particles_Size"]
        for s in range(0, len(sizes)):

            # Determine mugshot dimensions and frame position
            row_min, col_min, row_max, col_max = calculate_mugshot_crop_bounds(bbox[s], mugshot_width, mugshot_radius, padding, positions[s], sizes[s]) 

            # Check if mugshot dimensions intersect with frame edge.  If so, exclude from size ranking.
            if bbox[s] is not None:
                if bbox[s][0][0] <= mugshot_radius or \
                   bbox[s][0][0] + bbox[s][1][0] >= holo_shape[0] or \
                   bbox[s][0][1] <= mugshot_radius or \
                   bbox[s][0][1] +  bbox[s][1][1] >= holo_shape[1] or \
                   row_min <= 0 or \
                   col_min <= 0 or \
                   row_max >= config["raw_hologram_resolution"][0] or \
                   col_max >= config["raw_hologram_resolution"][1]:
                     particle_sizes[s] = None
                     sizes[s] = None  
                
        particle_sizes = [i for i in particle_sizes if i] # Remove Nones
        particle_sizes.sort()

        if classification == "motile":
            for x in range(0, num_samples):
                t = times[x]
                
                # Find priority of this particle's size
                size = sizes[x]
                if size:
                    size_score = particle_sizes.index(size)
                else:
                    # NaN size, never mugshot
                    continue

                # If particle size is smaller than priority cutoff, mugshot
                if size_score <= max_pertrack:
                    img = tiff_read(holograms[t])
                    if img is None:
                        logging.warning(f"Failed to generate mugshot: {holograms[t]}")
                        continue

                    # Determine mugshot dimensions and frame position
                    row_min, col_min, row_max, col_max = calculate_mugshot_crop_bounds(bbox[x], mugshot_width, mugshot_radius, padding, positions[x], size)  

                    # If mugshot dimensions extend beyond frame, stop cropping at frame edge.
                    if row_min <= 0:
                        row_min = 0
                    if col_min <= 0:
                        col_min = 0
                    if row_max >= config["raw_hologram_resolution"][0]:
                        row_max = config["raw_hologram_resolution"][0] - 1
                    if col_max >= config["raw_hologram_resolution"][1]:
                        col_max = config["raw_hologram_resolution"][1] - 1

                    # Mugshot crop and save
                    snapshot = img[row_min:row_max, col_min:col_max]
                    row_width, col_width = snapshot.shape[:2]

                    if row_width >= 1 and col_width >= 1:
                        if len(snapshot.shape) == 2:
                            outImg = Image.fromarray(snapshot, 'L')
                        elif len(snapshot.shape) == 3:
                            outImg = Image.fromarray(snapshot, 'RGB')

                        # Filename Convention:
                        # original rows - number of rows in original hologram
                        # original cols - number of columns in original hologram
                        # time - hologram index in time
                        # col left - top/left of bounding box in column space
                        # row top - top/left of bounding box in row space
                        # box row width - number of rows in bounding box
                        # box col width - number of cols in bounding box
                        # track ID - which track this mugshot is associated with
                        # size score - track relative index of mugshot size ranking
                        # unique count - absolute counter of all mugshots associated with this experiment.  Guarantees uniqueness
                        # row scale modifier for temporary 1024->2048 resolution issue
                        # col scale modifier for temporary 1024->2048 resolution issue
                        outImg.save(os.path.join(output_dir,f'{holo_shape[0]}_{holo_shape[1]}_{t}_{row_min}_{col_min}_{row_width}_{col_width}_{track_id}_{size_score}_{unique_count}_{int(mugshot_scale[0])}_{int(mugshot_scale[1])}.png'))
                        unique_count += 1
                        saved_mugshots += 1

                        if saved_mugshots == config['mugshot']['max_pertrack']:
                            break

def generate_SUEs(experiment_dir, asdp_dir, track_fpaths, sue_config):
    """Create and save a science utility for a HELM experiment

    Parameters
    ----------
    experiment_dir: str
        Path to experiment directory
    asdp_dir: str
        Path to ASDP directory where outputs will be saved
    track_fpaths: list
        List of an experiment's track files to compute SUE for
    sue_config: dict
        Subset of HELM config parameters relevent to the SUE calculation. Used
        to pull the desired weights and extrema.

    Returns
    -------
    sue: float
        Science Utility Estimate
    """

    n_motile, _, _ = get_track_classification_counts(track_fpaths)

    # Generate SUE vector and pull weights from config
    sue_vec = np.array([n_motile / sue_config['extrema']['n_motile']])
    sue_weights = np.array([sue_config['weights']['n_motile']])

    if np.sum(sue_weights) != 1.:
        logging.warning('Sum of SUE weights != 1. May lead to SUE that does not lie on interval [0, 1]')

    # Clip SUE vector between 0 and 1, and compute weighted average
    sue_clipped = np.clip(sue_vec, 0, 1)
    sue = np.round(np.sum(np.multiply(sue_clipped, sue_weights)), 3)

    # Write SUE to CSV and return value
    exp_name = Path(experiment_dir).name
    sue_csv_fpath = op.join(asdp_dir, f'{exp_name}_sue.csv')

    with open(sue_csv_fpath, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['SUE'])
        writer.writeheader()
        writer.writerow({'SUE': sue})

    return sue


def generate_DDs(experiment_dir, asdp_dir, track_fpaths, dd_config):
    """ Create and save a diversity descriptor for a HELM experiment

    Parameters
    ----------
    experiment_dir: str
        Path to experiment directory
    asdp_dir: str
        Path to ASDP directory where outputs will be saved
    track_fpaths: list
        List of an experiment's track files to compute DD for
    dd_config: dict
        Subset of HELM config parameters relevent to the DD calculation. Used
        to pull the desired weights and extrema.

    Returns
    -------
    dd: float
        Diversity Descriptor
    """

    # Get raw inputs to Diversity Descriptor
    n_motile, n_non_motile, n_other = get_track_classification_counts(track_fpaths)

    raw_dd = {'n_motile': n_motile,
              'n_non_motile': n_non_motile,
              'n_other': n_other}
    dd_vals = {}

    weight_sum = 0  # Keep track of total weighting
    for key in raw_dd:
        # Compute DD on [0, 1] interval
        clipped_dd_val = np.clip((raw_dd[key] / dd_config['extrema'][key]), 0, 1)
        # Weight DD and store
        dd_vals[key] = np.round(clipped_dd_val * dd_config['weights'][key], 3)

        # Sum each weight component
        weight_sum += dd_config['weights'][key]

    if weight_sum != 1.:
        logging.warning('Sum of DD weights != 1. May lead to DD that does not lie on interval [0, 1]')

    # Write DD to CSV and return value
    exp_name = Path(experiment_dir).name
    dd_csv_fpath = op.join(asdp_dir, f'{exp_name}_dd.csv')

    with open(dd_csv_fpath, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dd_vals.keys())
        writer.writeheader()
        writer.writerow(dd_vals)

    return dd_vals