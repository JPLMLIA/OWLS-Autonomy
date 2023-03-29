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
from scipy.stats import scoreatpercentile

from pathlib       import Path
from PIL           import Image
from PIL           import ImageDraw
from math          import ceil, floor
from tqdm          import tqdm

from utils.track_loaders     import load_json
from utils.file_manipulation import read_image
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

def get_track_classification_counts(track_fpaths, probabilities=False):
    """Tally up number of motile, non-motile, and other tracks

    Parameters
    ----------
    track_fpaths: list
        list of paths to track JSON file paths
    probabilities: bool (optional, default: False)
        return a list of probabilities instead of counts

    Returns
    -------
    n_motile: int
    n_non_motile: int
    other: int
        track classification counts

    or

    probabilities: list
        list of track motility probabilities
    """

    tracks = [load_json(fpath) for fpath in track_fpaths]

    if probabilities:
        return [t['probability_motility'] for t in tracks]

    n_motile = 0
    n_non_motile = 0

    # Calculate number of motile/non_motile tracks
    for track in tracks:
        if track['classification'] == "motile":
            n_motile += 1
        elif track['classification'] == "non-motile":
            n_non_motile += 1
        else:
            logging.warning('Motility type not understood. Got: "%s"',
                            track['classification'])

    return n_motile, n_non_motile

def get_track_info(track_fpaths):
    """Tally up number of motile, non-motile, and other tracks

    Parameters
    ----------
    track_fpaths: list
        list of paths to track JSON file paths

    Returns
    -------
    array
    """

    tracks = [load_json(fpath) for fpath in track_fpaths]

    output = []
    for t in tracks:
        p = float(t['probability_motility'])
        d = len(t['Times'])
        output.append([p, d])
    output = np.array(output)

    return output



def get_track_intensities(track_fpaths, track_statistic, by_id=False):
    """Get a summarized intensity value for each track

    Parameters
    ----------
    track_fpaths: list
        list of track JSON file paths
    track_statistic: function
        statistic function to apply to the track intensities
    by_id: bool (default: False)

    Returns
    -------
    list of 4-element arrays containing (gray, r, g, b) per-track summaries
    """
    tracks = [load_json(fpath) for fpath in track_fpaths]

    all_track_intensities = []
    all_track_ids = []

    for track in tracks:
        intensities = np.array([
            i for i in track.get('Particles_Max_Intensity', [])
            if i is not None
        ], dtype=float) / 255.

        if len(intensities) == 0:
            continue

        if len(intensities.shape) == 1:
            # Handle case where intensities are a single list
            intensities = intensities.reshape((-1, 1))

        if intensities.shape[1] == 1:
            intensities = np.column_stack(4 * [intensities])

        elif intensities.shape[1] == 3:
            intensities = np.column_stack([
                np.average(intensities, axis=1),
                intensities
            ])

        else:
            logging.warning(
                'Unknown intensity list with length %d',
                intensities.shape[1])

        all_track_intensities.append(track_statistic(intensities))
        all_track_ids.append(track['Track_ID'])

    if by_id:
        return dict(zip(all_track_ids, all_track_intensities))
    else:
        return all_track_intensities

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
                    img = read_image(holograms[t], config['raw_hologram_resolution'])
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
                        # row top - top/left of bounding box in row space
                        # col left - top/left of bounding box in column space
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


def generate_SUE_classic(track_fpaths, sue_config):
    """Calculate science utility using the "classic" approach (based on
    n_motile particles)

    Parameters
    ----------
    track_fpaths: list
        List of an experiment's track files to compute SUE for
    sue_config: dict
        Subset of HELM config parameters relevent to the classic SUE
        calculation. Used to pull the desired weights and extrema.

    Returns
    -------
    sue: float
        Science Utility Estimate
    """

    n_motile, _ = get_track_classification_counts(track_fpaths)

    # Generate SUE vector and pull weights from config
    sue_vec = np.array([n_motile / sue_config['extrema']['n_motile']])
    sue_weights = np.array([sue_config['weights']['n_motile']])

    if np.sum(sue_weights) != 1.:
        logging.warning('Sum of SUE weights != 1. May lead to SUE that does not lie on interval [0, 1]')

    # Clip SUE vector between 0 and 1, and compute weighted average
    sue_clipped = np.clip(sue_vec, 0, 1)
    sue = np.round(np.sum(np.multiply(sue_clipped, sue_weights)), 3)

    return sue

def generate_SUE_topk_timenorm(track_fpaths, sue_config, explen):
    """Calculate science utility using the topk timenorm method

    Parameters
    ----------
    track_fpaths: list
        List of an experiment's track files to compute SUE for
    sue_config: dict
        Subset of HELM config parameters relevent to the classic SUE
        calculation. Used to pull the desired weights and extrema.
    explen: int
        Length of the complete experiment

    Returns
    -------
    sue: float
        Science Utility Estimate
    """

    track_array = get_track_info(track_fpaths)
    topk = sue_config.get('topk', 5)
    if topk > track_array.shape[0]:
        k = track_array.shape[0]
    else:
        k = topk

    weighted = track_array[:, 0] * track_array[:, 1] / explen
    topk_vals = sorted(weighted)[-k:]
    sue = np.sum(topk_vals) / topk
    sue = np.clip(sue, 0, 1)

    return sue


def generate_SUE_topk_confidence(track_fpaths, sue_config):
    """Calculate a confidence-based SUE that estimates the likelihood that at
    least one of the top k most likely motile tracks is motile.

    Parameters
    ----------
    track_fpaths: list
        List of an experiment's track files to compute SUE for
    sue_config: dict
        Subset of HELM config parameters relevent to the SUE
        calculation. Used to pull the "topk" parameter.

    Returns
    -------
    sue: float
        Science Utility Estimate
    """

    # Get top k confidence values used
    topk = sue_config.get('topk', 3)

    probabilities = np.array(sorted(
        get_track_classification_counts(track_fpaths, probabilities=True)
    ))

    topk = min(topk, probabilities.size)

    if topk == 0:
        return 0.0
    else:
        return 1.0 - np.product(1 - probabilities[-topk:])


def generate_SUE_sum_confidence(track_fpaths, sue_config):
    """Calculate a confidence-based SUE that estimates the expected number of
    motile particles by averaging confidence values

    Parameters
    ----------
    track_fpaths: list
        List of an experiment's track files to compute SUE for
    sue_config: dict
        Subset of HELM config parameters relevent to the SUE
        calculation. No parameters are required for this SUE

    Returns
    -------
    sue: float
        Science Utility Estimate
    """

    max_sum = float(sue_config.get('max_sum', 50))

    probabilities = np.array(sorted(
        get_track_classification_counts(track_fpaths, probabilities=True)
    ))
    if probabilities.size == 0:
        return 0.0
    else:
        normed_sum = np.sum(probabilities) / max_sum
        return np.clip(normed_sum, 0.0, 1.0)


def intensity_statistic_function(function_name, function_params):
    """Return an intensity statistic function from name and parameters

    Parameters
    ----------
    function_name: string
        a valid statistic function name (maximum, median, minimum, percentile)
    function_params: dict
        dictionary of statistic function parameters (if applicable); currently,
        the percentile function requires a "percentile" parameter

    Returns
    -------
    function mapping an n-by-4 array of intensities to the summary statistic
    across the first (0th) axis; for an empty list of intensities, returns a
    zero vector of length 4
    """
    if function_name == 'maximum':
        base_function = np.max
    elif function_name == 'minimum':
        base_function = np.min
    elif function_name == 'median':
        base_function = np.median
    elif function_name == 'percentile':
        if 'percentile' not in function_params:
            logging.warning(
                'No percentile specified for percentile function; using 50')
        pc = function_params.get('percentile', 50.0)
        base_function = lambda x, **kwargs: scoreatpercentile(x, pc, **kwargs)
    else:
        logging.warning(
            'Unknown statistic "%s", using "maximum"', function_name)
        base_function = np.max

    def stat_f(x):
        if len(x) == 0:
            return np.zeros((4,))
        else:
            return base_function(x, axis=0)

    return stat_f


def generate_SUE_intensity(track_fpaths, sue_config):
    """Calculate an intensity-based SUE

    Parameters
    ----------
    track_fpaths: list
        List of an experiment's track files to compute SUE for
    sue_config: dict
        Subset of HELM config parameters relevent to the SUE
        calculation. Possible entries include weight_gray, weight_red,
        weight_green, weight_blue, and track_statistic, observation_statistic,
        track_statistic_params, and observation_statistic_params

    Returns
    -------
    sue: float
        Science Utility Estimate
    """
    track_statistic_name = sue_config.get('track_statistic', 'maximum')
    track_statistic_params = sue_config.get('track_statistic_params', {})
    observation_statistic_name = sue_config.get('observation_statistic', 'maximum')
    observation_statistic_params = sue_config.get('observation_statistic_params', {})

    track_statistic = intensity_statistic_function(
        track_statistic_name, track_statistic_params)
    observation_statistic = intensity_statistic_function(
        observation_statistic_name, observation_statistic_params)

    track_intensities = get_track_intensities(track_fpaths, track_statistic)
    intensities = observation_statistic(track_intensities)

    weights = np.array([
        sue_config.get('weight_gray', 0.0),
        sue_config.get('weight_red', 0.0),
        sue_config.get('weight_green', 0.0),
        sue_config.get('weight_blue', 0.0),
    ])

    # Normalize weights
    weight_sum = np.sum(weights)
    if weight_sum > 0: weights /= weight_sum

    return np.dot(intensities, weights)


def generate_SUEs(experiment_dir, asdp_dir, track_fpaths, config):
    """Create and save a science utility for a HELM experiment

    Parameters
    ----------
    experiment_dir: str
        Path to experiment directory
    asdp_dir: str
        Path to ASDP directory where outputs will be saved
    track_fpaths: list
        List of an experiment's track files to compute SUE for
    config: dict
        HELM configuration

    Returns
    -------
    sue: float
        Science Utility Estimate
    """
    sue_config = config['sue']
    method = sue_config['method']
    params = sue_config.get('params', {})
    if method == 'n_motile':
        sue = generate_SUE_classic(track_fpaths, params)
    elif method == 'topk_timenorm':
        rawdir = get_exp_subdir('hologram_dir', experiment_dir, config)
        explen = len(os.listdir(rawdir))
        sue = generate_SUE_topk_timenorm(track_fpaths, params, explen)
    elif method == 'topk_confidence':
        sue = generate_SUE_topk_confidence(track_fpaths, params)
    elif method == 'sum_confidence':
        sue = generate_SUE_sum_confidence(track_fpaths, params)
    elif method == 'intensity':
        sue = generate_SUE_intensity(track_fpaths, params)
    else:
        logging.warning(f'Unknown SUE method "{method}"; setting SUE = 0')
        sue = 0.0


    # Write SUE to CSV and return value
    exp_name = Path(experiment_dir).name
    sue_csv_fpath = op.join(asdp_dir, f'{exp_name}_sue.csv')

    with open(sue_csv_fpath, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['SUE'])
        writer.writeheader()
        writer.writerow({'SUE': sue})

    return sue


def extract_percentiles(feature_values, features, percentiles):
    """ Extracts the feature values at the given percentiles

    Parameters
    ----------
    feature_values: list
        list of dictionaries mapping feature names to values for each track
    features: list
        list of feature names for which percentiles will be computed
    percentiles: list
        list of percentiles to extract for each named feature

    Returns
    -------
    extracted_percentiles: list
        list of extracted percentiles for each feature
    """
    extracted_percentiles = []
    for feature, percentile in zip(features, percentiles):
        values = np.array(
            [row[feature] for row in feature_values],
            dtype=float
        )

        if len(values) == 0:
            # If no values for observation, just use 0.0
            pc_value = 0.0

        else:
            pc_value = scoreatpercentile(values, percentile)

        extracted_percentiles.append(pc_value)

    return extracted_percentiles


def apply_constrains_and_weights(values, extrema, weights):
    """ Applies a set of extrema and weights to a list of values

    Parameters
    ----------
    values: list
        list of input feature values
    extrema: list
        tuples containing the minimum and maximum allowed values for each
        feature; each value is normalized to [0, 1] within this range
    weights: list
        real-valued weight to multiply each feature; the sum of weights should
        equal 1.0, but this is not strictly required for the code to function
        correctly

    Returns
    -------
    updated_values: list
        list of values with extrema clipping and weights applied
    """

    updated_values = []
    for v, e, w in zip(values, extrema, weights):
        erange = e[1] - e[0]
        if erange == 0: erange = 1.0
        uval = w * np.clip((v - e[0]) / erange, 0.0, 1.0)
        updated_values.append(uval)

    return updated_values


def insert_intensity_features(features, track_fpaths):
    """
    Insert track intensity features into the list of existing features

    Parameters
    ----------
    features: list
        list of dicts containing the parsed features (must include track ids)
    track_fpaths: list
        list of track JSON path files

    Returns
    -------
    same list of features with inserted intensity values
    """

    # Get statistic functions
    stat_names = ('minimum', 'median', 'maximum')
    stat_abbv = ('min', 'med', 'max')
    stat_f = [
        intensity_statistic_function(n, {})
        for n in stat_names
    ]

    # Get intensity values for each statistic
    intensity_values = [
        get_track_intensities(track_fpaths, stat, by_id=True)
        for stat in stat_f
    ]
    colors = ('gray', 'red', 'green', 'blue')

    # For each track, insert intensity features
    for fv in features:
        track_id = int(fv['track'])
        for abbv, idict in zip(stat_abbv, intensity_values):
            ivals = idict.get(track_id, np.zeros((4,)))
            for c, v in zip(colors, ivals):
                fname = f'{c}_{abbv}_max_intensity'
                fv[fname] = v

    return features


def generate_DDs(experiment_dir, asdp_dir, track_fpaths, dd_config, feature_file):
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
    feature_file: str
        Path to feature file to read extracted track features

    Returns
    -------
    dd: dict
        Dictionary mapping Diversity Descriptor names to float values
    """

    feature_set = dd_config.get('include', [])
    percentile_conf = dd_config.get('percentiles', {})
    weight_conf = dd_config.get('weights', {})
    extrema_conf = dd_config.get('extrema', {})

    # Build list of expanded features, skipping any misspecified
    # features/percentiles/weights
    expanded_features = []
    for feature in feature_set:

        # Check for existence of config entries
        if feature not in percentile_conf:
            logging.warning(f'Percentiles not specified for feature "{feature}"')
            continue
        if feature not in weight_conf:
            logging.warning(f'Weights not specified for feature "{feature}"')
            continue
        if feature not in extrema_conf:
            logging.warning(f'Extrema not specified for feature "{feature}"')
            continue

        # Check for correct config type
        if type(percentile_conf[feature]) != list:
            logging.warning(f'Percentiles for "{feature}" must be a list')
            continue
        if type(weight_conf[feature]) != list:
            logging.warning(f'Weights for "{feature}" must be a list')
            continue
        if type(extrema_conf[feature]) != list:
            logging.warning(f'Extrema for "{feature}" must be a list')
            continue

        # Check for compatible percentiles/weights
        if len(percentile_conf[feature]) != len(weight_conf[feature]):
            logging.warning(f'Lengths differ for "{feature}" percentiles and weights')
            continue

        # Check for appropriate extrema config
        if len(extrema_conf[feature]) != 2:
            logging.warning(f'Extrema config for "{feature}" must have two entries')
            continue

        if extrema_conf[feature][0] >= extrema_conf[feature][1]:
            logging.warning(f'Extrema config for "{feature}" has zero range')
            continue

        percentiles = percentile_conf[feature]
        pc_suffixes = [('pc%.1f' % pc) for pc in percentiles]
        if len(set(pc_suffixes)) != len(pc_suffixes):
            logging.warning(f'Redundant percentiles after precision loss for "{feature}"')
            continue

        iterable = zip(percentiles, pc_suffixes, weight_conf[feature])
        for pc, suffix, weight in iterable:
            name = feature + '_' + suffix
            expanded_features.append((feature, name, pc, weight, extrema_conf[feature]))

    dd_vals = {}

    if len(expanded_features) > 0:

        features, names, percentiles, weights, extrema = list(zip(*expanded_features))

        # Normalize weights
        weights = np.array(weights)
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            logging.warning('Sum of DD weights is zero')
        else:
            weights /= weight_sum

        with open(feature_file, 'r') as f:
            reader = csv.DictReader(f)
            feature_values = list(reader)

        feature_values = insert_intensity_features(feature_values, track_fpaths)

        # Get DD values from percentiles
        extracted_values = extract_percentiles(feature_values, features, percentiles)

        # Apply constraints and weights
        updated_values = apply_constrains_and_weights(extracted_values, extrema, weights)

        dd_vals.update({ n:v for n, v in zip(names, updated_values) })

    # Write DD to CSV and return value
    exp_name = Path(experiment_dir).name
    dd_csv_fpath = op.join(asdp_dir, f'{exp_name}_dd.csv')

    with open(dd_csv_fpath, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dd_vals.keys())
        writer.writeheader()
        writer.writerow(dd_vals)

    return dd_vals
