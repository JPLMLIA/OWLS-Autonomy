"""
Functionality to simulate a hologram from a track
"""
import os.path as op
import json
import glob
import logging

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
from tqdm import tqdm
from astropy.convolution import Gaussian2DKernel, AiryDisk2DKernel

from helm_dhm.simulator.utils import create_dist_objs

logger = logging.getLogger(__name__)


def run_hologram_sim(config, exp_dir):
    """Generate hologram images from a config and exp_dir

    Parameters
    ----------
    config: dict
        HELM simulator configuration containing image, particle, experiment
        parameters.
    exp_dir: str
        Directory path to experiment. Simulated holograms and tracks will be
        saved here as subdirectories.
    """

    # Initializing some variables
    image_size = config['image_params']['resolution']
    depth = config['image_params']['chamber_depth']
    noise_params = config['image_params']['noise']
    noise_dist = create_dist_objs(**noise_params)[0]
    smooth_kernel_std = np.array(image_size) * noise_params['noise_smoothing_sigma']

    # Compute the furthest a particle could be from the focal plane
    if depth:
        max_focal_dist = np.max((config['image_params']['focus_plane'],
                                 depth - config['image_params']['focus_plane']))
    else:
        max_focal_dist = None

    # Load track dictionaries
    sim_track_dir = op.join(exp_dir, config['sim_track_dir'])
    sim_track_glob = op.join(sim_track_dir, '*' + config['track_ext'])
    track_fpaths = glob.glob(sim_track_glob)

    if not track_fpaths:
        logger.warning(f'No tracks found for glob string: "{sim_track_glob}"')
        return

    track_dicts = [load_track(fpath) for fpath in track_fpaths]

    # Determine if we need to simulate particle movement along depth dimension
    is_z_dim = track_dicts[0].get('Particles_Z_Dimension')

    # Find which particles exist at each timepoint
    active_particle_inds = []
    for time_val in range(0, config['exp_params']['n_frames']):
        track_incl_list = []
        for ti, td in enumerate(track_dicts):
            if time_val in td['Times']:
                track_incl_list.append(ti)
        active_particle_inds.append(track_incl_list)

    # Construct hologram images
    # XXX Could be handled with multiprocessing
    for frame_i, part_inds in tqdm(enumerate(active_particle_inds),
                                   total=config['exp_params']['n_frames'],
                                   desc='Simulating hologram frames'):
        frame_fpath = op.join(exp_dir, config['sim_hologram_dir'],
                              f'{frame_i:05}_holo.tif')
        frame = get_noise_frame(image_size, noise_dist, smooth_kernel_std)

        # Get each particle's location, appearance at current frame
        for part_ind in part_inds:
            t_index = track_dicts[part_ind]['Times'].index(frame_i)

            # Load individual particle info
            shape = track_dicts[part_ind]['Particle_Shape']
            size = track_dicts[part_ind]['Particle_Size']
            brightness = track_dicts[part_ind]['Particle_Brightness']
            row, col = np.around(track_dicts[part_ind]['Particles_Position'][t_index])
            if is_z_dim:
                zval = np.around(track_dicts[part_ind]['Particles_Z_Position'][t_index])
                focal_dist = np.abs(zval - config['image_params']['focus_plane'])
            else:
                focal_dist = 0


            # Get (normalized) kernal array
            kernel = get_kernel(shape, size, focal_dist, max_focal_dist)
            # Scale brightness of kernel to modulate how visually apparent it is
            kernel *= brightness

            # Get nominal min/max of the kernel coordinates
            kernel_shape = np.asarray(kernel.shape)
            half_size = (kernel_shape - 1) / 2
            min_row, max_row = int(row - half_size[0]), int(row + half_size[0] + 1)
            min_col, max_col = int(col - half_size[1]), int(col + half_size[1] + 1)

            # Get clipped kernel coords (if kernel runs off edge of image)
            k_min_row = np.abs(min_row) if min_row < 0 else 0
            k_max_row = image_size[0] - max_row if max_row > image_size[0] else kernel_shape[0]
            k_min_col = np.abs(min_col) if min_col < 0 else 0
            k_max_col = image_size[1] - max_col if max_col > image_size[1] else kernel_shape[1]

            # Add kernel to image
            frame[np.amax([0, min_row]):np.amin([image_size[0], max_row]),
                  np.amax([0, min_col]):np.amin([image_size[1], max_col])] += \
                      kernel[k_min_row:k_max_row, k_min_col:k_max_col]

        # XXX To dynamically scale, uncomment below. Recommended to keep this off as raw data isn't dynamically scaled
        #frame /= np.percentile(frame, 99.7)
        #frame = np.clip(frame, 0, 1) * 255

        # Save as unit8 image
        frame = np.clip(frame, 0, 255)
        pil_image = Image.fromarray(frame.astype(np.uint8))
        pil_image.save(frame_fpath)


def get_noise_frame(image_size, noise_distribution, smooth_kernel=None):
    """Generate a 2D image of noise given a noise distribution"""

    noise = noise_distribution.rvs(size=image_size)
    if smooth_kernel is not None:
        noise = gaussian_filter(noise, smooth_kernel)
    return  noise


def load_track(track_fpath):
    """Load a track json file and add min/max time"""

    with open(track_fpath, 'r') as json_f:
        loaded_dict = json.load(json_f)

    return loaded_dict


def get_kernel(shape, size, focal_dist=0, max_focal_dist=1024):
    """Get a 2D kernel representing a single particle"""

    # Compute some metrics to modulate the kernel based on distance from focal plane
    if max_focal_dist:
        normalize_focal_dist = focal_dist / max_focal_dist
        clip_factor = 1 - normalize_focal_dist
        scaled_size = size * (1 + normalize_focal_dist)
    else:
        clip_factor = 1
        scaled_size = size


    # Loop over possible kernels
    if shape == 'gaussian':
        kern = Gaussian2DKernel(scaled_size)
    elif shape == 'airy_disk':
        # Use a larger window than default so edges aren't lost (Astropy default is 8)
        window_size = round_up_to_odd(16 * size)
        kern = AiryDisk2DKernel(scaled_size, x_size=window_size, y_size=window_size)
    else:
        raise ValueError(f'Kernel specified ({shape}) not understood.')

    # Normalize and clip based on distance from focal plane
    # Clipping here helps augment the appearance of edges (or rings) the further
    # the particle is from the focal plane
    kern.normalize('peak')
    kern_arr = np.clip(kern.array, -1 * clip_factor, clip_factor)
    return kern_arr / np.max(kern_arr)


def round_up_to_odd(val):
    """Helper to get next largest odd integer"""
    val = int(np.ceil(val))
    return val + 1 if val % 2 == 0 else val