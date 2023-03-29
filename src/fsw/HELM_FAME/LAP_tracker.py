import sys
import os
import os.path as op
import glob
import logging
import json
import multiprocessing
from functools              import partial

from pathlib                import Path
from tqdm                   import tqdm
import numpy                as np
import matplotlib.pyplot    as plt
from sklearn.cluster        import DBSCAN
from scipy.stats            import rankdata
from scipy.spatial          import distance_matrix
from scipy.optimize         import linear_sum_assignment
from scipy.interpolate      import interp1d
import networkx             as nx

from utils.dir_helper        import get_batch_subdir, get_exp_subdir
from utils.file_manipulation import read_image


def get_diff_static(I, ds_median, config):
    """
    Computes a diff between current image I and the dataset median.

    Parameters
    ----------
    I: 2d array
        the current image frame
    ds_median: 2d array
        the dataset median
    config: dict
        configuration
    """

    diff = abs(I - ds_median)

    abs_threshold = config['absthresh'] 
    pc_threshold = config['pcthresh']
    # Threshold is the max value of the abs_threshold, and the value of diff at percentile pc_threshold
    threshold = max(abs_threshold, np.percentile(diff, pc_threshold))
    # Suppress values of rng_diff less than a threshold
    diff[diff < threshold] = 0

    return diff


def get_particles(range_diff, image, clustering_settings):
    """Get the detections using Gary's original method

    Returns a list of particles and their properties

    Parameters
    ----------
    range_diff:
        output from background subtraction
    image:
        original image frame for intensity calculation
        may have a different shape than range_diff
    cluster_settings:
        hyperparameters for the clustering algorithm

    Returns
    -------
    list of dicts with keys:
        pos:            (y, x) coordinates of particle
        size:           number of pixels in cluster
        bbox_tl:        bbox (top, left)
        bbox_hw:        bbox (height, width)
        max_intensity:  max intensity of pixels (list)
    """

    # select points above a threshold, and get their weights
    idx = (range_diff > 0)
    points = np.column_stack(np.nonzero(idx))
    weights = range_diff[idx].ravel().astype(float)

    # empty list to store particles
    particles = []

    if len(points) > 0:
        # use DBSCAN to cluster the points
        dbscan = DBSCAN(eps=clustering_settings['dbscan']['epsilon_px'], 
                        min_samples=clustering_settings['dbscan']['min_weight'])
        labels = dbscan.fit_predict(points, sample_weight=weights)
        n_clusters = int(np.max(labels)) + 1

        for l in range(n_clusters):
            idx = (labels == l)
            # must have specified minimum number of points
            # keep track of clusters that fall below this thresh
            if np.sum(idx) < clustering_settings['filters']['min_px']:
                continue

            relevant = points[idx]
            
            # Build particle properties
            particle = {}
            # center of particle
            particle['pos'] = [round(i, 1) for i in np.average(relevant, axis=0).tolist()]
            # number of pixels in particle
            particle['size'] = int(np.sum(idx))
            # bounding box top left anchor

            # bounding box calculations
            bbox_y, bbox_x = int(np.min(relevant[:,0])), int(np.min(relevant[:,1]))
            bbox_h, bbox_w = int(np.max(relevant[:,0]) - np.min(relevant[:,0])), \
                             int(np.max(relevant[:,1]) - np.min(relevant[:,1]))

            particle['bbox'] = ((bbox_y, bbox_x), (bbox_h, bbox_w))
            
            # convert bounding box indices to original resolution
            yres_ratio = image.shape[0] / range_diff.shape[0]
            xres_ratio = image.shape[1] / range_diff.shape[1]
            bbox_y_ores = int(bbox_y * yres_ratio)
            bbox_h_ores = int(bbox_h * yres_ratio)
            bbox_x_ores = int(bbox_x * xres_ratio)
            bbox_w_ores = int(bbox_w * xres_ratio)

            # max intensity for each channel
            if len(image.shape) == 2:
                # grayscale original image, single channel
                particle['max_intensity'] = [int(np.amax(image[bbox_y_ores:bbox_y_ores+bbox_h_ores+1, 
                                                              bbox_x_ores:bbox_x_ores+bbox_w_ores+1]))]
            else:
                # RGB original image, max per channel
                particle['max_intensity'] = np.amax(image[bbox_y_ores:bbox_y_ores+bbox_h_ores+1, 
                                                         bbox_x_ores:bbox_x_ores+bbox_w_ores+1],
                                                    axis=(0,1)).tolist()

            particles.append(particle)

    return particles


def linking_LAP(prev_particles, next_particles, max_link):
    """ Calculate LAP cost matrix between particles in consecutive frames

    Parameters
    ----------
    prev_particles: list
        list of particle dicts detected in frame n-1
    next_particles: list
        list of particle dicts detected in frame n
    max_link: float
        maximum linking distance between particles
    """

    # Get coordinates from list of particle dicts
    prev_coords = [p['pos'] for p in prev_particles]
    next_coords = [p['pos'] for p in next_particles]
    p = len(prev_coords)
    n = len(next_coords)

    # Top left is the euclidean cost matrix between the particles
    topleft = distance_matrix(prev_coords, next_coords)
    # If cost is higher than max, set it to inf
    topleft[topleft > max_link] = 1e8

    # Top right and bottom right are diagonal matrices of value 1.05 * max
    # for indicating starting & stopping tracks at this frame
    if len(topleft[topleft != 1e8]) != 0:
        topright = np.ones((p,p)) * 1e8
        np.fill_diagonal(topright, 1.05 * np.max(topleft[topleft != 1e8]))
        botleft = np.ones((n,n)) * 1e8
        np.fill_diagonal(botleft, 1.05 * np.max(topleft[topleft != 1e8]))
    else:
        # topleft is all 1e8, no links possible. fill idagonals with 1s for guaranteed solution.
        topright = np.ones((p,p)) * 1e8
        np.fill_diagonal(topright, 1)
        botleft = np.ones((n,n)) * 1e8
        np.fill_diagonal(botleft, 1)

    # Bottom right is a theoretical necessary, described in Supplement 3 of 
    # Jaqaman et al. 2008. It's the transpose of top left, with "non-inf" values
    # set to a minimal cost.
    botright = topleft.T.copy()
    botright[botright != 1e8] = 1e-8

    # Build final cost matrix
    left = np.concatenate((topleft, botleft), axis=0)
    right = np.concatenate((topright, botright), axis=0)
    LAP_cost = np.concatenate((left, right), axis=1)

    return LAP_cost


def stitch_LAP(track_ends, track_starts, max_link, max_skips):
    """ Calculate LAP cost matrix between track ends and starts for stitching

    Parameters
    ----------
    track_ends: list
        List of particles that are at the end of tracks
    track_starts: list
        List of particles that are at the start of tracks
    max_link: float
        Maximum distance between stitched start/end points
    max_skips: float
        Maximum skipped frames between start/end points
    """

    end_coords = [(e[0], e[1]) for e in track_ends]
    end_times = [[e[2]] for e in track_ends]
    start_coords = [(s[0], s[1]) for s in track_starts]
    start_times = [[s[2]] for s in track_starts]

    e = len(track_ends)
    s = len(track_starts)

    topleft = distance_matrix(end_coords, start_coords)
    frame_gaps = distance_matrix(end_times, start_times)
    topleft[np.where(frame_gaps > max_skips)] = 1e8
    topleft[topleft > max_link] = 1e8

    if len(topleft[topleft != 1e8]) != 0:
        topright = np.ones((e,e)) * 1e8
        np.fill_diagonal(topright, 1.05 * np.max(topleft[topleft != 1e8]))
        botleft = np.ones((s,s)) * 1e8
        np.fill_diagonal(botleft, 1.05 * np.max(topleft[topleft != 1e8]))
    else:
        # topleft is all 1e8, no links possible. fill idagonals with 1s for guaranteed solution.
        topright = np.ones((e,e)) * 1e8
        np.fill_diagonal(topright, 1)
        botleft = np.ones((s,s)) * 1e8
        np.fill_diagonal(botleft, 1)

    botright = topleft.T.copy()
    botright[botright != 1e8] = 1e-8

    left = np.concatenate((topleft, botleft), axis=0)
    right = np.concatenate((topright, botright), axis=0)
    LAP_cost = np.concatenate((left, right), axis=1)

    return LAP_cost

def plot_tracks(G, exp_name, plot_output_directory,
                win_size=(1024, 1024)):
    """Plot traces for all tracks on a dark background

    Parameters
    ----------
    track_fpaths: list of str
        Full filepaths to each track to be plotted
    exp_name: str
        Experiment name
    plot_output_directory: str
        Directory for saving the track plot
    win_size: iterable
        Number of pixels in row and column dimensions, respectively.
    """

    # Create plot and use dark background
    plt.style.use('dark_background')

    # Debug track plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Overlay track plot
    px = 1/128
    fig2 = plt.figure(frameon=False, dpi=128)
    fig2.set_size_inches(2048*px, 2048*px)
    ax2 = plt.Axes(fig2, [0., 0., 1., 1.])
    ax2.set_axis_off()
    fig2.add_axes(ax2)

    if len(G) == 0:
        logging.warning('No tracks were available to plot')
    
    for cc in list(nx.connected_components(G)):
        cc_sorted = sorted(cc, key = lambda x: x[2])
        positions = np.array([(x,y) for x,y,z in cc_sorted])
        ax.plot(positions[:, 1], positions[:, 0])
        ax2.plot(positions[:, 1], positions[:, 0])

    # Set up title and axis labels
    ax.set_title('Particle tracks identified in experiment\n' + exp_name)
    ax.invert_yaxis()
    ax.axis('equal')  # Force a square axis
    ax.set_xlim(0, win_size[1])
    ax.set_ylim(win_size[0], 0)

    ax2.invert_yaxis()
    ax2.axis('equal')
    ax2.axis('off')
    ax2.set_xlim(0, win_size[1])
    ax2.set_ylim(win_size[0], 0)

    fig.savefig(op.join(plot_output_directory, exp_name + "_track_plots.png"),
                dpi=150)
    fig2.savefig(op.join(plot_output_directory, exp_name + "_track_overlay.png"))
    plt.close()

def export_JSON(G, particle_dict, track_dir, config):
    # list of tracks and their nodes
    ccs = list(nx.connected_components(G))
    
    # sort tracks by starting node time
    ccs = sorted(ccs, key = lambda x: min([p[2] for p in x]))

    # for each connected component
    for idx, cc in enumerate(ccs):
        json_dict = {
            'Times': [],
            'Particles_Position': [],
            'Particles_Estimated_Position': [],
            'Particles_Size': [],
            'Particles_Bbox': [],
            'Particles_Max_Intensity': [],
            'Track_ID': idx,
            'classification': None
        }

        # sort track by timestamp
        cc_sorted = sorted(cc, key = lambda x: x[2])
        cc_coords = [[c[0], c[1]] for c in cc_sorted]
        cc_times = [int(c[2]) for c in cc_sorted]

        # function for interpolation
        interp_func = interp1d(cc_times, cc_coords, kind='linear', axis=0)

        # for each timestep in timerange
        for t in range(cc_times[0], cc_times[-1]+1):
            json_dict['Times'].append(t)

            if t in cc_times:
                # particle exists, no interpolation
                # get particle object
                particle = particle_dict[cc_sorted[cc_times.index(t)]]
                json_dict['Particles_Position'].append(particle['pos'])
                json_dict['Particles_Estimated_Position'].append(particle['pos'])
                json_dict['Particles_Size'].append(particle['size'])
                json_dict['Particles_Bbox'].append(particle['bbox'])
                json_dict['Particles_Max_Intensity'].append(particle['max_intensity'])

            else:
                # particle DNE, interpolate
                json_dict['Particles_Estimated_Position'].append(interp_func(t).tolist())
                json_dict['Particles_Position'].append(None)
                json_dict['Particles_Size'].append(None)
                json_dict['Particles_Bbox'].append(None)
                json_dict['Particles_Max_Intensity'].append(None)

        # save dictionary to JSON
        json_fpath = op.join(track_dir, f'{idx:05}.json')
        with open(json_fpath, 'w') as f:
            json.dump(json_dict, f, indent=2)

def _mp_particles(fpath, mf, conf):
    """ Multiprocessing function for reading and identifying particles """

    frame = read_image(fpath, conf['raw_dims'])
    diff = get_diff_static(frame, mf, conf['diff_comp'])
    detections = get_particles(diff, frame, conf['clustering'])
    return detections

def run_tracker(exp_dir, holograms, originals, config, n_workers=1):
    """Execute the tracker code for an experiment

    Parameters
    ----------
    exp_dir: str
        Experiment directory path
    holograms: list
        Ordered list of filepaths to holograms
    config: dict
        Loaded HELM configuration dictionary
    n_workers: int
        Number of workers to use for multiprocessed portions
    """

    exp_name = Path(exp_dir).name

    tracker_settings = config['tracker_settings']
    tracker_settings['raw_dims'] = config['preproc_resolution']
    track_plot = tracker_settings['track_plot']

    track_dir = get_exp_subdir('track_dir', exp_dir, config, rm_existing=True)
    plot_dir = get_exp_subdir('evaluation_dir', exp_dir, config, rm_existing=True)
    tracker_debug_dir = op.join(plot_dir, "tracker_debug")

    # Track and plot directories if they don't exist yet
    Path(tracker_debug_dir).mkdir(parents=True, exist_ok=True)
    logging.info(f'Track files dir: {op.join(*Path(track_dir).parts[-2:])}')
    logging.info(f'Track plots dir: {op.join(*Path(plot_dir).parts[-2:])}')

    # Read median image
    median_frame = read_image(op.join(get_exp_subdir('validate_dir', exp_dir, config), 
                                    f'{exp_name}_median_image.tif'),
                              config['preproc_resolution'])
    median_frame = median_frame.astype(np.float)

    # Get particles per frame

    with multiprocessing.Pool(n_workers) as pool:
        particle_stack = list(tqdm(pool.imap_unordered(
                                    partial(_mp_particles, mf=median_frame, conf=tracker_settings), 
                                    holograms), total=len(holograms)))


    # Link particles into tracks
    G = nx.Graph()
    particle_dict = {}
    for i in tqdm(range(1, len(particle_stack))):
        p = len(particle_stack[i-1])
        n = len(particle_stack[i])

        if p == 0 or n == 0:
            # No particles in previous or next frame, no edges
            continue

        linking_cost = linking_LAP(particle_stack[i-1], 
                                 particle_stack[i], 
                                 tracker_settings['LAPtracking']['max_assignment_dist'])

        rows, cols = linear_sum_assignment(linking_cost)
        for row, col in zip(rows, cols):
            if row < p and col < n:
                prev_coord = np.concatenate((particle_stack[i-1][row]['pos'], [i-1]))
                next_coord = np.concatenate((particle_stack[i][col]['pos'], [i]))
                # Add edge to graph
                G.add_edge(tuple(prev_coord), tuple(next_coord))
                # Add nodes to dict
                particle_dict[tuple(prev_coord)] = particle_stack[i-1][row]
                particle_dict[tuple(next_coord)] = particle_stack[i][col]

    # Track stitching
    track_starts = []
    track_ends = []
    for cc in list(nx.connected_components(G)):
        cc_sorted = sorted(cc, key = lambda x: x[2])
        track_starts.append(cc_sorted[0])
        track_ends.append(cc_sorted[-1])

    e = len(track_ends)
    s = len(track_starts)
    if e != 0 and s != 0:
        stitching_cost = stitch_LAP(track_ends, track_starts, 
                                    tracker_settings['LAPtracking']['max_assignment_dist'],
                                    tracker_settings['LAPtracking']['max_skip'])

        rows, cols = linear_sum_assignment(stitching_cost)
        for row, col in zip(rows, cols):
            if row < e and col < s:
                # Add stitched edges
                # TODO: when writing to JSON, handle interpolation
                G.add_edge(track_ends[row], track_starts[col])

    # Drop tracks with len < limit
    for component in list(nx.connected_components(G)):
        if len(component) < tracker_settings['LAPtracking']['min_track_obs']:
            for node in component:
                G.remove_node(node)

    # Plot tracks
    if track_plot:
        plot_tracks(G, exp_name, plot_dir)

    # export tracks to json
    export_JSON(G, particle_dict, track_dir, config)
