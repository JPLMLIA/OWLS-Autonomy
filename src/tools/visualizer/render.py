import os
import os.path as op
import sys
import joblib
import logging
import platform
import shutil
import subprocess
import yaml
import glob
import signal
import csv
import multiprocessing

import matplotlib.gridspec as gridspec
import numpy               as np
import matplotlib.pyplot   as plt
import matplotlib.lines    as mlines

import matplotlib
matplotlib.use('agg')

from tqdm        import tqdm
from PIL         import Image, ImageDraw, ImageFont
from typing      import Dict, Optional, List, Any
from typing      import Any, List, Optional, Tuple
from collections import OrderedDict

from tools.visualizer.util import get_rainbow_black_red_colormap
from tools.visualizer.util import create_y_range
from tools.visualizer.util import Track
from tools.visualizer.util import load_in_track
from tools.visualizer.util import load_in_autotrack
from tools.visualizer.util import count_motility
from utils.dir_helper      import get_batch_subdir
from utils.dir_helper      import get_exp_subdir

def gen_visualizer_frame(args):

    plt.style.use('dark_background')
    plt.suptitle(args['experiment_path'], fontsize=9, color="white")
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0, hspace=0)

    auto_motile_color = '#398900' # green
    auto_nonmotile_color = '#0000ff' # dark blue
    track_motile_color = '#37ff14' # neon green
    track_nonmotile_color = '#00ffff' # teal

    _mhi_npy = args['_mhi_npy']
    index = args['index']
    total = args['total']
    start_frame = args['start_frame']
    end_frame = args['end_frame']

    colormap = get_rainbow_black_red_colormap()
    plt.register_cmap(cmap=colormap, name='rainbow')

    a = _mhi_npy.copy()
    a = a / total

    # save index of nan points
    idx = np.isnan(a)

    colormap = get_rainbow_black_red_colormap()
    img_array = np.uint8(colormap(a) * 255)

    # set nan points to some gray
    img_array[idx] = 64

    # set to white color for all points at the frame number
    lowest_ind = np.amax([index - 2, 0])
    highest_ind = np.amin([index + 2, total - 1])
    white_inds = np.where(np.logical_and(_mhi_npy >= lowest_ind, _mhi_npy <= highest_ind))
    img_array[white_inds] = 255

    # Subplot region for MHI image
    plt.subplot(gs[1:7, 4:8], )
    plt.imshow(img_array, cmap='rainbow')
    plt.axis('off')
    
    # Subplot region for labels (bottom left)
    plt.subplot(gs[7, :4], )
    plt.axis('off')

    # Subplot region for colorbar
    plt.subplot(gs[7, 4:8], )
    plt.axis('off')

    plt.colorbar(fraction=1, orientation='horizontal').set_ticks([])

    x_range = [start_frame, end_frame]
    y_range = create_y_range(args['motile_count'], args['non_motile_count'], 
                             args['auto_motile_count'], args['auto_non_motile_count'])

    # Subplot region for motility ticker
    plt.rc('xtick', labelsize=5) 
    plt.rc('ytick', labelsize=5) 
    plt.subplot(gs[0, :8])

    # add hand labeled rack counts (motile/non-motile) to ticker if available
    if os.path.isfile(args['track_file_path']):
        plt.plot(args['frame_count'], args['motile_count'], color=track_motile_color, linewidth=1)
        plt.plot(args['frame_count'], args['non_motile_count'], color=track_nonmotile_color, linewidth=1)

    # add automated track counts (motile/non-motile) to ticker if available
    if os.path.isdir(args['auto_track_path']):
        plt.plot(args['frame_count'], args['auto_motile_count'], color=auto_motile_color, linewidth=1)
        plt.plot(args['frame_count'], args['auto_non_motile_count'], color=auto_nonmotile_color, linewidth=1)

    plt.axvline(x=index, color="white", linewidth=2)

    if x_range is not None and y_range is not None:
        plt.axis([x_range[0], x_range[1], y_range[0], y_range[1]])
    
    # Subplot region for frame with tracks plotted
    ax = plt.subplot(gs[1:7, :4])

    background_image_dir = os.path.join(args['experiment_path'], "holograms_baseline_subtracted")
    background_frame_prefix = ""
    frame_dir_path = background_image_dir
    _l = sorted(os.listdir(frame_dir_path))
    frame_dict = OrderedDict()
    for x in _l:
        if x[0:len(background_frame_prefix)] != background_frame_prefix:
            continue
        tmp = x[len(background_frame_prefix):].split(".")
        frame_number = int(tmp[0])

        frame_dict[frame_number] = x

    frame_name = frame_dict[index]

    if frame_name is None:
        logging.error("frame %s doesnt exist" % index)
        return

    _image_path = os.path.join(background_image_dir, frame_name)

    imager = Image.open(_image_path)
    imager = imager.convert("RGB")
    imager = np.array(imager)
    x_dim, y_dim, z_dim = imager.shape
    plt.imshow(imager)
    plt.axis('off')

    # draw hand labeled tracks on frame
    if os.path.exists(args['track_file_path']):
        track_list = args['handTrackList']

        for trackNumber in track_list:
            points = args['handTrackDict'][trackNumber]
            track = Track(trackNumber, points)

            if track is None:
                continue
            if not track.covers_frame(index):
                continue

            all_track_points = track.points
            listOfAllTrackPoints = all_track_points

            # collect track points seen up to the moment of the frame
            listOfTrackPoints = []
            for trackPoint in listOfAllTrackPoints:
                frameNumber = trackPoint["frame"]
                if frameNumber > index:
                    continue
                listOfTrackPoints.append(trackPoint)

            if len(listOfTrackPoints) == 0:
                logging.error("no track point of track %s for this frame" % trackNumber)
                return

            trackPoint = listOfTrackPoints[0]
            x, y = trackPoint["location"]

            firstDraw = True
            for i in range(len(listOfTrackPoints) - 1):
                trackPoint0 = listOfTrackPoints[i]
                x0, y0 = trackPoint0["location"]
                trackPoint1 = listOfTrackPoints[i + 1]
                x1, y1 = trackPoint1["location"]

                if trackPoint1["mobility"] == "Motile": # TODO - we need to unify all these keys (mobility/classification) and our classes (motile/other)
                    color = track_motile_color
                else:
                    color = track_nonmotile_color

                ax.add_line(mlines.Line2D([x0,x1], [y0,y1], linewidth=1, color=color))                      

                if firstDraw:
                    x_bar = x0 / x_dim
                    y_bar = 1 - (y0 / y_dim)
                    ax.add_artist(plt.Circle((x_bar, y_bar), 0.009, color=color, transform=ax.transAxes, fill=False))
                    plt.text(x_bar, y_bar, "%s" % (trackNumber), color=color, ha='left', va='bottom', transform=ax.transAxes, fontdict={'size': 5})
                    firstDraw = False

    # draw automated tracks on frame
    if os.path.exists(args['auto_track_path']):
        trackList = args['autoTrackDict'].keys()
        auto_track_list = sorted(trackList)
        for trackNumber in auto_track_list:
            points = args['autoTrackDict'][trackNumber]
            track = Track(trackNumber, points)

            if track is None:
                continue

            if not track.covers_frame(index):
                continue

            all_track_points = track.points
            listOfAllTrackPoints = all_track_points

            # collect track points seen up to the moment of the frame
            listOfTrackPoints = []
            for trackPoint in listOfAllTrackPoints:
                frameNumber = trackPoint["frame"]
                if frameNumber > index:
                    continue
                listOfTrackPoints.append(trackPoint)

            if len(listOfTrackPoints) == 0:
                logging.error("no track point of track %s for this frame" % trackNumber)
                return

            trackPoint = listOfTrackPoints[0]
            x, y = trackPoint["location"]

            firstDraw = True
            for i in range(len(listOfTrackPoints) - 1):
                trackPoint0 = listOfTrackPoints[i]
                x0, y0 = trackPoint0["location"]
                trackPoint1 = listOfTrackPoints[i + 1]
                x1, y1 = trackPoint1["location"]

                if trackPoint1["mobility"] == "motile": # TODO - we need to unify all these keys (mobility/classification) and our classes (motile/other)
                    color = auto_motile_color
                else:
                    color = auto_nonmotile_color

                ax.add_line(mlines.Line2D([x0,x1], [y0,y1], linewidth=1, color=color))                      

                if firstDraw:
                    x_bar = x0 / x_dim
                    y_bar = 1 - (y0 / y_dim)
                    ax.add_artist(plt.Circle((x_bar, y_bar), 0.009, color=color, transform=ax.transAxes, fill=False))
                    plt.text(x_bar, y_bar, "%s" % (trackNumber), color=color, ha='left', va='bottom', transform=ax.transAxes, fontdict={'size': 5})
                    firstDraw = False

    # Draw text on screen for static labels
    plt.text(975, 1125, '1', color='white', fontdict={'size': 7})
    plt.text(2055, 1125, f'{total-1}', color='white', fontdict={'size': 7})
    plt.text(1300, 1000, "Motion History Image", color='white')
    plt.text(125, 1000, "Background Subtracted Data", color='white')

    # Calculate position of current index in colorbar pixel space and draw white line over top.
    span = 2040 - 1015
    spot = index / total
    spot = spot * span + 1015
    plt.text(spot, 1140, "|", color='white', fontdict={'size': 15})

    label_spot = spot
    if index > 9 and index < 100:
        label_spot -= 7
    if index >= 100:
        label_spot -= 16

    plt.text(label_spot, 1200, f'{index}', color='white', fontdict={'size': 7})

    # Add bottom left label/color definitions for autonomous tracks if they were plottted
    if os.path.isdir(args['auto_track_path']):
        plt.text(20, 1100, "Autonomous Motile Track", color=auto_motile_color, weight="bold", fontdict={'size': 6})
        plt.text(20, 1150, "Autonomous Non-Motile Track", color=auto_nonmotile_color, weight="bold", fontdict={'size': 6})

    # Add bottom left label/color definition for hand labeled tracks if they were plotted
    if os.path.isfile(args['track_file_path']):
        plt.text(20, 1200, "Hand Labeled Motile Track", color=track_motile_color, weight="bold", fontdict={'size': 6})    
        plt.text(20, 1250, "Hand Labeled Non-Motile Track", color=track_nonmotile_color, weight="bold", fontdict={'size': 6})

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(os.path.join(args['output_dir_path'], str(index).zfill(4) + args['ext']), dpi=300)

def HELM_Visualization(experiment_path, config, n_workers=1, cleanup=False):
    '''Main function to generate HELM visualization'''
    experiment_name = experiment_path.split("/")[-1]
    back_image_dir = get_exp_subdir('baseline_dir', experiment_path, config)
    ext = config['validate']['baseline_subtracted_ext']
    hfiles = glob.glob(op.join(back_image_dir, f'*{ext}'))
    hfiles = [x.rstrip(ext) for x in hfiles]
    hfiles = [x.split("/")[-1] for x in hfiles]
    hfiles = [int(x) for x in hfiles]

    start_frame = min(hfiles)
    end_frame = max(hfiles)

    output_dir_path = op.join(get_exp_subdir('asdp_dir', experiment_path, config), 'movie/')

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    total = end_frame - start_frame
    _mhi_npy_path = list(glob.glob(experiment_path + "/validate/*mhi.npy"))[0]
    _mhi_npy = np.load(_mhi_npy_path).astype(float)

    track_file_path = os.path.join(experiment_path, 'labels', f'verbose_{experiment_name}.csv')
    auto_track_path = os.path.join(experiment_path, 'predict') # TODO - this needs to be tied into the config key

    frame_count = list(range(start_frame, end_frame))

    auto_motile_count = []
    auto_non_motile_count = []
    if os.path.isdir(auto_track_path):

        logging.info("try to load in auto track info from dir %s" % auto_track_path)

        autoTrackDict, autoFrameDict = load_in_autotrack(auto_track_path)

        autoTrackList = autoTrackDict.keys()
        autoTrackList = sorted(autoTrackList)

        autoFrameList = autoFrameDict.keys()
        autoFrameList = sorted(autoFrameList)

        auto_motile_count = []
        auto_non_motile_count = []
        for frameNumber in range(start_frame, end_frame):
            track_point_list = autoFrameDict.get(frameNumber, [])
            _motile, _non_motile = count_motility(track_point_list)
            auto_motile_count.append(_motile)
            auto_non_motile_count.append(_non_motile)

    motile_count = []
    non_motile_count = []
    if os.path.isfile(track_file_path):

        logging.info("try to load in hand track info from file %s" % track_file_path)

        handTrackDict, handFrameDict = load_in_track(track_file_path)

        handTrackList = list(handTrackDict.keys())
        handTrackList.sort()

        handFrameList = list(handFrameDict.keys())
        handFrameList.sort()

        motile_count = []
        non_motile_count = []
        for frameNumber in range(start_frame, end_frame):
            track_point_list = handFrameDict.get(frameNumber, [])
            _motile, _non_motile = count_motility(track_point_list)
            motile_count.append(_motile)
            non_motile_count.append(_non_motile)

    mp_batches = []
    for index in tqdm(range(start_frame, total)):
        args = {'index':index,
                'experiment_path':experiment_path,
                '_mhi_npy': _mhi_npy,
                'total': total,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'motile_count' : motile_count, 
                'non_motile_count': non_motile_count, 
                'auto_motile_count': auto_motile_count, 
                'auto_non_motile_count': auto_non_motile_count,
                'track_file_path': track_file_path,
                'auto_track_path': auto_track_path,
                'frame_count': frame_count,
                'output_dir_path': output_dir_path,
                'ext': ext}

        if os.path.isfile(track_file_path):
            args['handTrackDict'] = handTrackDict
            args['handTrackList'] = handTrackList
        if os.path.isdir(auto_track_path):
            args['autoTrackDict'] = autoTrackDict
                
        mp_batches.append(args)

    # Run multiprocessing batches for intensity/diff/MHI calculations
    with multiprocessing.Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(gen_visualizer_frame, mp_batches), total=total, desc='Calculate visualizer frames'))

    save_fpath = op.join(get_exp_subdir('asdp_dir', experiment_path, config), f"{experiment_name}_visualizer.mp4")

    if os.path.exists(save_fpath):
        os.remove(save_fpath)

    # Generate the mp4 movie from the frames saved in the movie tmp dir.
    ffmpeg_command = ['ffmpeg', '-framerate', '15', '-i',
                      os.path.join(output_dir_path, '%04d' + ext), '-y', '-vf',
                      'format=yuv420p', save_fpath]

    cmd = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = cmd.communicate()
    cmd.send_signal(signal.SIGINT)
    cmd.wait()

    if cleanup:
        with os.scandir(output_dir_path) as it:
            for entry in it:
                if entry.is_dir():
                    shutil.rmtree(entry.path)
