import os
import json
import sys
import logging
import shutil
import subprocess
import glob
import signal
import multiprocessing

import matplotlib.gridspec as gridspec
import numpy               as np
import matplotlib.pyplot   as plt
import matplotlib.lines    as mlines
import os.path             as op

import matplotlib
matplotlib.use('agg')

from tqdm                  import tqdm
from PIL                   import Image, ImageDraw, ImageFont
from collections           import OrderedDict
from numpy                 import genfromtxt
from pathlib               import Path

from gsw.HELM_FAME.util    import get_rainbow_black_red_colormap
from gsw.HELM_FAME.util    import create_y_range
from gsw.HELM_FAME.util    import Track
from gsw.HELM_FAME.util    import load_in_track
from gsw.HELM_FAME.util    import load_in_autotrack
from gsw.HELM_FAME.util    import count_motility, max_particle_intensity
from utils.dir_helper      import get_batch_subdir
from utils.dir_helper      import get_exp_subdir
from skimage.transform     import resize

def gen_visualizer_frame(args):

    plt.style.use('dark_background')
    plt.suptitle(args['experiment_path'], fontsize=9, color="white")
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0, hspace=0)

    auto_motile_color = 'fuchsia'
    auto_nonmotile_color = 'cyan'
    fame_particle_color = 'yellow'
    fame_frame_color = 'orange'
    track_motile_color = 'purple'
    track_nonmotile_color = 'mediumblue'

    _mhi_npy = args['_mhi_npy']
    index = args['index']
    total = args['total']
    start_frame = args['start_frame']
    end_frame = args['end_frame']
    instrument = args['instrument']
    rehydrated_image_dir = args['rehydrated_image_dir']

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
    plt.imshow(img_array, cmap=colormap)
    plt.axis('off')
    
    # Subplot region for labels (bottom left)
    plt.subplot(gs[7, :4], )
    plt.axis('off')

    # Subplot region for colorbar
    plt.subplot(gs[7, 4:8], )
    plt.axis('off')

    plt.colorbar(ax=plt.gca(), fraction=1, orientation='horizontal').set_ticks([])

    # Subplot region for motility ticker
    plt.rc('xtick', labelsize=5) 
    plt.rc('ytick', labelsize=5) 
    plt.subplot(gs[0, :8])

    if instrument == "HELM":

        y_range = create_y_range(args['motile_count'], args['non_motile_count'], 
                                 args['auto_motile_count'], args['auto_non_motile_count'])
        
        # add hand labeled track counts (motile/non-motile) to ticker if available
        if os.path.isfile(args['track_file_path']):
            plt.plot(args['frame_count'], args['motile_count'], color=track_motile_color, linewidth=1)
            plt.plot(args['frame_count'], args['non_motile_count'], color=track_nonmotile_color, linewidth=1)

        # add automated track counts (motile/non-motile) to ticker if available
        if os.path.isdir(args['auto_track_path']):
            plt.plot(args['frame_count'], args['auto_motile_count'], color=auto_motile_color, linewidth=1)
            plt.plot(args['frame_count'], args['auto_non_motile_count'], color=auto_nonmotile_color, linewidth=1)

    elif instrument == "FAME":

        y_range = None

        # adds max particle intensity to ticker if available
        if os.path.isdir(args['auto_track_path']):
            plt.plot(args['frame_count'], args['particle_intensity_list'], color=fame_particle_color, linewidth=1)
            plt.plot(args['frame_count'], args['frame_intensity_list'], color=fame_frame_color, linewidth=1)

    plt.axvline(x=index, color="white", linewidth=2)
    plt.xlim(start_frame, end_frame)
    
    if y_range is not None:
        plt.ylim(y_range[0], y_range[1])

    # Subplot region for frame with tracks plotted
    ax = plt.subplot(gs[1:7, :4])

    background_image_dir = os.path.join(args['experiment_path'], args['config']['experiment_dirs']['baseline_dir'])
    background_frame_prefix = ""
    frame_dir_path = background_image_dir
    _l = sorted(os.listdir(frame_dir_path))

    if len(_l) > 0:

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

    else:

        frame_name = f"{index:05d}.png"
        _image_path = os.path.join(rehydrated_image_dir, frame_name)

    imager = Image.open(_image_path)
    imager = imager.convert("RGB")
    imager = np.array(imager)

    imager = resize(imager, (1024, 1024), anti_aliasing=True)

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

                if instrument == "HELM":
                    if trackPoint1["mobility"] == "motile":
                        color = track_motile_color
                    else:
                        color = track_nonmotile_color
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

                if instrument == "HELM":
                    if trackPoint1["mobility"] == "motile": # TODO - we need to unify all these keys (mobility/classification) and our classes (motile/other)
                        color = auto_motile_color
                    else:
                        color = auto_nonmotile_color
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
    if len(_l) > 0:
        plt.text(125, 1000, "Background Subtracted Data", color='white')
    else:
        plt.text(275, 1000, "Rehydrated Data", color='white')

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

    if instrument == "HELM":
        # Add bottom left label/color definitions for autonomous tracks if they were plottted
        if os.path.isdir(args['auto_track_path']):
            plt.text(20, 1100, "Autonomous Motile Track", color=auto_motile_color, weight="bold", fontdict={'size': 6})
            plt.text(20, 1150, "Autonomous Non-Motile Track", color=auto_nonmotile_color, weight="bold", fontdict={'size': 6})

        # Add bottom left label/color definition for hand labeled tracks if they were plotted
        if os.path.isfile(args['track_file_path']):
            plt.text(20, 1200, "Hand Labeled Motile Track", color=track_motile_color, weight="bold", fontdict={'size': 6})    
            plt.text(20, 1250, "Hand Labeled Non-Motile Track", color=track_nonmotile_color, weight="bold", fontdict={'size': 6})

    elif instrument == "FAME":
            plt.text(20, 1100, "Autonomous Track", color=auto_nonmotile_color, weight="bold", fontdict={'size': 6})
            plt.text(20, 1150, "Hand Labeled Track", color=track_nonmotile_color, weight="bold", fontdict={'size': 6})  
            plt.text(20, 1200, "Max Particle Intensity", color=fame_particle_color, weight="bold", fontdict={'size': 6})
            plt.text(20, 1250, "Max Frame Intensity", color=fame_frame_color, weight="bold", fontdict={'size': 6})

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(os.path.join(args['output_dir_path'], str(index).zfill(4) + args['ext']), dpi=300)

def visualization(experiment_path, config, instrument, n_workers=1, cleanup=False):
    '''Main function to generate HELM/FAME visualizations'''
    ext = config['validate']['baseline_subtracted_ext']

    experiment_name = Path(experiment_path).name
    back_image_dir = get_exp_subdir('baseline_dir', experiment_path, config)

    output_dir_path = op.join(get_exp_subdir('asdp_dir', experiment_path, config), 'movie/')
    rehydrated_image_dir = op.join(get_exp_subdir('asdp_dir', experiment_path, config), 'rehydrated/')

    num_bl_subtracted = len(list(glob.glob(os.path.join(output_dir_path,"*.png"))))
    num_rehydrated = len(list(glob.glob(os.path.join(rehydrated_image_dir,"*.png"))))

    if num_bl_subtracted > 0:
        start_frame = 1
        end_frame = num_bl_subtracted
    else:
        start_frame = 1
        end_frame = num_rehydrated

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    total = end_frame - start_frame
    _mhi_npy_path = list(glob.glob(os.path.join(experiment_path, config['experiment_dirs']['validate_dir'],"*mhi.npy")))[0]
    _mhi_npy = np.load(_mhi_npy_path).astype(float)

    track_file_path = os.path.join(experiment_path, config['experiment_dirs']['label_dir'], f'{experiment_name}_labels.csv')
    
    if instrument == "HELM":
        auto_track_path = os.path.join(experiment_path, config['experiment_dirs']['predict_dir'])
    elif instrument == "FAME":
        auto_track_path = os.path.join(experiment_path, config['experiment_dirs']['track_dir'])

    max_intensity_path = list(glob.glob(os.path.join(experiment_path, config['experiment_dirs']['validate_dir'], "*_timestats_max_intensity.csv")))[0]
    frame_max_intensity = genfromtxt(max_intensity_path, delimiter=',')[1:,:]

    frame_count = list(range(start_frame, end_frame))

    auto_motile_count = []
    auto_non_motile_count = []
    if os.path.isdir(auto_track_path):

        logging.info(f'load auto track: {op.join(*Path(auto_track_path).parts[-2:])}')

        autoTrackDict, autoFrameDict = load_in_autotrack(auto_track_path)

        autoTrackList = autoTrackDict.keys()
        autoTrackList = sorted(autoTrackList)

        autoFrameList = autoFrameDict.keys()
        autoFrameList = sorted(autoFrameList)

        auto_motile_count = []
        auto_non_motile_count = []
        particle_intensity_list = []
        frame_intensity_list = []
        for frameNumber in range(start_frame, end_frame):
            track_point_list = autoFrameDict.get(frameNumber, [])

            if instrument == "HELM":
                _motile, _non_motile = count_motility(track_point_list)
                auto_motile_count.append(_motile)
                auto_non_motile_count.append(_non_motile)
            elif instrument == "FAME":
                particle_intensity = max_particle_intensity(track_point_list)
                particle_intensity_list.append(particle_intensity)

                frame_intensity_list.append(frame_max_intensity[frameNumber,1])

    motile_count = []
    non_motile_count = []
    if os.path.isfile(track_file_path):

        logging.info(f'load hand track: {op.join(*Path(track_file_path).parts[-2:])}')

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
                'particle_intensity_list': particle_intensity_list,
                'frame_intensity_list': frame_intensity_list,
                'track_file_path': track_file_path,
                'auto_track_path': auto_track_path,
                'frame_count': frame_count,
                'output_dir_path': output_dir_path,
                'ext': ext,
                'instrument': instrument,
                'rehydrated_image_dir': rehydrated_image_dir,
                'config': config}

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
        shutil.rmtree(output_dir_path)
        logging.info(f'Cleaned up visualizer frame directory: {output_dir_path}')
