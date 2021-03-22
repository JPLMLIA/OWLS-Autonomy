'''
Command line interface to reconstruct a folder of mugshots into source images
'''
import sys
sys.path.append("../")

import logging
import argparse
import os
import glob
import signal
import json

import os.path    as op
import numpy      as np
import subprocess as SP
import skimage.io as sio

from math  import ceil, floor
from PIL   import Image
from PIL   import ImageDraw

from utils import logger

def reconstruct_frames_to_video(input_dir, output_dir):
    
    files = list(glob.glob(input_dir + '/*.png'))
    files = [x.split("/")[-1] for x in files]

    params = [x.rstrip(".png").split("_") for x in files]
    params = np.stack(params)
    params = params.astype(float)
    num_tracks = int(np.max(params[:,6]))

    # TODO - +1 for off by one error?
    total_frames = max([int(x.split("_")[2]) for x in files]) + 1
    
    split = files[0].rstrip(".png").split("_")
    split = [int(x) for x in split]
    recon_row = split[0]
    recon_col = split[1]
    row_width = split[5]
    col_width = split[6]

    background_path = os.path.join(input_dir, "background.tif")
    
    if not op.exists(output_dir):
        os.makedirs(output_dir)

    for x in range(0, total_frames):

        if os.path.exists(background_path):
            base = Image.open(background_path)
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

        tracks = list(glob.glob(input_dir + '/*.json'))
        for track in tracks:

            with open(track, 'r') as f:
                d = json.load(f)

                times = np.asarray(d["Times"])
                if x in times:

                    track_index = int(track.split("/")[-1].rstrip(".json"))
                    tmp = list(glob.glob(f'{input_dir}/*_*_*_*_*_*_*_{track_index}_*_*_*_*.png'))
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

                        snapshot = Image.open(os.path.join(input_dir,f)) 
                        snapshot = np.array(snapshot)
                        rows, cols = snapshot.shape

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

                        base[row_min:row_max, col_min:col_max, 0] = snapshot[:row_size, :col_size]
                        base[row_min:row_max, col_min:col_max, 1] = snapshot[:row_size, :col_size]
                        base[row_min:row_max, col_min:col_max, 2] = snapshot[:row_size, :col_size]

                        img = Image.fromarray(base)
                        draw = ImageDraw.Draw(img)
                        draw.rectangle([(col_min, row_min), (col_max, row_max)], outline='#00ffff')
                        draw.text((col_min+2, row_min+2), f"{track_index}", fill='white')
                        base = np.array(img)

        img.save(os.path.join(output_dir,'{x:05}.png'.format(x=x)))   

    out_file = '{output_dir}/reconstructed_mugshot_movie.mp4'.format(output_dir=output_dir)
    ffmpeg_command = ['ffmpeg', '-framerate', '5', '-i', output_dir + '/%05d.png', '-y', '-vf', 'format=yuv420p', out_file]
    cmd = SP.Popen(ffmpeg_command, stdout=SP.PIPE, stderr=SP.PIPE)
    out, err = cmd.communicate()
    cmd.send_signal(signal.SIGINT)
    cmd.wait()  
    
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--mugshot_folder',         required=True,
                                                    help="Path to folder of mugshot images")

    parser.add_argument('--reconstruction_folder',  required=True,
                                                    help="Folder to place reconstructed images and movie")

    parser.add_argument('--log_name',               default="reconstruct_mugshot_video.log",
                                                    help="Filename for the pipeline log")

    parser.add_argument('--log_folder',             default=op.join(op.abspath(op.dirname(__file__)), "logs"),
                                                    help="Folder path to store logs. Default is cli/logs")

    args = parser.parse_args()

    logger.setup_logger(args.log_name, args.log_folder)

    reconstruct_frames_to_video(args.mugshot_folder, args.reconstruction_folder)
