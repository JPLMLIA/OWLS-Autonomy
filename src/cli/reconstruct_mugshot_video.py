'''
Command line interface to reconstruct a folder of mugshots into source images
'''
import sys
sys.path.append("../")

import logging
import argparse
import os
import os.path as op
import glob

import numpy      as np
import subprocess as SP

from PIL               import Image

def reconstruct_frames_to_video(input_dir, output_dir):
    
    files = list(glob.glob(input_dir + '/*.png'))
    files = [x.split("/")[-1] for x in files]

    # TODO - +1 for off by one error?
    total_frames = max([int(x.split("_")[2]) for x in files]) + 1
    
    split = files[0].rstrip(".png").split("_")
    split = [int(x) for x in split]
    recon_row = split[0]
    recon_col = split[1]
    mugshot_width = split[5] * 2

    if not op.exists(output_dir):
        os.makedirs(output_dir)

    for x in range(0, total_frames):
        snapshot = np.zeros((recon_row, recon_col))
        img = Image.fromarray(snapshot, 'L')
        img.save(os.path.join(output_dir,'{x:05}.png'.format(x=x)))
    
    for file in files:
        split = file.rstrip(".png").split("_")
        split = [int(x) for x in split]

        snapshot = Image.open(os.path.join(input_dir,file)) 
        snapshot = np.array(snapshot)

        base = Image.open(os.path.join(output_dir, "{base_frame:05}.png").format(base_frame=split[2]))
        base = np.array(base)

        base[split[3]:split[3] + mugshot_width, split[4]:split[4] + mugshot_width] = snapshot
        img = Image.fromarray(base, 'L')
        img.save(os.path.join(output_dir,'{x:05}.png'.format(x=split[2])))
    
    out_file = '{output_dir}/reconstructed_mugshot_movie.avi'.format(output_dir=output_dir)
    ffmpeg_command = ('ffmpeg -framerate 5 -i ' + output_dir + '/%05d.png -y -vf format=yuv420p -vcodec rawvideo -qp 0 -preset veryslow ' + '"{}"'.format(out_file))
    SP.call(ffmpeg_command, shell=True)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--mugshot_folder',         required=True,
                                                    help="Path to folder of mugshot images")

    parser.add_argument('--reconstruction_folder',  required=True,
                                                    help="Folder to place reconstructed images and movie")

    args = parser.parse_args()

    reconstruct_frames_to_video(args.mugshot_folder, args.reconstruction_folder)
