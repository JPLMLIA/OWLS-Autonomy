import os
import os.path as op
import shutil
from pathlib import Path
from glob import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from multiprocessing import Pool

from skimage.registration import phase_cross_correlation
from itertools import repeat
from functools import partial
import csv
import subprocess
import signal

def read_image(path, size):
    with open(path, 'rb') as f:
        return Image.frombytes('L', size, f.read())

def raw2png(p, out):
    if not op.isfile(op.join(out, Path(p).stem + ".png")):
        rawim = read_image(p, (2048, 2048))
        rawim.save(op.join(out, Path(p).stem + ".png"))

def get_shift(currfn, basefn):
    basearr = np.array(Image.open(basefn))
    currarr = np.array(Image.open(currfn))
    shift, _, _ = phase_cross_correlation(basearr, currarr, upsample_factor=10)
    return shift

def get_pixdiff(curridx, pngs):
    prevarr = np.array(Image.open(pngs[curridx-1 if curridx-1 >=0 else 0])).astype(float)
    currarr = np.array(Image.open(pngs[curridx])).astype(float)
    return np.mean(np.abs(prevarr - currarr))

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(
        description="Generate debug products for DHM vibration."
    )

    parser.add_argument('exp',  help='directory path to experiment.')
    parser.add_argument('--res',  help='resolution of pixel shifts.',
                                default=0.1,
                                type = float)

    args = parser.parse_args()

    # read timestamp
    ts_file = op.join(args.exp, 'timestamps.txt')
    with open(ts_file, 'r') as f:
        ts = [x.strip().split() for x in f.readlines()]
    
    database = {}
    for row in ts:
        database[int(row[0])] = {
            'time': row[1],
            'date': row[2],
            'sec': row[3]
        }

    # get raw paths
    expdir = args.exp
    expraws = sorted(glob(op.join(expdir, 'raw', '*.raw')))

    # raw to png conversion
    print("[INFO] Converting RAW to PNG")
    pngdir = op.join(expdir, 'png')
    if not op.isdir(pngdir):
        os.mkdir(pngdir)

    with Pool(8) as pool:
        _ = list(tqdm(pool.imap(partial(raw2png, out=pngdir), expraws), total=len(expraws), desc='RAW2PNG'))
    
    # shifts
    print("[INFO] Measuring shifts")
    exppngs = sorted(glob(op.join(expdir, 'png', '*.png')))
    with Pool(8) as pool:
        shifts = np.array(list(tqdm(pool.imap(partial(get_shift, basefn=exppngs[0]), exppngs), total=len(exppngs), desc='PCC')))

    for p, s in zip(exppngs, shifts):
        database[int(Path(p).stem)]['shift0'] = s[0]
        database[int(Path(p).stem)]['shift1'] = s[1]

    # intensity diffs
    print("[INFO] measuring pixeldiffs")
    with Pool(8) as pool:
        diffs = np.array(list(tqdm(pool.imap(partial(get_pixdiff, pngs=exppngs), range(len(exppngs))), total=len(exppngs), desc='pixdiff')))
    
    for p, d in zip(exppngs, diffs):
        database[int(Path(p).stem)]['pixdiff'] = d

    # CSV

    print("[INFO] Exporting results")

    with open(op.join(expdir, f"{Path(expdir).stem}.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['frameID', 'time', 'date', 'sec', 'shift0', 'shift1', 'pixdiff'])
        for fid in sorted(database.keys()):
            writer.writerow([
                fid,
                database[fid]['time'],
                database[fid]['date'],
                database[fid]['sec'],
                database[fid]['shift0'] if 'shift0' in database[fid] else "",
                database[fid]['shift1'] if 'shift1' in database[fid] else "",
                database[fid]['pixdiff'] if 'pixdiff' in database[fid] else ""
            ])

    # plots
    print("[INFO] Generating figure")

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(10,7.5))

    dkeys = sorted(list(database.keys()))
    plotx = []
    plot1y = []
    plot2y = []
    plot3y = []
    for x in tqdm(dkeys):
        if 'shift0' in database[x] and 'shift1' in database[x] and 'pixdiff' in database[x]:
            plotx.append(x)
            plot1y.append(database[x]['shift0'])
            plot2y.append(database[x]['shift1'])
            plot3y.append(database[x]['pixdiff'])

    ax1.plot([min(plotx), max(plotx)], [0,0], c='gray', ls='--', alpha=0.5)
    ax1.plot(plotx, plot1y)
    ax1.set_title("ax0 offset")
    ax1.set_ylabel("subpixels")

    ax2.plot([min(plotx), max(plotx)], [0,0], c='gray', ls='--', alpha=0.5)
    ax2.plot(plotx, plot2y)
    ax2.set_title("ax1 offset")
    ax2.set_ylabel("subpixels")
    
    ax3.plot([min(plotx), max(plotx)], [0,0], c='red', ls='--', alpha=0.5)
    ax3.plot([min(plotx), max(plotx)], [2.4, 2.4], c='red', ls='--', alpha=0.5)
    ax3.plot(plotx, plot3y)
    ax3.set_title("pixel diff")
    ax3.set_ylabel("intensity diff")
    ax3.set_xlabel("frames")

    fig.suptitle(Path(args.exp).stem)
    fig.savefig(op.join(args.exp, f"{Path(args.exp).stem}_debug.png"), dpi=300, facecolor='white', transparent=False)
    
    print("[INFO] Figure saved to", f"{Path(args.exp).stem}_debug.png")



    # video
    print("[INFO] generating video")
    ffmpeg_input = op.join(pngdir, "*.png")
    #ffmpeg_command = ['ffmpeg', '-framerate', '15', '-i', ffmpeg_input, '-y', f"{Path(avail_exps[EXP_ID]).parent.stem}.mp4"]
    ffmpeg_command = ['ffmpeg', '-framerate', '15', '-pattern_type', 'glob', '-i', ffmpeg_input, '-y', op.join(Path(expdir), f"{Path(expdir).stem}.mp4")]

    cmd = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = cmd.communicate()
    cmd.send_signal(signal.SIGINT)
    cmd.wait()