'''
Takes an AVI movie and converts to 2048x2048 frame snapshots.
'''
import argparse
import cv2
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--movie',  default=None,
                                help='Full path to avi movie')

parser.add_argument('--outdir', default=None,
                                help='Full path to frame output directory')
args = parser.parse_args()

vidcap = cv2.VideoCapture(args.movie)
success,image = vidcap.read()
count = 0
while success:
	image = image[0:2048, 0:2048]
	im = Image.fromarray(image)
	im.save(f"{args.outdir}/{count:05}_holo.tif")     
	success,image = vidcap.read()
	count += 1