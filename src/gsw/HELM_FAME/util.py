from collections import OrderedDict
import json
import logging
import os
import yaml
import csv

import numpy as np

from utils.track_loaders import transpose_xy_rowcol
from mpltools import color as C
from typing import Dict, Optional, List, Any
from PIL import Image, ImageDraw, ImageFont

def get_rainbow_black_red_colormap():
    colors = (
        (0, 0, 0),  # k
        (1, 0, 0),  # r
        (1, .5, 0),  # o
        (1, 1, 0),  # y
        (0, 1, 0),  # g
        (0, 0, 1),  # b
        (1, 0, 1),  # v
    )

    return C.LinearColormap("rainbow_black_red", colors)



def create_y_range(motile_count,
                   non_motile_count,
                   auto_motile_count,
                   auto_non_motile_count):
    """
    Generate the y range on the motility bar

    :param motile_count: the amount of motile life at this frame
    :param non_motile_count: the amount of non motile life at this frame
    :param auto_count: the amount of hand labeled tracks at this frame
    :return: the amount of life at this frame
    """
    y = []
    if motile_count is not None:
        y += motile_count
    if non_motile_count is not None:
        y += non_motile_count
    if auto_motile_count is not None:
        y += auto_motile_count
    if auto_non_motile_count is not None:
        y += auto_non_motile_count
    _min = None
    _max = None
    if y:
        _min = min(y)
        _max = max(y) + 1
    return [_min, _max]

def count_motility(track_points_list):
    """
    Give a count of motility for each frame

    :param track_points_list: the list to iterate over to find motility
    :return: the count of motile and non motile from trackpoints list
    """
    _motile = 0
    _non_motile = 0
    for trackPoint in track_points_list:
        if trackPoint["mobility"] is None:
            _non_motile += 1
        else:
            if trackPoint["mobility"].lower() == "motile": 
                _motile += 1
            else:
                _non_motile += 1

    return [_motile, _non_motile]


def max_particle_intensity(track_points_list):
    """
    Given a list of particles in a frame, return the maximum intensity
    If the object intensity is 3-channel, return the maximum mean intensity

    :param track_points_list: list of particle dictionaries that exist in the current frame
    :return: the intensity of the higest intensity particle in the scene
    """
    intensities = []
    for trackPoint in track_points_list:
        intensities.append(trackPoint["intensity"]) 
    if intensities:
        mean_intensities = [np.mean(i) for i in intensities]
        return max(mean_intensities)
    else:
        return None


def load_in_autotrack(trackDirPath, SCALE_FACTOR=1.0, TRACK_FILE_EXT="json"):
    """Load an predicted track"""

    trackDict = OrderedDict()
    frameDict = OrderedDict()
    #count = 0

    for x in os.listdir(trackDirPath):

        path = os.path.join(trackDirPath, x)

        # skip non file
        if not os.path.isfile(path):
            continue
        # skip non track file
        a = -len(TRACK_FILE_EXT)
        if x[-len(TRACK_FILE_EXT):] != TRACK_FILE_EXT:
            continue

        """
        tmpNumber = 1000
        if count % tmpNumber == 0:
            logging.info("loading %sth track file %s and possible next %s track files" % (count, path, tmpNumber))
        count += 1
        """

        f = open(path, "r")
        d = json.load(f)
        f.close()

        trackNumber = d["Track_ID"]
        timeList = d["Times"]
        classification = d["classification"]
        intensities = d["Particles_Max_Intensity"]
        sizes = d["Particles_Size"]

        for i in range(len(timeList)):

            frameNumber = timeList[i]
            position = d["Particles_Position"][i]
            if position == None:
                continue

            # Transpose coordinates (from matrix coords to XY coords)
            x, y = transpose_xy_rowcol(np.array(position).reshape(1, 2))[0]

            x = float(x) * SCALE_FACTOR
            y = float(y) * SCALE_FACTOR

            trackPoint = {"location": [int(x), int(y)], "frame": frameNumber, "mobility": classification, "size": sizes[i], "intensity":intensities[i]}
            if trackNumber not in trackDict:
                trackDict[trackNumber] = []
            trackDict[trackNumber].append(trackPoint)

            trackPoint = {"location": [int(x), int(y)], "track": trackNumber, "mobility": classification, "size": sizes[i], "intensity":intensities[i]}
            if frameNumber not in frameDict:
                frameDict[frameNumber] = []
            frameDict[frameNumber].append(trackPoint)

    return [trackDict, frameDict]


def load_in_track(trackFilePath):
    '''Reads hand label files for visualization'''

    trackDict = OrderedDict()
    frameDict = OrderedDict()

    with open(trackFilePath, 'r') as csvFile:

        csvReader = csv.reader(csvFile)

        # skip header
        next(csvReader)

        for row in csvReader:

            #trackName, x, y, frameName, species, movementType, sizeType = row
            frameName, trackName, x, y, movementType = row
            trackNumber = int(trackName)

            frameNumber = int(frameName) - 1

            trackPoint = {"location": [round(float(x)), round(float(y))], "frame": frameNumber, 
                          "mobility": movementType, "size": 0}

            if trackNumber not in trackDict:
                trackDict[trackNumber] = []

            trackDict[trackNumber].append(trackPoint)

            trackPoint = {"location": [round(float(x)), round(float(y))], "track": trackNumber,
                          "mobility": movementType, "size": 0}

            if frameNumber not in frameDict:
                frameDict[frameNumber] = []

            frameDict[frameNumber].append(trackPoint)

    return [trackDict, frameDict]

class Track(object):

    def __init__(self, number, points):
        self.number = number

        # this track consists of these points
        self.points = points

        # this track covers these frames
        self.frames = []
        for point in self.points:
            frameNumber = point["frame"]
            self.frames.append(frameNumber)
        self.frames.sort()

    def covers_frame(self, frameNumber):

        if frameNumber < self.frames[0]:
            return False
        if frameNumber > self.frames[-1]:
            return False
        return True


