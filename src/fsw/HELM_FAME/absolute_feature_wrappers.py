"""
Wrappers to absolute feature functions. These wrappers accept tracks jsons as
input and then call the appropriate underlying function.
"""
import logging

import numpy as np

from fsw.HELM_FAME import absolute_features


def speed_and_acceleration(track):
    points = np.array(track['Particles_Estimated_Position'])
    times = np.array(track['frame'])
    return absolute_features.speed_and_acceleration(points, times)


def local_angle_2D(track):
    points = np.array(track['Particles_Estimated_Position'])
    return absolute_features.local_angle_2D(points)


def displacement(track):
    points = np.array(track['Particles_Estimated_Position'])
    return absolute_features.displacement(points)


def bbox_sizes(track):
    valid_bboxes = [bbox for bbox in track['Particles_Bbox'] if bbox]  # Remove `None`s
    bbox_dims = [bbox[1] for bbox in valid_bboxes]  # Get just width/height of each bbox
    return absolute_features.bbox_size(np.array(bbox_dims))


def msd_slope(track):
    points = np.array(track['Particles_Estimated_Position'])
    time_diffs = np.diff(np.array(track['frame']))

    # Make sure we only see 1 time step interval over lifetime of track
    if len(np.unique(time_diffs)) != 1:
        logging.warning('Multiple time intervals encountered. Using mean in MSD calculation.')
        tau = np.mean(time_diffs)
    else:
        tau = time_diffs[0]

    # Look for flow info in track, otherwise assume no flow
    flow_offset = track.get('flow_offset', np.zeros_like(points[0]))

    return absolute_features.msd_slope(points, tau, flow_offset)