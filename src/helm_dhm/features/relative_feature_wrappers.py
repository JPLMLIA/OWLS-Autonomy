"""
Wrappers to relative feature functions. These wrappers accept tracks jsons as
input and then call the appropriate underlying function.
"""
import numpy as np

from helm_dhm.features import relative_features


def relative_speeds(track_group):
    group_speeds = [track['speed_mean'] for track in track_group
                    if np.isfinite(track['speed_mean'])]
    return relative_features.relative_speeds(group_speeds)


def relative_direction_feats(track_group):
    mean_displacements = [(t['disp_mean_v'], t['disp_mean_h'])
                          for t in track_group
                          if np.all(np.isfinite([t['disp_mean_v'], t['disp_mean_h']]))]
    return relative_features.relative_direction_feats(mean_displacements)