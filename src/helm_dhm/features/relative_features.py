"""
Functions that leverage data from all tracks are housed here
"""
import numpy as np

def relative_speed(tracks):
    """
    Calculates the ratio of a track's mean relative speed to the average mean
    relative speed of all tracks.

    Parameters
    ----------
    tracks: list of dict
        A list of track dictionaries

    Returns
    -------
    features: dict
        Dictionary of {track: {new_features_dict}} that will be iterated over
        to update the track data upon return

    Notes
    -----
    "rel_speed" is the ratio of a track's mean relative speed to the average mean
    relative speed of all tracks in the movie
    """
    speeds  = [track['mean_speed'] for track in tracks if np.isfinite(track['mean_speed'])]
    relmean = np.mean(speeds)

    # Store track features in a feature dict to return
    features = {}
    for track in tracks:
        # Create the feature dictionary for this track
        feats = features[track['track']] = {}
        if relmean > 0 and np.isfinite(track['mean_speed']):
            feats['rel_speed'] = track['mean_speed'] / relmean
        else:
            feats['rel_speed'] = np.inf

    return features

def relative_direction(tracks):
    """
    Calculates relative direction features. See notes

    Parameters
    ----------
    tracks: list of dict
        A list of track dictionaries

    Returns
    -------
    features: dict
        Dictionary of {track: {new_features_dict}} that will be iterated over
        to update the track data upon return

    Notes
    -----
    "rel_theta_displacement" is the directional difference in radians between the
    vector forming a track's mean direction and the average of the mean
    direction of all tracks in a movie.

    "rel_dir_dot" is the dot product of the two vectors used to calculate
    "rel_theta_displacement"
    """
    # Store track features in a feature dict to return
    features = {}

    # Calculate the average mean_disp_angle, ignore infs
    disps   = [track['mean_disp_angle'] for track in tracks if np.isfinite(track['mean_disp_angle'])]
    reldisp = np.mean(disps)

    # Calculate the average mean_disp_x and mean_disp_y, ignore infs
    mean_disp_xs     = [track['mean_disp_x'] for track in tracks if np.isfinite(track['mean_disp_x'])]
    mean_disp_ys     = [track['mean_disp_y'] for track in tracks if np.isfinite(track['mean_disp_y'])]
    mean_mean_disp_x = np.mean(mean_disp_xs)
    mean_mean_disp_y = np.mean(mean_disp_ys)

    # Calculate relative mean_disp_angle and dot product for each track
    for track in tracks:
        feats = features[track['track']] = {}

        mean_disp_x = track['mean_disp_x']
        mean_disp_y = track['mean_disp_y']
        if np.isfinite([mean_disp_x, mean_disp_y]).all():
            feats['rel_theta_displacement'] = track['mean_disp_angle'] - reldisp
            feats['rel_dir_dot'] = np.abs(np.dot([mean_disp_x, mean_disp_y], [mean_mean_disp_x, mean_mean_disp_y]))

    return features
