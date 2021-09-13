"""
Functions that leverage data from all tracks are housed here
"""
import numpy as np


def relative_speeds(group_speeds):
    """
    Calculates the ratio of a each track's speed to the mean of all other
    tracks using leave-one-out loop.

    Parameters
    ----------
    group_speeds: iterable
        A list of track speeds from all tracks in an experiment

    Returns
    -------
    features: list of dict
        Speed for track of interest under the `rel_speed` key
    """

    # Store track features in a feature dict to return
    features = []

    # If only a single track, we can't compute relative features
    if len(group_speeds) == 1:
        return[{'rel_speed': np.inf}]

    for si, speed in enumerate(group_speeds):
        # Need to get mean of all OTHER tracks
        rel_mean = np.mean([other_speed for osi, other_speed in enumerate(group_speeds)
                            if np.isfinite(other_speed)
                            and osi != si])

        # Create the feature dictionary for this speed
        if not rel_mean:
            features.append({'rel_speed': np.inf})
        elif rel_mean <= 0 or not np.isfinite(speed):
            features.append({'rel_speed': np.inf})
        else:
            features.append({'rel_speed': speed / rel_mean})

    return features


def relative_direction_feats(mean_displacements):
    """
    Calculates relative directional features to estimate how each track compares
    to the rest.

    Parameters
    ----------
    mean_displacements: list of tuple
        List of tuples where each tuple contains the
        (mean_vertical_disp, mean_horizontal_disp) for one track in matrix
        coords (0th coordinate increases downward, 1st coordinate increases rightward)

    Returns
    -------
    features: list of dict
        Dictionary containing metrics describing how each list differs in
        direction of travel from the rest. This includes cosine similarity and
        angle difference (in radians). See Notes.

    Notes
    -----
    "rel_disp_cosine_similarity" gives the cosine similarity (on interval
    [0, 1]) between the track of interest's mean displacement vector and the
    mean of all other mean track displacement vectors.

    "rel_step_angle" gives the difference between the mean step angle for a
    track of interest and the mean of all other mean track step angles
    """

    all_track_feats = []

    # If only a single track, we can't compute relative features
    if len(mean_displacements) == 1:
        return[{'rel_disp_cosine_similarity': np.inf,
                'rel_step_angle': np.inf}]

    for doi, disp in enumerate(mean_displacements):
        track_feats = {}

        # Get mean displacement for track of interest if finite
        if np.all(np.array(disp) == 0) or not np.all(np.isfinite(disp)):
            track_feats['rel_disp_cosine_similarity'] = np.inf
            track_feats['rel_step_angle'] = np.inf
            all_track_feats.append(track_feats)
            continue

        # Get the mean displacement averaged across all other tracks
        disp_other_tracks = np.array(mean_displacements[0:doi] + mean_displacements[doi+1:])
        disp_mean_other_tracks = np.nanmean(disp_other_tracks, axis=0)

        if np.all(disp_mean_other_tracks == 0) or not np.all(np.isfinite(disp_mean_other_tracks)):
            track_feats['rel_disp_cosine_similarity'] = np.inf
            track_feats['rel_step_angle'] = np.inf
            all_track_feats.append(track_feats)
            continue

        # Calculate cosine similarity between track of interest and other tracks
        disp_track_of_interest = np.array(disp)

        dot_product = np.dot(disp_track_of_interest, disp_mean_other_tracks)
        norm_factor = np.linalg.norm(disp_track_of_interest) * np.linalg.norm(disp_mean_other_tracks)
        cosine_similarity = dot_product / norm_factor

        if np.all(disp_track_of_interest == 0) and np.all(disp_mean_other_tracks == 0):
            # Special case where both features are 0, so tracks are actually similar
            track_feats['rel_disp_cosine_similarity'] = 1
        elif norm_factor == 0:
            # Special case where only one feature is zero, so features are dissimilar
            track_feats['rel_disp_cosine_similarity'] = 0
        else:
            track_feats['rel_disp_cosine_similarity'] = cosine_similarity

        # Calculate the difference in step angle between track of interest and other tracks
        # Convert vertical displacement from matrix to X,Y coords by multiplying 0th coord by -1
        # (Conversion not strictly necessary since we're computing a diff, but stays consistent with rest of code)
        step_angle_track_of_interest = np.arctan2(-1 * disp_track_of_interest[0], disp_track_of_interest[1])
        step_angle_global_mean = np.arctan2(-1 * disp_mean_other_tracks[0], disp_mean_other_tracks[1])
        track_feats['rel_step_angle'] = np.abs(step_angle_track_of_interest - step_angle_global_mean)

        all_track_feats.append(track_feats)

    return all_track_feats