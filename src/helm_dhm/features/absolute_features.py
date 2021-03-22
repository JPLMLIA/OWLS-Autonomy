"""
Functions that use data on a per track basis are housed here
"""
import logging
import numpy as np

from helm_dhm.features import feature_utils as utils


def speed_and_acceleration(track):
    """
    Calculates speed and acceleration features. First calculates speed,
    then leverages that information to calculate acceleration. By doing so as
    nested functions, memory can be reused while separating the logic for
    organization.

    Parameters
    ----------
    track: dict
        Dictionary of track data

    Returns
    -------
    features: dict
        Dictionary of features to add into the track's feature dict
    """
    def speed():
        """
        Calculates speed features

        Returns
        -------
        features: dict
            Dictionary of features to add into the track's feature dict
        """
        x    = track['X']
        y    = track['Y']
        
        vx = np.diff(x)
        vy = np.diff(y)
        speeds = np.sqrt(vx ** 2 +vy ** 2)

        if speeds.size > 0:
            vsum  = np.sum(speeds)
            vmax  = np.max(speeds)
            vmean = np.mean(speeds)

            if len(speeds) > 2:
                vstd = np.std(speeds)
            else:
                vstd = np.inf
        else:
            logging.info(f"Track {track['track_id']} is not long enough to extract speed features from")
            vsum  = 0
            vmax  = np.inf
            vmean = np.inf
            vstd  = np.inf

        # Autocorrelation
        if len(speeds) > 2:
            ac_lag1, ac_lag2 = utils.estimated_autocorrelation(speeds, lags=[15, 30])
        else:
            ac_lag1 = np.inf
            ac_lag2 = np.inf

        # Save the features
        features = {
            'track_length': vsum,
            'max_speed': vmax,
            'mean_speed': vmean,
            'stdev_speed': vstd,
            'autoCorr_speed_lag1': ac_lag1,
            'autoCorr_speed_lag2': ac_lag2,
        }

        # Return the speeds list to be used in the acceleration function
        return features, speeds

    def acceleration(speeds):
        """
        Calculates acceleration features

        Parameters
        ----------
        speeds: list of float
            List of speeds returned from speed()

        Returns
        -------
        features: dict
            Dictionary of features to add into the track's feature dict
        """
        # Calculate the accelerations
        accs = np.diff(speeds)

        if accs.size > 0:
            aabs  = np.abs(accs)
            amax  = np.max(aabs)
            amean = np.mean(aabs)

            if len(accs) > 2:
                astd = np.std(accs)
            else:
                astd = np.inf
        else:
            logging.info(f"Track {track['track_id']} is not long enough to extract acceleration features from")
            amax  = np.inf
            amean = np.inf

        # Autocorrelation
        if len(accs) > 2:
            ac_lag1, ac_lag2 = utils.estimated_autocorrelation(accs, lags=[15, 30])
        else:
            ac_lag1 = np.inf
            ac_lag2 = np.inf

        # Save the features
        features = {
            'max_accel'          : amax,
            'mean_accel'         : amean,
            'stdev_accel'        : astd,
            'autoCorr_accel_lag1': ac_lag1,
            'autoCorr_accel_lag2': ac_lag2,
        }

        return features

    # Run speed() first to get the speeds, then calculate acceleration()
    speed_feats, speeds = speed()
    accel_feats = acceleration(speeds)

    # Combine the features together to one dict and return
    features = dict(**speed_feats, **accel_feats)

    return features

def local_angle(track):
    """
    Calculates step angles in radians

    Parameters
    ----------
    track: dict
        Dictionary of track data

    Returns
    -------
    features: dict
        Dictionary of features to add into the track's feature dict
    """
    diffx  = np.diff(track['X'])
    diffy  = np.diff(track['Y'])
    angles = np.arctan2(diffy[1:], diffx[1:]) - np.arctan2(np.negative(diffy[:-1]), np.negative(diffx[:-1]))
    angles = np.abs(angles)

    if angles.size > 0:
        max_ang  = np.max(angles)
        mean_ang = np.mean(angles)
    else:
        logging.info(f"Track {track['track_id']} is not long enough to extract step angle features from")
        max_ang  = np.inf
        mean_ang = np.inf

    # Autocorrelation
    if len(angles) > 2:
        ac_lag1, ac_lag2 = utils.estimated_autocorrelation(angles, lags=[15, 30])
    else:
        ac_lag1 = np.inf
        ac_lag2 = np.inf

    # Save the features
    features = {
        'max_stepAngle'          : max_ang,
        'mean_stepAngle'         : mean_ang,
        'autoCorr_stepAngle_lag1': ac_lag1,
        'autoCorr_stepAngle_lag2': ac_lag2
    }

    return features

def displacement_vector(track):
    """
    Calculates the displacement vector between the X and Y of each frame

    Parameters
    ----------
    track: dict
        Dictionary of track data

    Returns
    -------
    features: dict
        Dictionary of features to add into the track's feature dict
    """
    diffx = np.diff(track['X'])
    diffy = np.diff(track['Y'])
    mean_disp_x = np.mean(diffx)
    mean_disp_y = np.mean(diffy)

    mean_mag = np.sqrt(mean_disp_x**2 + mean_disp_y**2)
    if mean_mag > 0:
        mean_disp_x /= mean_mag
        mean_disp_y /= mean_mag
        displacement = np.abs(np.arctan2(mean_disp_y, mean_disp_x))
    else:
        displacement = 0

    # Save the features
    features = {
        'mean_disp_x'    : mean_disp_x,
        'mean_disp_y'    : mean_disp_y,
        'mean_disp_angle': displacement
    }

    return features
