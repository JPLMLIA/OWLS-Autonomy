"""
Implements diversity/similarity calculations for JEWEL
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from utils.logger import get_logger

# Global variable for logging
logger = get_logger()

# Note for developers: follow the example of `gaussian_similarity` to implement
# additional similarity functions. The outer wrapper function should take any
# necessary parameters and return a function that computes the pairwise
# similarities using a single `metadata` argument. If custom code is needed to
# parse parameters from the YAML configuration, define a function that takes a
# YAML dictionary containing configuration and returns a dictionary containing
# the loaded parameters, then assign this to the `load_params` attribute of the
# similarity wrapper function. If the parameters can be passed in as-is to the
# similarity wrapper function, then the `load_params` function does not need to
# be defined. Alternatively, the similarity function could be a class which as a
# `load_params` class method defined.

def gaussian_similarity(scale_factor=None):
    """
    Return a function that computes Gaussian similarity using the supplied scale
    factor, or the median distance as a heuristic if none is supplied.

    Parameters
    ----------
    scale_factor: float or None
        a scale factor to use in the Gaussian exponent; if None, the median
        pairwise distance is used as the scale factor

    Returns
    -------
    similarity_func: callable
        a function that takes a dict containing a array-like `dd` entry that
        holds diversity descriptor features in its rows, and returns an n-by-n
        array of pairwise similarities between the corresponding n ASDPs; values
        are guaranteed to be within the range [0, 1]
    """

    def similarity_func(metadata):
        dd = metadata['dd']

        # The `pdist` function computes the upper triangular entries of the
        # symmetric pairwise distance matrix. The `squareform` function converts
        # these entries to a standard symmetric square matrix with zero entries
        # along the diagonal. This requires less than half of the distance
        # function calculations as would be needed using `cdist(dd, dd)`.
        D = squareform(pdist(dd))

        if scale_factor is not None:
            gamma = scale_factor
        else:
            gamma = (1.0 / np.median(D))

        return np.exp(-(gamma * D)**2)

    return similarity_func


def load_similarity_func(config):
    """
    Loads a similarity function from a YAML config dict.

    Parameters
    ----------
    config: dict
        dictionary containing YAML configuration with a "name" entry for a
        similarity function defined within the global namespace of this module,
        and (optionally) a "parameters" entry containing a dict of parameters
        required by that similarity function

    Returns
    -------
    similarity_func: callable
        returns a parameterized similarity function corresponding to the
        provided configuration, or None if an error occurred during loading
    """
    sim_name = config.get('name', None)
    if sim_name is None:
        logger.warning(f'No similarity function name specified')
        return

    # Look for the function by name in the global namespace
    sim = globals().get(sim_name, None)
    if sim is None:
        logger.warning(f'Similarity function "{sim_name}" not found')
        return

    # Get parameters from config and load them with the custom similarity
    # function/class code if defined
    param_config = config.get('parameters', {})
    if hasattr(sim, 'load_params'):
        params = sim.load_params(param_config)
    else:
        params = param_config

    if params is None:
        return

    return sim(**params)
