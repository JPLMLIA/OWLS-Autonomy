"""
Utility functions for absolute_features.py and relative_features.py
"""
import numpy as np

def estimated_autocorrelation(x, lags=None):
    """
    Computes statistical autocorrelation of a time-series/sequence

    Parameters
    ----------
    x: list of int or float
        1D sequence of int or floats
    lags: list of int
        List of lag indices to return. If None, returns all.

    Returns
    -------
    result: list
        Biased autocorrelation function.
    """
    xlen = len(x)
    acf  = []

    for lag in range(xlen):
        # Get subseries of given lag
        y1 = x[:(xlen - lag)]
        y2 = x[lag:]
        try:
            # Calculate covariance, then normalize with sample variance
            xmean    = np.mean(x)
            sum_prod = np.sum((y1 - xmean) * (y2 - xmean))
            norm     = sum_prod / (xlen * np.var(x))
            acf.append(norm)
        except TypeError:
            raise TypeError('Input array values must be only int or float type')

    # If only a specific subset of lags is requested, select those
    if lags:
        return [acf[l] if l < len(acf) else np.inf for l in lags]
    else:
        return acf