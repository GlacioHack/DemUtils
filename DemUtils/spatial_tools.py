"""
DemUtils.spatial_tools.py contains statistical tools for robust of manipulation DEM data
"""
import numpy as np

def nmad(data:np.ndarray, nfact:float=1.4826)->float:
    """
    Calculate the normalized median absolute deviation (NMAD) of an array.

    :param data: input data
    :param nfact: normalization factor for the median absolute deviation; default is 1.4826

    :returns nmad: (normalized) median absolute deviation of data.
    """
    m = np.nanmedian(data)
    return nfact * np.nanmedian(np.abs(data - m))


def rmse(data:np.ndarray)->float:
    """
    Return root mean square of input data.
    :param data: differences to calculate root mean square of

    :returns rmse: root mean square error
    """
    r = np.sqrt(np.nanmean(np.asarray(data) ** 2))
    return r





