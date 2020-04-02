"""
This is the collection of function, sometimes performed elsewhere as well,
but who have in common to use machine learning under the hood.
"""

import numpy

def anomalousExponent(traj, train = False):
    """
    This is an implementation of a recurrent neural network used to extract
    the anomalous exponent alpha from a time series of position.

    taken from Bo, 2019     https://doi.org/10.1103/PhysRevE.100.010102

    Parameters
    ----------
    train (bool):   if true, the RNN will be trained using the data in traj
                    if false, use the pre-trained model
    
    Notes
    -----
    priority for development is VERY LOW
    """
    alpha = 0
    return alpha