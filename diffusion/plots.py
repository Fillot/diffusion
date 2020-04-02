import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from diffusion import utils
import datetime
import warnings
from warnings import warn
import pandas as pd
from scipy.stats import norm


def displacementHistogram(Pos, bins = 20):
    """
    Arguments
    ---------
        
    Pos: pandas dataframe containing the position of the particule
            at each frame of form   x1  x2  x3 ... xN
                                t1|____|___|____|____|
                                t2|____|___|____|____|
                                t3|____|___|____|____|
                                ...
    bins (int): number of bins for the histogram

    Returns
    -------
    An histogram of the displacement between each frame for the particule

    Notes
    -----
    TODO: this takes in only one particule because of the way I simulate so far
    TODO: make it so that it takes in the standard format
    """
    disp = utils.displacements(Pos)
    plt.hist(disp, bins = bins, density=True)
    plt.show()
    print(_1gaussian(disp))
    #TODO: fit gaussian (or chi-square ?), two gaussians, arbitrary number
    #of guassians. Output residues, show best fit, stuff like that.

def _1gaussian(disp):
    """
    Arguments
    disp: a 1 dimensional list containing all displacements
    """
    mu, std = norm.fit(disp)
    return mu, std

def HansenHeatMap(Pos, bins = 0):
    """
    Displays a heatmap showing the anisotropy between two consecutive jumps
    depending on the two jumps respective lengths.
    Observed anisotropy can be revealing of a trapping mechanism.
    
    similar to fig 1. of Hansen, 2018 https://doi.org/10.1038/s41589-019-0422-3

    Parameters
    ----------
    Pos: numpy array containing the position of the particule
            at each frame of form   x1  x2  x3 ... xN
                                t1|____|___|____|____|
                                t2|____|___|____|____|
                                t3|____|___|____|____|
                                ...
    bins (int): number of bins for the histogram
    """
    #TODO:rewrite Simulations/BrownianMotionTest2.py implementation of this and test it.

def plotTraj(traj, cmap = None, ):
    """Plot the trajectories of multiple particles"""
    return 0
