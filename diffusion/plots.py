import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from diffusion import utils
import datetime
import warnings
from warnings import warn
import pandas as pd
from scipy.stats import norm
import matplotlib.collections as mcoll


def displacementHistogram(traj, bins = 20, pos_columns=['x', 'y']):
    """
    Arguments
    ---------
    bins (int): number of bins for the histogram

    Returns
    -------
    An histogram of the displacement between each frame for the particule

    Notes
    -----
    TODO: this takes in only one particule because of the way I simulate so far
    TODO: make it so that it takes in the standard format
    """
    disp = utils.displacements(traj, pos_columns = pos_columns)
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

def HansenHeatMap(traj, minLength, maxLength, nbBins = 11):
    """
    Plots the heatmap of anisotropy depending on the length of previous and next jump.
    following Hansen, 2020 https://www.doi.org/10.1038/s41589-019-0422-3

    Arguments
    ---------
    traj        DataFrame
    minLength   float
    maxLength   float
    nbBins      int
    """
    #TODO:beautify. 
    #TODO:Weird padding to get rid of. 
    #TODO:Test. 
    #TODO:Make unit test
    #TODO:some bins numbers come out with a thousand zeros
    #TODO:https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
    #follow this to make beautiful heatmaps and/or see seaborn library
    """Creation of a numpy array of the jump length
    Note: this is for a single continuous track. Code
    has to be adapted for multiple short tracks eventually"""
    vectors = utils.get_vectors(traj)
    jumpLength = utils.displacements(traj)
    angles = utils.angles(vectors)
    jumpLength = np.array(jumpLength) #required for digitized to work
    #problem (or not ?), angle_between outputs NaN when computing the angle 
    # between the 0 vector and something else
    # test = utils.angle_between((1,1),(0,0))

    """Creation of a 2D array used for the heat map. What the columns and lines
    represent in terms of actual dimension is set by the number of bins we want,
    and their range. So far, only linear segmentation"""
    bins = np.linspace(minLength, maxLength, nbBins)
    digitized = np.digitize(jumpLength, bins)
    # bin_means = [jumpLength[digitized == i].mean() for i in range(1, len(bins))]

    heatmap30_150 = np.zeros((len(bins)+1,len(bins)+1))
    heatmapAllJumps = np.zeros((len(bins)+1,len(bins)+1))

    #iterating through the digitized array of jumps
    #if, for the current jump, angle(jump) anisotrope
    #then we'll tick the box corresponding to
    #line : bin to which this jump belongs,
    #column : bin to which the next jump belongs
    for jmp in range(len(digitized)-1):
        if (angles[jmp]>150 or angles[jmp]<30):
            heatmap30_150[digitized[jmp], digitized[jmp+1]] += 1
            heatmapAllJumps[digitized[jmp], digitized[jmp+1]] +=1
        else:
            heatmapAllJumps[digitized[jmp], digitized[jmp+1]] += 1


    #this reads as follows : element-wise division of heatmap30_150 by heatmapAllJumps
    #                       out = where the results will be stored
    #                       where = do the division only where heatmapAllJumps is not 
    #                               0, otherwise just copy number from heatmap30_150
    heatmapAnisotropy = np.divide(heatmap30_150, heatmapAllJumps, out=np.zeros_like(heatmap30_150), where=heatmapAllJumps!=0)

    """
    Plot creation of the heatmap 
    """
    fig, ax = plt.subplots()
    im = ax.imshow(heatmapAnisotropy)
    fig.tight_layout()
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(bins)))
    ax.set_yticks(np.arange(len(bins)))
    # ... and label them with the respective list entries
    #if we take numerical values for the labels, matplotlib 
    # wants to scale them to the axes
    labels = [str(bin) for bin in bins] 
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # for i in range(len(bins)):
    #     for j in range(len(bins)):
    #         text = ax.text(j, i, heatmapAnisotropy[i, j],
    #                        ha="center", va="center", color="w")

    ax.set_title("Anisotropy of the jump depending on the jump length")
    fig.tight_layout()
    plt.show()

def plotTraj(traj, pos_columns = ['x', 'y']):
    
    unstacked = traj.set_index(['particle', 'frame'])[pos_columns].unstack()
    ax = plt.gca()
    for _, trajectory in unstacked.iterrows():
        ax.plot(trajectory['x'], trajectory['y'])
    plt.show()

def trajLength(traj):
    """Plots the lengths of the trajectories recorded in a histogram"""
    lengthOfTraj = []
    for pid, ptraj in traj.reset_index(drop=True).groupby('particle'):
        lengthOfTraj.append(len(ptraj))
    lengthOfTraj = np.array(lengthOfTraj)
    bins = np.arange(1, lengthOfTraj.max()+2,1) -0.5
    plt.hist(lengthOfTraj, bins = bins)
    plt.show()

def colorline(
    x, y, z=None, cmap=plt.get_cmap('viridis'), norm=plt.Normalize(0.0, 1.0),
        linewidth=1, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = _make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc

def plotSingleTraj(traj):
    axes = plt.gca()
    axes.set_xlim(traj['x'].min()-1,traj['x'].max()+1)
    axes.set_ylim(traj['y'].min()-1,traj['y'].max()+1)
    colorline(traj['x'], traj['y'])

def _make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments