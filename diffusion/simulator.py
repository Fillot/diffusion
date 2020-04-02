import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
from scipy.stats import norm

def simulateOneTraj(steps=20, radius=10):
    """
    simulate one trajectory of a particule for a given number of steps, confined in a given radius
    """
    X = []
    Y = []
    Z = []
    #put the first steps otherwise X[-1] will Index out of Range
    X.append(np.random.normal(0, 1))
    Y.append(np.random.normal(0, 1))
    Z.append(np.random.normal(0, 1))
    
    #check if we are in the max radius.
    for _ in range(steps):
        outOfBounds = 0
        x = np.random.normal(0, 0.5)
        y = np.random.normal(0, 0.5)
        z = np.random.normal(0, 0.5)
        # print(x, y, z)
        #if, when adding this step, we are outside the boundary
        if np.linalg.norm((X[-1] + x,Y[-1] + y,Z[-1] + z)) >= radius:
            #then subtract the step, 
            X.append(X[-1] - x)
            Y.append(Y[-1] - y)
            Z.append(Z[-1] - z)
            outOfBounds += 1
        else:
            X.append(X[-1] + x)
            Y.append(Y[-1] + y)
            Z.append(Z[-1] + z)
        
    Pos = np.column_stack((X,Y,Z))
#    Pos = pd.DataFrame(Pos)
    
    return Pos

def concatTracklets(trackPopulation):
    """
    Concatenate a population of N tracks into the usual format

    input:
        list of array, with n lines, and d columns (d = dimension)
                x1  x2  ... ...xd
            t1|____|___|____|____|
            t2|____|___|____|____|
                ..............
            tn|____|___|____|____|

     ouput:
              x1  x2  ... ...id
            |____|___|____|__1_|
            |____|___|____|__1_|
            |____|___|____|__2_|
            ....................
            |____|___|____|__N_|
                

    TODO: should drop tracks of length < 2
    """
    #take the first column of each array in trackPop, then concatenate into a
    #single continuous array of a single list
    X = [track[:,0] for track in trackPopulation]
    X = np.concatenate(X)

    #same logic for Y, Z and ID of the particle 
    Y = [track[:,1] for track in trackPopulation]
    Y = np.concatenate(Y)
    # Z = [track[:,2] for track in trackPopulation]
    # Z = np.concatenate(Z)
    partID = [[index] * len(track) for index, track in enumerate(trackPopulation)]
    partID = np.concatenate(partID)

    #we stack the 4 different list on top of each other and take the transpose
    #to have our usual form of data, which we return
    return np.vstack((X,Y, partID)).T

def simulateTrajectories(size = 100, dimension = 2, meanLength = 5, radius = 10):
    """
    Generate a number of trajectories of varying length following
    an exponantial distribution, within a radius (~nucleus) of a given
    radius

    Parameters
    ----------
    size (int) : number of individual tractories to simulate
    dimension (int) : number of dimension of freedom for movement of particle
    meanLength (int) : mean value/scale of the exponential distribution
    radius (float) : radius of the container

    Returns
    -------
    Numpy array with columns [x, y, id] ou [x, y, z, id]
    """

    # generate an array of numbers taken from the exponantial distribution
    # and round up to have the desired length of each simulated tracks
    array = np.random.exponential(scale=5, size=100)
    array = np.ceil(array).astype(int)

    trackPopulation = []

    for track in array:
        pos = simulateOneTraj(track, radius)
        trackPopulation.append(pos)

    Pos = concatTracklets(trackPopulation)

    #if we only want 2 dim, we drop column 2
    if (dimension == 2):
        Pos = np.delete(Pos, 2, 1)

    return Pos

def displace(postion, diffusivity, fps):
    time_interval = 1 / fps
    dx = np.random.normal(0, np.sqrt(2 * diffusivity * time))
    dy = np.random.normal(0, np.sqrt(2 * diffusivity * time))
    dz = np.random.normal(0, np.sqrt(2 * diffusivity * time))
    position += [dx, dy, dz]
    return position
