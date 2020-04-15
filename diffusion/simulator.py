import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
from scipy.stats import norm
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point


def simulateTrajectories(number, meanLength = 5, ROI = None, trapList = None):
    """
    Generate a number of trajectories in a given environement

    Parameters
    ----------
    number (int)
        the number of wanted individual particles
    meanLength (int)
        average duration (in frames) of trajectories, from exponantial distribution
    ROI (Polygon)
        the region within which particles will diffuse (nucleus)
    trapList (Polygon)
        list of zones considered traps

    Returns
    -------
    Numpy array with columns [x, y, id]

    Notes
    -----
    We could create an environment class, otherwise we'll keep taking on extra arguments
    """

    # generate an array of numbers taken from the exponantial distribution
    # and round up to have the desired length of each simulated tracks
    lengthList = np.random.exponential(scale=5, size=100)
    lengthList = np.ceil(lengthList).astype(int)

    trackPopulation = []

    #TODO: figure out if there is a nicer way to do this
    #if no ROI nor traps are passed, use the simpler diffusing function
    #TODO: !!! this CRASHES if traplList but no ROI !!! is there a way to tell
    #a polygon to be the entire 2d plane ? -> apparently not
    #if there is either a ROI or trapList, use the other one
    if (ROI == None and trapList == None):
        for length in lengthList:
            startingPos = [np.random.normal(0, 1), np.random.normal(0, 1)]
            trackPopulation.append(simulateOneTraj(length, startingPos))
    else:
        for length in lengthList:
            startingPos = [np.random.normal(0, 1), np.random.normal(0, 1)]
            trackPopulation.append(simulateOneTraj(length, startingPos, ROI, trapList))
    return concatTracklets(trackPopulation)

def vanillaDiffusion(length, startingPos):
    """
    simulate one trajectory of a given length, from a starting position

    Parameters
    ----------
    TODO: complete that part

    Returns
    -------
    ndarray 

    Notes
    -----
    is that the optimal way to do this ? sort of a helper function
    here to handle the special case of no ROI nor traps passed in simulateTrajectories
    """
    X = []
    Y = []
    #put in the starting position
    X.append(startingPos[0])
    Y.append(startingPos[1])
    #for the desired length of the trajectory
    i = 1
    while (i<length):
        i += 1
        #draw the displacement 
        x = 0.5
        y = 0

        X.append(X[-1] + x)
        Y.append(Y[-1] + y)

    return np.column_stack((X,Y))


def simulateOneTraj(length, startingPos, ROI = None, trapList = None):
    """
    simulate one trajectory of a given length, from a starting position, 
    within specified geometry.

    Parameters
    ----------
    TODO: complete that part

    Returns
    -------
    ndarray 

    Notes
    -----
    Defaults settings are diffusion on a flat surface with no boundaries 
    """
    X = []
    Y = []
    #put in the starting position
    X.append(startingPos[0])
    Y.append(startingPos[1])
    particle = Point(X[0], Y[0])
    #for the desired length of the trajectory
    i = 1
    while (i<length):
        i += 1
        particle.x = X[-1]
        particle.y = Y[-1]
        #draw the displacement 
        x = 0.5
        y = 0
        particle.x += x
        particle.y += y
        #check if we are inside the ROI
        if (ROI.contains(particle)):
            X.append(X[-1] + x)
            Y.append(Y[-1] + y)
        else:
            #if we are not, we add the inverse of the displacement
            X.append(X[-1] - x)
            Y.append(Y[-1] - y)

        trap = isTrapped((X[-1], Y[-1]), trapList)
        #if isTrapped has returned something, it means the particle is indeed trapped
        if (trap != None):
            XinTrap, YinTrap = diffuseInTrap([X[-1], Y[-1]], 1, trap, length - 1 - i)
            X.extend(XinTrap)
            Y.extend(YinTrap)
            i += len(XinTrap)
        

    return np.column_stack((X,Y))

def diffuseInTrap(pos, Ptrap, trapZone, maxFrames):
    """
    simulates diffusion within a 'trap zone'. Diffusivity is
    reduced, if the particle diffuses out of the trap,
    it has a probability Ptrap of being put back inside

    Parameters
    ----------
    pos (tuple)
        the initial position of the particule
    Ptrap (float) 0 < Ptrap < 1
        the probability of being pulled back in on escape
    trapZone (Polygon)
        the position and shape of the trap zone
    maxFrames (int)
        maximum number of position to be calculated

    Returns
    -------
    ndarray
        the list of successive positon until the particle
        successfully escapes

    Notes
    -----
    May overshoot number of time points specified by user in
    diffuseWithTraps
    """
    X = []
    Y = []

    #put in the first coordinates
    X.append(pos[0])
    Y.append(pos[1])

    #draw
    x = 0.5
    y = 0

    diffuse = True
    currentFrame = 1 
    while(diffuse):
        if (trapZone.contains(Point(X[-1] + x, Y[-1] + y))):
                X.append(X[-1] + x)
                Y.append(Y[-1] + y)
        else:
            #draw from uniform distribution, if above Ptrap threshold : escape
            if (np.random.rand(1) > Ptrap):
                X.append(X[-1] + x)
                Y.append(Y[-1] + y)
                X.pop(0)
                Y.pop(0)
                return X, Y
            else:
                X.append(X[-1] - x)
                Y.append(Y[-1] - y)
        diffuse = (currentFrame <= maxFrames)
        currentFrame += 1

    X.pop(0)
    Y.pop(0)
    return X, Y

def isTrapped(position, listTraps):
    """
    Returns the Polygon in which the particle is trapped, None if
    the particle is free

    Parameters
    ----------
    position (float, tuple)
        the position of the particule to be checked
    listTraps (Polygon, list)
        list of all Polygons defined as traps

    Returns
    -------
    Polygon

    Notes
    -----
    #TODO:this is not how you do notes
    Initially isFree, but this way allows to have a single pythonic function
    instead of a function that returns a bool and the polygon in which the
    particle is trapped if it is. Here, it always returns a polygon, and it
    returning none means the particle is not trapped in any polygon.
    """
    point = Point(position) #careful with format of position (x, y)
    for trap in listTraps:
        #if any of the listed trap contains the point
        if (trap.contains(point)):
            print("I know I was trapped at"+str(position[0])+", "+str(position[1]))
            return trap
    return None

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

    #same logic for Y and ID of the particle 
    Y = [track[:,1] for track in trackPopulation]
    Y = np.concatenate(Y)
    partID = [[index] * len(track) for index, track in enumerate(trackPopulation)]
    partID = np.concatenate(partID)

    #we stack the 4 different list on top of each other and take the transpose
    #to have our usual form of data, which we return
    return np.vstack((X,Y, partID)).T



#this is supposed to be the function giving the correct dx, dy 
#given our parameters for time intervals and diffusivity
def displace(postion, diffusivity, fps):
    position = [0,0,0]
    time_interval = 1 / fps
    dx = np.random.normal(0, np.sqrt(2 * diffusivity * time_interval))
    dy = np.random.normal(0, np.sqrt(2 * diffusivity * time_interval))
    dz = np.random.normal(0, np.sqrt(2 * diffusivity * time_interval))
    position += [dx, dy, dz]
    return position

########################################################
##  GEOMETRY
########################################################

#ultimately, this would be the entry point to take in ROI
#from ImageJ and all other softwares people are using
def createPolygon(points):
    """
    Creates a polygon based on an ordered list of points
    
    Parameters
    ----------
    points: ordered list of points
        vertices are joined row by row
    
    Returns
    -------
    Polygon object from the shapely library
    """
    return Polygon(points)

def createCircle(r, center, nbPoints):
    """
    Creates a polygonal ROI for which all vertices are on
    the circle defined by its center and radius

    Parameters
    ----------
    r (float) : radius
        the radius for the circle
    center (tuple)
        coordinate of the center of the circle
    nbPoints (int)
        number of vertices that we want for the circle
    
    Returns
    -------
    Polygon object from the shapely library

    Notes
    -----
    This is not a perfect circle
    """
    pi = math.pi
    circle = [(math.cos(2*pi/nbPoints*x)*r,math.sin(2*pi/nbPoints*x)*r) for x in range(0,nbPoints)]
    circle = circle + np.array(center)
    return Polygon(circle)

#this is weird so far, and might turn into a class outright
#if we want to have traps to have specific behavior 
#like a diffusivity different from the rest of the nucleus
def createTrap(radius, center):
    """
    Creates a circle of radius and center, automatically with 50 points
    """
    return createCircle(radius, center, 50)