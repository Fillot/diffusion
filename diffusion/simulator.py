import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
from scipy.stats import norm
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import random
import imageio

def trajectories(number, map = None, meanLength = 5, ROI = None, trapList = None, bindingSiteList = None):
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
    bindingSiteList (Polygon)
        list of zones considered binding sites   

    Returns
    -------
    DataFrame

    Notes
    -----
    We could create an environment class, otherwise we'll keep taking on extra arguments
    """
    #TODO:it would be nice to be able to specify a set length if so wanted.
    # generate an array of numbers taken from the exponantial distribution
    lengthList = np.random.exponential(scale=meanLength, size=number)
    lengthList = np.ceil(lengthList).astype(int) +1

    #if no ROI is passed, all starting positions will be 0
    startingPosList = generate_random(number, ROI)
    trackPopulation = []

    #TODO: if no ROI nor traps are passed, use the simpler diffusing function

    for i, (length,startingPos) in enumerate(zip(lengthList, startingPosList)):
        tracklet = singleTraj(length, startingPos, map, ROI=ROI, trapList=trapList, bindingSiteList = bindingSiteList)
        tracklet.loc[:,'particle'] = i
        trackPopulation.append(tracklet)
    
    traj = pd.concat(trackPopulation, ignore_index=True)

    #TODO: add a temporal dimension to the simulated data
    #with random choosing of the starting frame and cutting
    #the data past last frame.

    return traj

def freeDiffusion(length, startingPos, map = None, trapList = None, bindingSiteList = None, xmin=-20, xmax=20):
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

        x, y = displace()
        X.append(X[-1] + x)
        Y.append(Y[-1] + y)

        trap = isTrapped((X[-1], Y[-1]), trapList)
        #if isTrapped has returned something, it means the particle is indeed trapped
        if (trap != None):
            XinTrap, YinTrap = diffuseInTrap([X[-1], Y[-1]], 0.9, trap, length - 1 - i, xmin, xmax, map)
            X.extend(XinTrap)
            Y.extend(YinTrap)
            i += len(XinTrap)
        
        site = isBound((X[-1], Y[-1]), bindingSiteList)
        if (site is not None):
            Xbind, Ybind = binding([X[-1], Y[-1]], 0.01, site, length - 1 - i)
            X.extend(Xbind)
            Y.extend(Ybind)
            i += len(Xbind)

    frame = np.arange(1,len(X)+1)
    particle = np.zeros(len(X)).astype(int)
    traj = pd.DataFrame({'frame': frame,
                        'particle': particle,
                        'x':X, 
                        'y':Y
                        })

    return traj


def singleTraj(length, startingPos, map=None, ROI = None, trapList = None, bindingSiteList = None):
    """
    Helper function doing the simulation for a single particle in a given time
    """
    X = []
    Y = []

    #get the extend of the frame
    if (ROI is not None):
        xmin, ymin, xmax, ymax = _getFrame(ROI) #TODO:may want to return a vector [frame] with all that, and adapt getDiffusivityMap
    else:
        return freeDiffusion(length, startingPos, trapList=trapList, bindingSiteList=bindingSiteList)
    
    #put in the starting position
    X.append(startingPos[0])
    Y.append(startingPos[1])
    #for the desired length of the trajectory
    i = 1
    while (i<length):
        i += 1
        #draw the displacement 
        D = getDiffusivity((X[-1], Y[-1]), map, xmax, xmin)
        x, y = displace(diffusivity=D)

        #check if we are inside the ROI
        if (ROI.contains(Point(X[-1] + x, Y[-1] + y))):
            X.append(X[-1] + x)
            Y.append(Y[-1] + y)
        else:
            #if we are not, we add the inverse of the displacement
            X.append(X[-1] - x)
            Y.append(Y[-1] - y)

        trap = isTrapped((X[-1], Y[-1]), trapList)
        #if isTrapped has returned something, it means the particle is indeed trapped
        if (trap != None):
            XinTrap, YinTrap = diffuseInTrap([X[-1], Y[-1]], 0.9, trap, length - 1 - i, xmin, xmax, map)
            X.extend(XinTrap)
            Y.extend(YinTrap)
            i += len(XinTrap)
        
        site = isBound((X[-1], Y[-1]), bindingSiteList)
        if (site is not None):
            Xbind, Ybind = binding([X[-1], Y[-1]], 0.01, site, length - 1 - i)
            X.extend(Xbind)
            Y.extend(Ybind)
            i += len(Xbind)

    frame = np.arange(1,len(X)+1)
    particle = np.zeros(len(X))
    traj = pd.DataFrame({'frame': frame,
                        'particle': particle,
                        'x':X, 
                        'y':Y
                        })

    return traj

def diffuseInTrap(pos, Ptrap, trapZone, maxFrames, xmin, xmax, map):
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
    """
    X = []
    Y = []

    #put in the first coordinates
    X.append(pos[0])
    Y.append(pos[1])

    

    diffuse = True
    currentFrame = 1 
    while(diffuse):
        #draw
        D = getDiffusivity((X[-1], Y[-1]), map, xmax, xmin)
        x, y = displace(diffusivity=D)

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

def binding(pos, pLib, site, maxFrames):
    """Simulates the binding of a molecule to a binding site
    with a flat probability of escape each time step"""
    X = []
    Y = []
    xsite = site.centroid.x
    ysite = site.centroid.y
    #put in the first coordinates
    X.append(xsite)
    Y.append(ysite)

    currentFrame = 1
    while (currentFrame <= maxFrames):
        if (np.random.rand() < pLib):
            Xe, Ye = escape(site, 0.5)
            X.append(Xe)
            Y.append(Ye)  
            return X, Y
        else:
            X.append(xsite)
            Y.append(ysite)
        currentFrame += 1
    return X, Y

#TODO:verify that this gives a uniform distrib of alpha
def escape(site, escapeRadius):
    """Gives a new location for the escaping molecule,
    at a given radius from the site, with uniform radial
    distribution"""
    alpha = 2 * math.pi * np.random.rand()
    x = site.centroid.x + math.cos(alpha)
    y = site.centroid.y + math.sin(alpha)
    return x, y

#TODO:isTrapped and isBound are the same function, just merge them
#isInside or something
def isTrapped(position, listTraps):
    """
    Checks in which polygon the particle is.
    Returns
    -------
    Polygon | None
    """
    point = Point(position) #careful with format of position (x, y)
    for trap in (listTraps or []):
        #if any of the listed trap contains the point
        if (trap.contains(point)):
            return trap
    return None

def isBound(position, bindingSiteList):
    """Returns the binding site to which the particle is bound
    Returns None if not bound"""
    point = Point(position)
    for site in (bindingSiteList or []):
        #if any of the listed trap contains the point
        if (site.contains(point)):
            return site
    return None

def concatTracklets(trackPopulation):
    """
    Concatenate several tracks into a single dataframe
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
def displace(diffusivity=30, fps=1000, mpp=1):
    #TODO:sort out the units in all of this
    time_interval = 1 / fps
    dx = np.random.normal(0, np.sqrt(2 * diffusivity * time_interval)) * mpp
    dy = np.random.normal(0, np.sqrt(2 * diffusivity * time_interval)) * mpp
    return dx, dy

def _getFrame(ROI):
    x, y = ROI.exterior.coords.xy
    return np.min(x), np.min(y), np.max(x), np.max(y)

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

def generate_random(number, polygon):
    """generates a number of point within geometry"""
    list_of_points = []
    #TODO:test
    if (polygon is None):
        list_of_points = [(0,0) for i in range(number)]
    
    minx, miny, maxx, maxy = polygon.bounds
    counter = 0
    while counter < number:
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pnt):
            list_of_points.append([pnt.x, pnt.y])
            counter += 1
    return list_of_points

def drawGeometry(ROI=None, trapList=None, bindingSiteList = None):
    """Draws the location of the elements boundaries to help
    vizualize the diffusion process"""
    axes = plt.gca()
    if (ROI is not None):
        ROIx, ROIy = ROI.exterior.xy
        xlimitMin, xlimitMax = np.min(ROIx), np.max(ROIx)
        ylimitMin, ylimitMax = np.min(ROIy), np.max(ROIy)
        plt.plot(ROIx, ROIy, color='blue')
        axes.set_ylim(xlimitMin,xlimitMax)
        axes.set_xlim(ylimitMin,ylimitMax)

    if (bindingSiteList is not None):
        for site in bindingSiteList:
            x, y = site.exterior.xy
            plt.plot(x,y, color='red')

    if (trapList is not None):
        for site in trapList:
            x, y = site.exterior.xy
            plt.plot(x,y, color='green')


#block of functions for getting a value for diffusivity constant from a map file
def getDiffusivityMap(fileName):
    Dmap = imageio.imread(fileName, pilmode='RGB')
    return Dmap[:,:,2]

def _inverseVerticalIndex(index, resolution):
    new = resolution - 1 - index
    new = new.astype(int)
    return new

def translatePosToArray(position, x_max, x_min, res):
    increment = abs((x_max - x_min) / res)
    i_index = np.floor(position[1]/increment).astype(int)
    j_index = np.floor(position[0]/increment).astype(int)
    i_index = _inverseVerticalIndex(i_index, res)
    return i_index, j_index

def getDiffusivity(position, map, x_max, x_min):
    #TODO:quick fix. How to handle no map ???
    #TODO:get resolution of the map rather than hardcode it.
    if (map is None):
        return 30
    i_index, j_index = translatePosToArray(position, x_max, x_min, 512)
    try:
        D = map[i_index, j_index]
    except IndexError:
        D = 30
    return D