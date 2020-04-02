import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diffusion import simulator as sim
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import math

#definition zone

#this is a very simple first approach to creating a 
# region of interest. It might need tweeking to be compatible
# with imageJ ROI 
def createPolygon(points):
    """creates a polygon based on an ordered list of points
    
    Parameters
    ----------
    points: ordered list of points
        vertices are joined row by row
    
    Returns
    -------
    Polygon object from the shapely library
    """
    return Polygon(points)

#we should have a bunch of ready-made geometry makers
#to facilitate setting up a complex environment in which
#the particles will diffuse
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
    #(!): this part is sensible to input format -> TODO:make it robust
    circle = circle + np.array(center)
    return Polygon(circle)

def createTrap(radius, center):
    return createCircle(radius, center, 50)

def simulateTraj(number, scale, ROI):
    """
    Simulates N trajectories of average duration scale inside
    a given region of interest

    Parameters
    ----------
    N (int)
        the number of wanted individual particles
    scale (int)
        average duration (in frames) of trajectories, from exponantial distribution
    ROI (Polygon)
        the region within which particles will diffuse

    Returns
    -------
    Pos (ndarray)
        numpy array of form 
             x1  x2  ... ...id
            |____|___|____|__1_|
            |____|___|____|__1_|
            |____|___|____|__2_|
            ....................
            |____|___|____|__N_|
    """
    #this is copy pasted from sim.simulateTrajectories
    #except we don't need third dimension
    # generate an array of numbers taken from the exponantial distribution
    # and round up to have the desired length of each simulated tracks
    array = np.random.exponential(scale=5, size=100)
    array = np.ceil(array).astype(int)

    trackPopulation = []

    for length in array:
        pos = Traj(length, ROI)
        trackPopulation.append(pos)

    Pos = sim.concatTracklets(trackPopulation)
    return Pos

#simulate the trajectory of a single particle inside given ROI
#TODO: find a solution that doesn't involve instancing a shapely Point
#for each check
def Traj(length, ROI):
    """
    Creates one trajectory of length 'length' for a particle diffusing within ROI
    in 2d
    """
    X = []
    Y = []
    #put the first steps otherwise X[-1] will Index out of Range
    X.append(np.random.normal(0, 1))
    Y.append(np.random.normal(0, 1))
    #check if we are in the max radius.
    for _ in range(length):
        x = np.random.normal(0, 0.5)
        y = np.random.normal(0, 0.5)

        #if we are inside the ROI
        if (ROI.contains(Point(X[-1] + x, Y[-1] + y))):
            X.append(X[-1] + x)
            Y.append(Y[-1] + y)
        else:
            X.append(X[-1] - x)
            Y.append(Y[-1] - y)
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
    x = np.random.normal(0, 0.05)
    y = np.random.normal(0, 0.05)
    #put in the first coordinates
    X.append(pos[0])
    Y.append(pos[1])
    diffuse = True
    while(diffuse):
        if (trapZone.contains(Point(X[-1] + x, Y[-1] + y))):
                X.append(X[-1] + x)
                Y.append(Y[-1] + y)
        else:
            #draw from uniform distribution, if above Ptrap threshold : escape
            if (np.random.rand(1) > Ptrap):
                X.append(X[-1] + x)
                Y.append(Y[-1] + y)
                return np.column_stack((X,Y))
            else:
                X.append(X[-1] - x)
                Y.append(Y[-1] - y)
        diffuse = [currentFrame =< maxFrames]
    Pos = np.column_stack((X,Y))
    return Pos

#to make this line work, i had to create an appropriate __init__.py file
#in the diffusion folder
Pos = sim.simulateTrajectories(100, 2, 5, 10)

#now we are trying to define a polygon
#here a square with only 4 points
X = ([(0, 0), (0, 10), (10, 10), (10, 0)])
ROI = createPolygon(X)

#checking if a point is inside ROI
pointA = Point(0.5,0.5)
pointB = Point(1.5,1.5)
pointC = Point(1, 0)

print(ROI.contains(pointA))
print(ROI.contains(pointB))
print(ROI.contains(pointC))
#pointC is on the edge and returns false

#we want a function to do the diffusion simulation inside the ROI
traj = simulateTraj(100, 5, ROI)

#we want a handy function to create a circle ROI
ROI2 = createCircle(1, (1,0), 20)

# This part keeps raising a 
# Value 'npROI2.shape' is unsubscriptable
# error message from pylint while the script still runs fine

# npROI2 = np.array(ROI2)
#to check if these points are truly on the circle, let's calculate the norm
#of (point - center)
# norm = []
# print(npROI2.shape[0])
# for point in range(npROI2.shape[0]):
#     norm.append(np.linalg.norm(npROI2[point, :] - center))

#we implement the trapping behavior
radius = 1
center = (5,5)
trap = createTrap(radius, center)

check = np.random.rand(1)

#first add comment
