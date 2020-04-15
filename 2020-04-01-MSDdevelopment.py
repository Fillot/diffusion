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
    circle = circle + np.array(center)
    return Polygon(circle)

def createTrap(radius, center):
    return createCircle(radius, center, 50)

def simulateTraj(number, scale, ROI, trapList = None):
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
        startingPos = [np.random.normal(0, 1), np.random.normal(0, 1)]
        trackPopulation.append(Traj(length, startingPos, ROI, trapList))

    Pos = sim.concatTracklets(trackPopulation)
    return Pos

#simulate the trajectory of a single particle inside given ROI
def Traj(length, startingPos, ROI, listTraps):
    """
    Creates one trajectory of length 'length' for a particle diffusing within ROI
    in 2d
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
        #check if we are inside the ROI
        if (ROI.contains(Point(X[-1] + x, Y[-1] + y))):
            X.append(X[-1] + x)
            Y.append(Y[-1] + y)
        else:
            #if we are not, we add the inverse of the displacement
            X.append(X[-1] - x)
            Y.append(Y[-1] - y)

        trap = isTrapped((X[-1], Y[-1]), listTraps)
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

# print(ROI.contains(pointA))
# print(ROI.contains(pointB))
# print(ROI.contains(pointC))
#pointC is on the edge and returns false

#we want a function to do the diffusion simulation inside the ROI
# traj = simulateTraj(100, 5, ROI)

#we want a handy function to create a circle ROI
ROI2 = createCircle(10, (0,0), 20)

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

#reprise le 2020-04-06
#createTrap creates a "trap" in name at least
#we were coding the behavior of a molecule inside the trap.

#i must make sure that Traj(2, ROI) then Traj(3, ROI) == Traj(5, ROI)
#i.e. Traj is abortable and restartable no problem
#first problem : the initial frame. Solution : the user must pass in the first
#location of the particule.
#if I pass Traj(3, [0,0], ROI), I want a final result of three consecutive positions
#where  the first is [0,0], in other word, Traj must draw two displacement, so n-1

#test (avec dx not random but 1 each time)
# Pos1 = Traj(5, [0,0], ROI2)
# Pos2 = Traj(3, [0,0], ROI2)
# Pos3 = Traj(2, [Pos2[-1,0], Pos2[-1,1]], ROI2)
#note: Pos3 fait 2 et 3, since I ask 2 frames, including the starting one
#to chain them, I have to call Traj(remainingFrames + 1) and then clip the starting one
# Pos1 = Traj(5, [0,0], ROI2)
# Pos2 = Traj(3, [0,0], ROI2)
# Pos3 = Traj(2+1, [Pos2[-1,0], Pos2[-1,1]], ROI2)
# Pos3 = np.delete(Pos3, 0, 0)
# Pos1bis = np.concatenate((Pos2, Pos3))
# check = (Pos1 == Pos1bis)
# check = np.mean(check) #check == 1 only if the comparison is true for all coordinates
#this poses the problem that we can't just concatenate, we have to take care 
# of the first frame to avoid repetitions 

trapList = []


#un premier challenge et que l'on doit sauter X boucles quand diffuseInTrap retourne son array

# Pos1 = Traj(5, [0,0], ROI2, trapList)
# Pos2 = Traj(3, [0,0], ROI2, trapList)
# Pos3 = Traj(2+1, [Pos2[-1,0], Pos2[-1,1]], ROI2, trapList)
# Pos3 = np.delete(Pos3, 0, 0)
# Pos1bis = np.concatenate((Pos2, Pos3))
# check = (Pos1 == Pos1bis)
# check = np.mean(check)

#success

trapList.append(createTrap(1, (2,0)))
#test with a flag in the isTrapped function
Pos1 = Traj(10, [0,0], ROI2, trapList)
#success
#avec Ptrap = 1 in the call to diffuseInTrap, on a bien ce qu'on veut.

#test of how flimsy createCircle is.
circ = createCircle(1, (1,1), 4)
circ1 = createCircle(1, [1,1], 4)
#en fait ça va

