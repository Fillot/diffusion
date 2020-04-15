import numpy as np
import matplotlib.pyplot as plt
from diffusion import simulator as sim
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import math


trapList = []
trapList.append(sim.createTrap(1, (2,0)))
trapList.append(sim.createTrap(1, (-2,0)))

#we expect a list of two polygons, each with 50 vertices
print(trapList[0].centroid)
#this seems to work