#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:13:00 2020

@author: tom
"""

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

def importTracks(fileName):
    """
    Reads track data from the TrackMate plugin and gets them
    in a convenient format      
    Parameters
    ----------
    fileName (string) : path to the file  TODO: make it more robust to different types of OS

    Returns
    -------
    traj:
        list of array, with n lines, and d columns (d = dimension) TODO: is it a pandas df ? np ?
              x1  x2    t   id
            |____|___|____|____| 
            |____|___|____|____|
                ..............
            |____|___|____|____|

    Notes
    -----
    TODO: pandas df and numpy ?
    TODO: time ? frame number ? What does the line mean ?
    TODO: third dimension ?
    """
    tree = ET.parse(fileName)
    root = tree.getroot()

    print(root.get('nTracks'))

    x = []
    y = []
    t = []
    Id = []


    i = 1
    for particle in root.iter('particle'):
        for detection in particle.iter('detection'):
            x.append(float(detection.get('x')))
            y.append(float(detection.get('y')))
            t.append(float(detection.get('t')))
            Id.append(i)
        i += 1

    return np.column_stack((x,y,t,Id))


def _unit_vector(vector):
    """ Returns the unit vector of the vector. 
    
    >>>unit_vector((2,0,0))
    (1,0,0)
    >>>unit_vector([1,1,1])
    [0.57735027 0.57735027 0.57735027]
    
    """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = _unit_vector(v1)
    v2_u = _unit_vector(v2)
    return math.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def angles(vectors):
    #TODO:super moche. But relies on the fuck up in get vectors
    #TODO: angles between 0 vector and anything
    """Computes the angles between each successive displacement vectors"""
    angles = []
    for i in range(len(vectors)-1):
        angles.append(angle_between(vectors[i], vectors[i+1]))
    return angles

def displacements(traj, pos_columns = ['x', 'y']):
    """Computes the displacement between each consecutive frames for multiple particles
    
    Parameters
    ----------
    traj (DataFrame)
        the dataframe containing consecutive position of the particle
    pos_columns (string vector)
        names of the columns containing the position at each frames
    Returns
    -------
    ndarray
    """
    disp = []
    unstacked = traj.groupby(['particle'])

    for name, trajectory in unstacked:
        pos = trajectory[pos_columns].values
        jumps = displacement_tracklet(pos)
        disp.append(jumps)

    return disp

def displacement_tracklet(pos):
    jumps = []
    for t in range(len(pos)-1):
        v = pos[t+1,:] - pos[t,:]   
        jump = np.linalg.norm(v)
        jumps.append(jump)
    return jumps



def get_vectors(traj, pos_columns=['x', 'y']):
    #TODO: this outputs a list of vector which is lame. I think iterrows or the equivalent in numpy is what
    #we are looking for
    vectors = []
    pos = traj[pos_columns].values
    for t in range(len(pos)-1):
        v = pos[t+1,:] - pos[t,:]   
        vectors.append(v)
    return vectors
