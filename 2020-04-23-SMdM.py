#try to use Davide's code and use the SMdM framework
#on particles diffusing in a pattern of diffusivity.

import numpy as np
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
from diffusion import simulator
from diffusion import analyse
from diffusion import plots



map = simulator.getDiffusivityMap('./diffusivity.png')
cell = simulator.createCircle(10, (10,10), 100)
traj = simulator.singleTraj(1, map, ROI=cell)

simulator.drawGeometry(ROI=cell)
plots.plotTraj(traj)

t_step = 0.005
threshold = 2
GridSize = 32
GridStep = 0.2168

traj['frame']= np.arange(1,len(traj)+1)
nFrames = traj['frame'].max()

#nested list equivalent of cell(GridSize)
grid = [[[] for _ in range(GridSize)] for _ in range(GridSize)]
count = np.zeros((GridSize,GridSize))

for frame in range(1, nFrames +1, 2):

    if (traj.loc[traj['frame'] == frame + 1].empty == False):
        posThisFrame = traj.loc[traj['frame'] == frame][['x', 'y']]
        posNextFrame = traj.loc[traj['frame'] == frame + 1][['x', 'y']]
        valuesThisFrame = posThisFrame.values
        valuesNextFrame = posNextFrame.values

        for i, this_pos in enumerate(valuesThisFrame):
            idx = np.zeros(len(valuesNextFrame))
            for j, next_pos in enumerate(valuesNextFrame):
                distance = np.linalg.norm(next_pos - this_pos)
                if (distance < threshold):
                    idx[j] = 1
            
            if (idx.sum() == 1):
                indexOfNextFrame = np.where(idx == 1)[0][0]
                next_pos = valuesNextFrame[indexOfNextFrame]
                distance = np.linalg.norm(next_pos - this_pos)
                i_index, j_index = simulator.translatePosToArray(next_pos, 20, 0, GridSize)
                count[i_index, j_index] += 1
                grid[i_index][j_index].append(distance)

def printNestedList(l):
    shape = np.shape(l)
    heatmap = np.zeros((shape[0], shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            mean = np.mean(l[i][j])
            heatmap[i,j] = mean
    plt.imshow(heatmap, cmap='hot')
    plt.show()

def pdf(r, a, b):
    f = 2 * r/a * np.exp((-r ** 2)/a) +b*r
    return f

printNestedList(grid)
abc = pdf(2,2,2)