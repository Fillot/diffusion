#MSDTestCases

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diffusion import simulator as sim
from diffusion import analyse
import csv


#set up geometry and variables
N = 5
radius = 10
ROI = sim.createCircle(radius, (0,0), 20)
trapList = []
trapList.append(sim.createTrap(2, (-2,0)))


#check that what is on 2020-04-15.py works on import of analyse
# Pos = sim.simulateTrajectories(N, 5, ROI=ROI, trapList=trapList)
# traj = pd.DataFrame({'id': Pos[:,2], 'x': Pos[:,0], 'y': Pos[:,1]})
# msds = analyse.emsd(traj)

#toy exemple with csv file
#4 particles, moving in each of the cardinal directions
#3 move 1 unit each time
#1 moves 2 units each time
#we expect the output
# tau | msd
#  1  | 1.75
#  2  | 7
#  3  | 15.75
header_list = ['id', 'x', 'y']
traj = pd.read_csv("./tests/ensembleMSD.csv", names=header_list)
msds = analyse.emsd(traj)
ax = msds.plot(x="tau", y="msds", logx=True, logy=True, legend=False)
#success