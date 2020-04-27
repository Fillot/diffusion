import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
from diffusion import simulator as sim
from diffusion import plots
from diffusion import analyse


ROI = sim.createCircle(10, (0,0), 50)
trapList=[]
trapList.append(sim.createCircle(2, (20,2), 10))
traj = sim.Trajectories(5, 5, ROI=ROI, trapList=trapList)


fig, ax = plt.subplots()
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
tp.plot_traj(traj)


# pos_columns = ['x', 'y']
# unstacked = traj.set_index(['particle', 'frame'])[pos_columns].unstack()
# fig, ax = plt.subplots()
# for i, trajectory in unstacked.iterrows():
#     ax.plot(trajectory['x'], trajectory['y'])
# ax.set_xlim(-10,10)
# ax.set_ylim(10,-10)
# plt.show()

#same results !
#i still need to understand exactly what unstack does tho
# setIndex = traj.set_index(['particle', 'frame'])[pos_columns]
# setIndex.head()
# unstacked.head()
#df.set_index(['particle', 'frame'])
#returns
#part   frame   x   y
#0  0   x0  y0
#   1   x1  y1
#   .........
#n  0   x0  y0
#
#unstacking it returns
#frame      0   1   2   3   ...
#particle1  x0  x1 ....
#part2      x0  x1 ...
#part3      NaN NaN x2
# 
# where particle 3 appears only at frame 2

# plots.plotTraj(traj)


msd = analyse.emsd(traj)
ax = msd.plot(x="tau", y="msds", logx=False, logy=False, legend=False)
plt.show()

plots.trajLength(traj)