import numpy as np
import pandas as pd
from diffusion import simulator as sim
from diffusion import plots, analyse, utils
import matplotlib.pyplot as plt

cell = sim.createCircle(10, (10,10), 100)
traj = sim.Trajectories(100, meanLength = 5, ROI=cell)

sim.drawGeometry(cell)
plots.plotTraj(traj)

plots.displacementHistogram(traj, bins = 40)

