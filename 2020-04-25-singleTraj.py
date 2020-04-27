import numpy as np
import pandas as pd
from diffusion import simulator as sim
from diffusion import plots, analyse
import matplotlib.pyplot as plt

traj = sim.singleTraj(1000, (0,0))

plots.plotSingleTraj(traj)