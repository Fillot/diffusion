import numpy as np
from diffusion import plots, geometry, solver, simulator
np.random.seed(42)

vert, faces = geometry.parse_obj("./meshes/sample_mesh.obj")
sphere = geometry.Mesh(vert, faces)

bsList = [20000]
locList = np.genfromtxt("./meshes/sample_chromatin.xyz", delimiter=' ')
locList *= 0.10
chromatin = geometry.Chromatin(locList, 133, 0.02, bindingSiteList=bsList)

sim = simulator.Simulator(sphere, chromatin)

sim.SetSpeBindingStrength(1)

nonSpeBinding = [0.75,0.80,0.85,0.90,0.95]
for ns in nonSpeBinding:
    sim.SetNonSpeBindingStrength(ns)
    sim.Simulate(1000,3000)
    traj = sim.GetTraj()
    print("Ended")
    traj.to_csv(f"traj_{ns}_non_spe_binding.csv")

import pandas as pd
import numpy as np
search_time = np.zeros((1000,5))
search_time[:,:]=3000

for p_index, ns in enumerate(nonSpeBinding):
    traj = pd.read_csv(f"traj_{ns}_non_spe_binding.csv")
    grouped = traj.groupby("particle")
    for i, df in grouped:
        sliced = df[df.Bind == 1]
        if sliced.shape[0] == 0:
            continue
        index = sliced.index[0]
        part=sliced["particle"][index]
        search_time[part,p_index] = sliced["frame"][index]
sm = np.argsort(search_time, axis=0)
column_sorted = np.take_along_axis(search_time, sm, axis=0)
df=pd.DataFrame(column_sorted)
df.to_csv(f"search_time_only.csv")

#--------------------------------------------#
import pandas as pd
import numpy as np
time_spent_sliding = np.zeros((1000,5))
nonSpeBinding = [0.75,0.80,0.85,0.90,0.95]

for p_index, ns in enumerate(nonSpeBinding):
    traj = pd.read_csv(f"traj_{ns}_non_spe_binding.csv")
    grouped = traj.groupby("particle")
    for i, df in grouped:
        sum_sliding = df["Sliding"].sum()
        time_spent_sliding[i, p_index] = sum_sliding

time_spent_sliding = time_spent_sliding/3000
mean_sliding = np.mean(time_spent_sliding, axis=0)
# but you should stop counting when binding specifially

#---------occupency of each monomer----------#
import pandas as pd
import numpy as np

nonSpeBinding = [0.75,0.80,0.85,0.90,0.95]
occupency_array = np.zeros((736,5))#len(monomers)
for p_index, ns in enumerate(nonSpeBinding):
    traj = pd.read_csv(f"traj_{ns}_non_spe_binding.csv")
    BP = traj['Pos BP']
    for p in BP:
        if pd.isnull(p):
            continue
        mono = int(p//133)
        occupency_array[mono, p_index] += 1

mean_occupency = np.mean(occupency_array, axis=1)
import matplotlib.pyplot as plt
plt.plot(occupency_array[:,2])
plt.ylim(800, 2000)
plt.show()