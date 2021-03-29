import numpy as np
from diffusion import plots, geometry, solver, simulator

vert, faces = geometry.parse_obj("./meshes/sample_mesh.obj")
sphere = geometry.Mesh(vert, faces)

locList = np.genfromtxt("./meshes/sample_chromatin.xyz", delimiter=' ')
chromatin = geometry.Chromatin(locList, 133, 0.02)

sim = simulator.Simulator(sphere, chromatin)
sim.Simulate(10,10)
traj = sim.GetTraj()
print("Ended")