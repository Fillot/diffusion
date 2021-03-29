import numpy as np
import geometry, solver

vert, faces = geometry.parse_obj("./3Dmesh/small_icosphere.obj")
sphere = geometry.Mesh(vert, faces)


locusList = np.genfromtxt(
    "/home/tom/Documents/Scientifique/Thesis/Code/simulatorvChromatin/3Dmesh/converted_xyz.xyz", delimiter=' ')
chromatin = geometry.Chromatin(locusList, 333, 0.02)

sim = solver.Simulator(sphere, chromatin)
sim.Simulate(1,100)
traj = sim.GetTraj()
print("Ended")

