import numpy as np
from diffusion import plots, geometry, solver, simulator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from itertools import product, combinations

vert, faces = geometry.parse_obj("./meshes/sample_mesh.obj")
sphere = geometry.Mesh(vert, faces)
print(isinstance(sphere.faces[0], geometry.Face))

locList = np.genfromtxt("./meshes/sample_chromatin.xyz", delimiter=' ')
chromatin = geometry.Chromatin(locList, 133, 0.02)

sim = simulator.Simulator(sphere, chromatin)
sim.Simulate(1,100)

#04-04-2021
# test number of collisions checks avoided using this 
# --> MEAN AVOIDED 77 over 80
# --> 21 AABB comparisons on average
# real life test
# --> integrated into solver
# adapt for chromatin
# --> done
# integrate into the update function
# --> done, to test


# fig = plt.figure()
# ax = fig.gca(projection='3d')
# for node in tree.nodeList[62:67]:
#     print(node.index)
#     node.printAABB()
#     si = node.pointerToFirstObject
#     nb = node.numObjects
#     faces = tree.objectsArray[si:si+nb]
#     for face in faces:
#         v1, v2, v3 = face.getFaceVertices()
#         x = [v1.x, v2.x, v3.x]
#         y = [v1.y, v2.y, v3.y]
#         z = [v1.z, v2.z, v3.z]
#         verts = [list(zip(x, y, z))]
#         ax.add_collection3d(Poly3DCollection(
#             verts, 
#             edgecolor='black',
#             linewidths=2, 
#             alpha=0))
# #PQ
# x = [p[0], q[0]]
# y = [p[1], q[1]]
# z = [p[2], q[2]]
# verts = [list(zip(x, y, z))]
# ax.add_collection3d(Poly3DCollection(
#     verts, 
#     edgecolor='red',
#     linewidths=2, 
#     alpha=0))