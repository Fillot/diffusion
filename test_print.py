import numpy as np
from diffusion import plots, geometry, solver, simulator

vert, faces = geometry.parse_obj("./meshes/sample_mesh.obj")
sphere = geometry.Mesh(vert, faces)

faces = sphere.faces
tree = geometry.BVH(faces)
tree.print_tree()

# self.leaf = False
# self.AABB = []
# self.numObjects = 1
# self.pointerToFirstObject = 0
# self.left = None
# self.right = None


