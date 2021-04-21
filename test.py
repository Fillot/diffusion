import numpy as np
from diffusion import plots, geometry, solver, simulator
import ctypes
import time
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

verts = np.array([
                [1,-0.1,0],
                [1,-0.1,1],
                [2,-1.1,0],
                [2,0.1,0],
                [2,0.1,1],
                [3,-0.9,0],
                [2,-1.1,0],
                [2,-1.1,1],
                [3,-2.1,0],
                [2.5,0.1,0],
                [2.5,0.1,1],
                [3.5,-0.9,0],
                ])

faces = np.array([
                [0,2,1],
                [3,5,4],
                [6,7,8],
                [9,11,10]
                ])


class testFace():
    def __init__(self, vertsList, myVertices):
        self.vertsList = vertsList
        self.v1 = vertsList[myVertices[0]]
        self.v2 = vertsList[myVertices[1]]
        self.v3 = vertsList[myVertices[2]]
        self.normal = np.array([0,0,0], dtype=float)
        self.calc_face_normal()

    def calc_face_normal(self):
        if self.normal.all() != np.array([0,0,0]).all():
            self.normal = np.array([0,0,0], dtype=float)
        verts = [self.v1, self.v2, self.v3, self.v1]
        for i in range(3):
            self.normal[0] += (verts[i][1] - verts[i+1][1]) * (verts[i][2] + verts[i+1][2])
            self.normal[1] += (verts[i][2] - verts[i+1][2]) * (verts[i][0] + verts[i+1][0])
            self.normal[2] += (verts[i][0] - verts[i+1][0]) * (verts[i][1] + verts[i+1][1])
        self.normal = self.normal / np.linalg.norm(self.normal)

    def get_face_vertices(self):
        return self.v1, self.v2, self.v3

faceList = []
for f in faces:
    faceList.append(testFace(verts, f))

# bestT = 1
# for i, f in enumerate(faceList):
#     t = sim.solver.TimeOfCrossing(position, new_position, f)
#     if (t<bestT):
#         print(f"potential coll with {i}")
#         if sim.solver.InsideTriangle(position, new_position, f):
#             print('collision')
#             new_position = sim.solver.IntersectionPoint(position, new_position, f)
#     else:
#         print("no")



p = np.array([0,0,0])
q = np.array([4,0,0])

lib = ctypes.cdll.LoadLibrary('./diffusion/function.so')
lib.TriangleInteresect.restype = ctypes.c_bool


"""NarrowPhase with C"""
# tracking variables to keep track of the current
# closest colliding face and its time of crossing
u = ctypes.c_double()
v = ctypes.c_double()
w = ctypes.c_double()
t = ctypes.c_double()
bestT = 1
collidingFaceIndex  = None
for i, f in enumerate(faceList):
    #TODO:GetVerticesAsArray
    A, B, C = f.get_face_vertices()
    # t = self.TimeOfCrossing(position, new_position, f)
    if (lib.TriangleInteresect(
        ctypes.c_void_p(p.ctypes.data),
        ctypes.c_void_p(q.ctypes.data),
        ctypes.c_void_p(A.ctypes.data),
        ctypes.c_void_p(B.ctypes.data),
        ctypes.c_void_p(C.ctypes.data),
        ctypes.byref(u),
        ctypes.byref(v),
        ctypes.byref(w),
        ctypes.byref(t))):
        print("yes")
        if t.value<bestT:
            q = u.value*A+v.value*B+w.value*C
            bestT = t.value
            collidingFaceIndex = i

fig = plt.figure()
ax = Axes3D(fig)
for id, face in enumerate(faceList):
    v1, v2, v3 = face.get_face_vertices()
    x = [v1[0], v2[0], v3[0]]
    y = [v1[1], v2[1], v3[1]]
    z = [v1[2], v2[2], v3[2]]
    verts = [list(zip(x, y, z))]
    ax.add_collection3d(Poly3DCollection(
        verts, 
        edgecolor='black',
        linewidths=2, 
        alpha=0))
    normal = face.normal
    center = (v1 + v2 + v3)/3
    ax.plot([center[0], center[0]+normal[0]], 
        [center[1], center[1]+normal[1]],
        [center[2], center[2]+normal[2]],
        color='yellow')
ax.plot([p[0], q[0]],[p[1], q[1]],[p[2], q[2]], color='blue')
plt.show()