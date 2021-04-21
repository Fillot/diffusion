import numpy as np
from diffusion import plots, geometry, solver, simulator
import ctypes
import time
import cProfile


vert, faces = geometry.parse_obj("./meshes/sample_mesh.obj")
sphere = geometry.Mesh(vert, faces)

locList = np.genfromtxt("./meshes/sample_chromatin.xyz", delimiter=' ')
chromatin = geometry.Chromatin(locList, 133, 0.02)

sim = simulator.Simulator(sphere, chromatin)
cProfile.run('sim.Simulate(1,100)', 'profiler_output')

import pstats
import io
s = io.StringIO()
ps = pstats.Stats('profiler_output', stream=s).sort_stats('cumulative')
ps.print_stats()
with open('test.txt', 'w+') as f:
    f.write(s.getvalue())