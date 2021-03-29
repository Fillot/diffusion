import numpy as np

A = np.array([0,0,0])
B = np.array([10,10,0])
C = np.array([6,10,0])

x=B-A
y=C-A

proj = np.dot(x, y) / np.linalg.norm(y)
print(proj)