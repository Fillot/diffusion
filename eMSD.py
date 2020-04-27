#20200317
#dev ensemble MSD

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diffusion import simulator as sim

N = 10
scale = 5

lengthList = np.random.exponential(scale=scale, size=N)
lengthList = np.ceil(lengthList).astype(int)