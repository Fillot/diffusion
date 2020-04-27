import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diffusion import simulator as sim

def msd(trajectory, coords=['x', 'y']):

    pos = trajectory[coords].values
    lagtimes = np.arange(1, len(pos))
    msds = np.zeros(lagtimes.size)
    msds_std = np.zeros(lagtimes.size)

    for i, lt in enumerate(lagtimes):
        # diffs = traj[coords] - traj[coords].shift(-shift)
        # diffs = diffs.dropna()
        diffs = pos[lt:] - pos[:-lt]
        sqdist = np.square(diffs).sum(axis=1)
        msds[i] = sqdist.mean()
        msds_std[i] = sqdist.std()
    
    msds = pd.DataFrame({'msds': msds, 'tau': lagtimes, 'msds_std': msds_std})
    return msds

def autocorrFFT(x):
  N=len(x)
  F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding
  PSD = F * F.conjugate()
  res = np.fft.ifft(PSD)
  res= (res[:N]).real   #now we have the autocorrelation in convention B
  n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
  return res/n #this is the autocorrelation in convention A

def msd_fft(traj, coords=['x', 'y']):

    r=traj[coords].values
    N=len(r)
    D=np.square(r).sum(axis=1) 
    D=np.append(D,0) 
    S2=sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q=2*D.sum()
    S1=np.zeros(N)
    for m in range(N):
        Q=Q-D[m-1]-D[N-m]
        S1[m]=Q/(N-m)
    msds = S1-2*S2
    msds = np.delete(msds, 0, 0)
    results = pd.DataFrame({'msds':msds, 'tau': np.arange(1, N)})
    return results

N = 5
radius = 10
ROI = sim.createCircle(radius, (0,0), 20)
trapList = []
trapList.append(sim.createTrap(2, (-2,0)))

# Pos = sim.simulateOneTraj(N, [0,0], ROI=ROI, trapList=trapList)


max_time = 5
dt = max_time / N
t = np.linspace(0, max_time, N)

# traj = pd.DataFrame({'t': t, 'x': Pos[:,0], 'y': Pos[:,1]})

# #draw the trajectory
# ax = traj.plot(x='x', y='y', alpha=0.6, legend=False)
# ax.set_xlim(-radius, radius)
# ax.set_ylim(-radius, radius)

# msd = msd(traj)
# ax = msd.plot(x="tau", y="msds", logx=True, logy=True, legend=False)
# ax.fill_between(msd['tau'], msd['msds'] - msd['msds_std'], msd['msds'] + msd['msds_std'], alpha=0.2)



#20200317
#dev ensemble MSD
scale = 5
# Pos = sim.simulateTrajectories(1000, scale, ROI=ROI, trapList=trapList)
# traj = pd.DataFrame({'id': Pos[:,2], 'x': Pos[:,0], 'y': Pos[:,1]})


def emsd(traj, coords=['x', 'y']):
    ids = []
    msds = []
    for pid, ptraj in traj.reset_index(drop=True).groupby('id'):
        msds.append(msd_fft(ptraj, coords= coords))
        ids.append(int(pid))

    msds = pd.concat(msds, keys = ids, names=['particle', 'frame'])
    results = msds.mean(level=1)
    return results
