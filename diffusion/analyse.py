"""
intended as the library grouping all functions to analyse trajectories in a statistical
manner, classifier not included as it is in its own thing.
"""


import numpy as np
import pandas as pd

def emsd(traj, coords=['x', 'y']):
    """compute the ensemble mean square displacement for a group of trajectories
    
    Parameters
    ----------
    traj: (DataFrame)
            pandas DF containing the trajectories of a group of particles,
            with an particle 'id' column
    
    output
    ------
    eMSD: (DataFrame)
        [tau, MSD]

    Notes
    -----
    TODO: units (either pixel/nm or frame/s)
    """
    ids = []
    msds = []
    for pid, ptraj in traj.reset_index(drop=True).groupby('particle'):
        msds.append(msd_fft(ptraj, coords= coords))
        ids.append(int(pid))

    msds = pd.concat(msds, keys = ids, names=['particle', 'frame'])
    results = msds.mean(level=1)
    return results

def msd_fft(traj, coords=['x', 'y']):
    """Computes the msd for one particle using FFT to speed up.
    Based on the SO answer:http://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft#34222273
     """
    r=traj[coords].values
    N=len(r)
    D=np.square(r).sum(axis=1) 
    D=np.append(D,0) 
    S2=sum([_autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q=2*D.sum()
    S1=np.zeros(N)
    for m in range(N):
        Q=Q-D[m-1]-D[N-m]
        S1[m]=Q/(N-m)
    msds = S1-2*S2
    msds = np.delete(msds, 0, 0)
    results = pd.DataFrame({'msds':msds, 'tau': np.arange(1, N)})
    return results

def _autocorrFFT(x):
    "helper function for msd_fft"
    N=len(x)
    F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res= (res[:N]).real   #now we have the autocorrelation in convention B
    n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
    return res/n #this is the autocorrelation in convention A


def msd(trajectory, coords=['x', 'y']):
    """
    Compute the mean square displacement of one particule over a range of time interval
    For a Brownian particle in n-dimension, its position noted as x = (x1, ..., xn)
    <x> la moyenne de x
    MSD(tau) = <|x(t + tau) - x(t)|^2> pour tout t
    with MSD(tau) = 2 x d x D x Tau ; d dimension of freedom ; D diffusivity constant

    Parameters
    ----------
    traj (DataFrame)
        DF 
    tau (int) : the lagTime to be considered TODO:either in frames (int) or seconds (float)
        TODO:check if tau element of |R+ or |N+ depending on the TODO just above.   
    
    """
    pos = trajectory[coords].values
    lagtimes = np.arange(1, len(pos))
    msds = np.zeros(lagtimes.size)
    msds_std = np.zeros(lagtimes.size)
    N = np.zeros(lagtimes.size)
    for i, lt in enumerate(lagtimes):
        # diffs = traj[coords] - traj[coords].shift(-shift)
        # diffs = diffs.dropna()
        diffs = pos[lt:] - pos[:-lt]
        sqdist = np.square(diffs).sum(axis=1)
        msds[i] = sqdist.mean()
        msds_std[i] = sqdist.std()
        N[i]=len(diffs)
    
    msds = pd.DataFrame({'msds': msds, 'tau': lagtimes, 'msds_std': msds_std, 'observations': N})
    return msds


#TODO: this
def anomalousExponent(Pos):
    """
    Computes the anomalous exponent alpha. 
    The mean square dispacement MSD(tau) is proportional to D x tau^alpha
    if alpha > 1, superdiffusion, if alpha < 1, subdiffusion
    alpha > 2 is hyperbalistic, and is typically off-range except for optical systems
    
    """
    alpha = 0
    return alpha


def lengthOfConstraint(Pos):
    
    """The length of constraint Lc is defined as the SD of the locus
    position with respect to its mean averaged over time. This parameter 
    provides estimation for the apparent radius of the volume 
    explored by a finite trajectory.

    Taken from (Amitai, 2017) doi: 10.1016/j.celrep.2017.01.018.
    
    Input : a numpy array containing the position of the particule
            at each frame of form   x1  x2  x3 ... xN
                                t1|____|___|____|____|
                                t2|____|___|____|____|
                                t3|____|___|____|____|
                                ...
    
    Ouput : Lc of type float.
    
    Exemples : (note: shown with lists, but needs numpy arrays)
        >>lengthOfConstraint([[1,	0,	0,],
                              [2,	0,	0,],
                              [3,	0,	0,]])
        0.816496580927726
        
        >>lengthOfConstraint([[1,	0,	0,],
                              [1,	0,	0,],
                              [1,	0,	0,]])
        0.0
        >>lengthOfConstraint([[1,	0,	0,],
                              [-1,	0,	0,],
                              [1,	0,	0,]])
        0.9428090415820634
    """
    
    diffWRTMean = np.square(Pos - Pos.mean(axis=0))  #for each Xi, do (Xi (frame n) - Xi mean)^2 ; avec Pos.mean(axis=0) the mean for each dimension
    sumOfSquare = diffWRTMean.sum(axis=1)           #for each frame, add along all dimension getting the distance squared
    Lc = np.sqrt(sumOfSquare.mean())                #mean of dsitance squared is variance, Lc is the SD.
    
    return Lc

#TODO: rest of Amitai

def discard_smaller_than(traj, min_length):
    """
    Gets rid of tracks that have less frame than the specified minimum
    """
    return traj.groupby('particle').filter(lambda x: len(x) >= min_length)