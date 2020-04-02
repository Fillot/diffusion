"""
intended as the library grouping all functions to analyse trajectories in a statistical
manner, classifier not included as it is in its own thing.
"""


import numpy as np

def emsd(traj, maxLagTime):
    """compute the ensemble mean square displacement for a group of trajectories
    
    Parameters
    ----------
    traj:
        list of array, with n lines, and d columns (d = dimension) TODO: is it a pandas df ? np ?
                x1  x2   x3   id
            t1|____|___|____|____| TODO:time ? frame number ? What does the line mean ?
            t2|____|___|____|____|
                ..............
            tn|____|___|____|____|
            
    maxLagTime: max interval between frames from which MSD is calculated
    
    output
    ------
    eMSD (float) : the ensemble MSD of the different trajectories studied

    Notes
    -----
    Careful about the units (either pixel/nm or frame/s)
    TODO: sort that shit out
    """
    eMSD = 0
    return eMSD

def msd(traj, tau):
    """
    Compute the mean square displacement of one particule over a range of time interval
    For a Brownian particle in n-dimension, its position noted as x = (x1, ..., xn)
    <x> la moyenne de x
    MSD(tau) = <|x(t + tau) - x(t)|^2> pour tout t
    with MSD(tau) = 2 x d x D x Tau ; d dimension of freedom ; D diffusivity constant

    Parameters
    ----------
    traj
        list of array, with n lines, and d columns (d = dimension) TODO: is it a pandas df ? np ?
            x1  x2   x3   id
        t1|____|___|____|____| TODO:time ? frame number ? What does the line mean ?
        t2|____|___|____|____|
            ..............
        tn|____|___|____|____|
    
    tau (int) : the lagTime to be considered TODO:either in frames (int) or seconds (float)
        TODO:check if tau element of |R+ or |N+ depending on the TODO just above.   
    
    """
    msd = 0
    return msd

"""TODO: fast fourier transform implementation"""

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