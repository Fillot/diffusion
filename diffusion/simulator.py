import numpy as np
import pandas as pd
from diffusion import solver
import ctypes

class Simulator():
    def __init__(self, mesh, chromatin, starting_pos='random'):
        # Simu should be about the conditions of the experiment
        # like, set-up, resolution, noise, stuff like that
        self.mesh = mesh
        self.traj = None
        self.chromatin = chromatin
        self.starting_pos = starting_pos
    
    def Simulate(self, n_particle, n_frames):
        self._initParticles(n_particle, n_frames)
        self.solver = solver.Solver(self.particleList, self)
        #-1 because we have already initialized a first position
        for i in range(n_frames-1):
            if i%50 == 0:
                print(i)
            self.solver.Update()
        self._AssembleTraj()

    #TODO: handle those kind of parametrization of the simulator
    def SetSlidingDiffusivity(self, diffusivity):
        self.chromatin.diffusivity = diffusivity
    
    def SetNonSpeBindingStrength(self, strength):
        self.NSB = strength
    
    def SetSpeBindingStrength(self, strength):
        self.SB = strength
    
    def _initParticles(self, n_particle, n_frames):
        """Creates a list of particles inside the ROI with random first position"""
        self.particleList = []
        for _ in range(n_particle):
            self.particleList.append(\
                Particle(n_frames, self.GetRandomStartingPosition()))
    
    def GetRandomStartingPosition(self):
        """Returns a random point inside the diffusible space"""
        minx, miny, minz, maxx, maxy, maxz = self.mesh.getAABB()
        done = False
        while not done:
            position = np.array([np.random.uniform(minx, maxx),\
                np.random.uniform(miny, maxy),\
                np.random.uniform(minz, maxz)], dtype = ctypes.c_double)
            if self.mesh.contains(position):
                done = True
        return position
    
    def _AssembleTraj(self):
        """Collates the trajectory lists of all particles in the 
        simulator in the usual DataFrame
        """
        trackPopulation = []
        n_frames = len(self.particleList[0].positionArray)
        frames = np.arange(n_frames)

        for id, particle in enumerate(self.particleList):
            arr = particle.positionArray
            sli = particle.slidingArray
            res = self.chromatin.TranslateArray(sli)
            tracklet = pd.DataFrame({'frame': frames,
                    'particle': id,
                    'x':arr[:,0], 
                    'y':arr[:,1],
                    'z':arr[:,2],
                    'Sliding':res[:,0],
                    'Pos BP':res[:,1],
                    'Bind':res[:,2]
                    })
            trackPopulation.append(tracklet)
        self.traj = pd.concat(trackPopulation, ignore_index=True)
    
    def GetTraj(self):
        """Little getter"""
        if self.traj is None:
            self._AssembleTraj()
        return self.traj

class Particle():
    def __init__(self, n_frames, startingPos):
        self.positionArray = np.zeros((n_frames, 3), dtype = ctypes.c_double)
        self.positionArray[0,:] = startingPos
        self.slidingArray = np.empty((n_frames,2), dtype=object)
        self.bound = False