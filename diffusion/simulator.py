import numpy as np
import pandas as pd
from diffusion import solver

class Simulator():
    def __init__(self, mesh, chromatin, starting_pos='random'):
        # Simu should be about the conditions of the experiment
        # like, set-up, resolution, noise, stuff like that
        self.mesh = mesh
        self.traj = None
        self.chromatin = chromatin
        self.starting_pos = starting_pos
    
    def Simulate(self, n_particle, n_frames):
        self._initParticles(n_particle)
        self.solver = solver.Solver(self.mesh, \
                            self.particleList, \
                            self.chromatin)
        #-1 because we have already initialized a first position
        for i in range(n_frames-1):
            if i%10 == 0:
                print(i)
            self.solver.Update()
            # for p in self.particleList:
            #     print(p.positionList[-1])
        self._AssembleTraj()
    
    def _initParticles(self, n_particle):
        """Creates a list of particles inside the ROI with random first position"""
        self.particleList = []
        for n in range(n_particle):
            self.particleList.append(\
                Particle(self.GetRandomStartingPosition()))
    
    def GetRandomStartingPosition(self):
        """Returns a random point inside the diffusible space"""
        #TODO: we might want a bounding box for the ROI
        minx, miny, minz, maxx, maxy, maxz = self.mesh.getAABB()
        done = False
        while not done:
            position = [np.random.uniform(minx, maxx),\
                np.random.uniform(miny, maxy),\
                np.random.uniform(minz, maxz)]
            if self.mesh.contains(position):
                done = True
        return position
    
    def _AssembleTraj(self):
        """Collates the trajectory lists of all particles in the 
        simulator in the usual DataFrame
        """
        trackPopulation = []
        n_frames = len(self.particleList[0].positionList)
        frames = np.arange(n_frames)

        for id, particle in enumerate(self.particleList):
            arr = np.array(particle.positionList)
            tracklet = pd.DataFrame({'frame': frames,
                    'particle': id,
                    'x':arr[:,0], 
                    'y':arr[:,1],
                    'z':arr[:,2]
                    })
            trackPopulation.append(tracklet)
        self.traj = pd.concat(trackPopulation, ignore_index=True)
    
    def GetTraj(self):
        """Little getter"""
        if self.traj is None:
            self._AssembleTraj()
        return self.traj

class Particle():
    def __init__(self, startingPos):
        self.positionList = [startingPos]
        self.slidingList = [False]
        self.diffusivity = 30
        self.sliding = False
        #reference to the monomer it is currently on
        self.slidingOn = None