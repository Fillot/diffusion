import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from itertools import product, combinations
from diffusion import geometry
import ctypes

class Solver():
    def __init__(self, particleList, simulator):
        self.mesh = simulator.mesh
        self.particleList = particleList
        self.chromatin = simulator.chromatin
        self.chromatin.solver = self
        self.NonSpeBinding = simulator.NSB
        self.SpeBinding = simulator.SB
        #initiate the bounding volume trees
        self.BVHfaces = BVH(self.mesh.faces)
        self.BVHchroma = BVH(self.chromatin.monomers)
        #import the C library
        #WARNING: this string reference is the devil
        lib = ctypes.cdll.LoadLibrary('./diffusion/CollisionChecks.so')
        self.TriangleInteresect = lib.TriangleInteresect
        self.TriangleInteresect.restype = ctypes.c_bool
        #tracking references
        self.currentParticle = None#DANGER for threads
        self.currentFrame = 0
    
    def Update(self):
        # for all particles in particleList,
        # update their position, check for collision
        for particle in self.particleList:
            self.currentParticle = particle
            if particle.slidingArray[self.currentFrame, 0]:
                new_position, _ = self.HandleSliding()
            else:
                position = particle.positionArray[self.currentFrame,:]
                velocity = self.Displace()#we could return the velocity ndarray directly...
                new_position, new_velocity = self.SolveCollision(position, velocity)

            particle.positionArray[self.currentFrame+1,:] = new_position

        self.currentFrame += 1

    def HandleSliding(self):
        """
        Draws randomly to decide if particle keeps sliding or not.
        Returns new position, new_velocity in 3D to Update
        """
        if self.currentParticle.bound:
            if np.random.uniform(0,1)>self.SpeBinding:#escape
                velocity = self.Displace()
                position = self.currentParticle.positionArray[self.currentFrame,:]
                monomer = self.currentParticle.slidingArray[self.currentFrame, 0]
                if self.CheckExitFromChroma(position, velocity, monomer):
                    return self.SolveCollision(position, velocity)
                else:
                    return self.HandleRecapture(position+velocity, monomer)

            else:#stay at binding site
                #update the bp position, return 3D position
                self.currentParticle.slidingArray[self.currentFrame+1, :]\
                    =self.currentParticle.slidingArray[self.currentFrame, :]
                return self.currentParticle.positionArray[self.currentFrame, :], np.array([0,0,0]) 


        if np.random.uniform(0,1)>self.NonSpeBinding:
            velocity = self.Displace()
            position = self.currentParticle.positionArray[self.currentFrame,:]
            monomer = self.currentParticle.slidingArray[self.currentFrame, 0]

            if self.CheckExitFromChroma(position, velocity, monomer):
                return self.SolveCollision(position, velocity)
            else:
                return self.HandleRecapture(position+velocity, monomer)
        else:
            # Slide in 1D because we haven't tried to escape
            return self.Handle1DDiffusion(), np.array([0,0,0]) 
        
    def Handle1DDiffusion(self):
        """
        Basically just calls the sliding function that sits on the monomer.

        Returns:
            the new_position (in 3D coord) and a null velocity vector."""
        new_position = self.currentParticle.slidingArray[self.currentFrame, 0].Slide(self.currentParticle)
        return new_position

    def HandleRecapture(self, position, monomer):
        """
        Returns the new 3D position in the update,
        while updating the bp position on the monomer
        """
        vec = position-monomer.from_position
        d = monomer.direction
        # scale factor of vec on the directing vector
        a=np.dot(d, vec)/np.dot(d,d)
        new_position_bp = int(a*monomer.length_bp)
        new_pos_3D = monomer.UpdateParticlePosition(self.currentParticle, new_position_bp)
        return new_pos_3D, np.array([0,0,0])
    
    def CheckExitFromChroma(self, position, velocity, monomer):
        """Returns True if the particle's displacement has
        brought it outside of the monomer."""
        n = velocity
        d = monomer.direction
        m = position - monomer.from_position
        r = self.chromatin.captureRadius
        md = np.dot(m,d)
        nd = np.dot(n,d)
        dd = np.dot(d,d)
        nn = np.dot(n,n)
        mn = np.dot(m,n)
        a = dd*nn-nd*nd
        k = np.dot(m,m)-r*r
        c = dd*k-md*md
        # check if the end point is outside either cap of the cylinder 
        # TODO:maybe this can be put before computing the stuff before
        normal=d/np.linalg.norm(d)
        new_position=position+velocity
        #signed distance to P side
        distP=np.dot(normal, (new_position-monomer.from_position))
        if distP<0:
            #print("DEBUG: End point farther than P side")
            return True
        # signed distance to Q side
        distQ=np.dot(normal, (new_position-monomer.to_position))
        if distQ>0:
            #print("DEBUG: End point farther than Q side")
            return True
        #special case, displacement is parallel to cylinder
        if (abs(a)< 1e-11):
            # print("DEBUG: a=0")
            return False
        
        b = dd*mn-nd*md
        discr = b*b-a*c#it is NOT -4ac, this is not a mistake
        #no roots, no intersection
        if discr<0:
            # print(f"DEBUG: discriminant below 0.")
            return False
        t = (-b+np.sqrt(discr))/a#compare to the other function, it do +sqrt(discrim)
        #intersection is outside segment, this means we didn't leave the cylinder
        #not through the side at least
        if not 0<t<1:
            # print(f"DEBUG: the segment does reach out")
            return False
        #if we end up here, the particle has escape through the side of the cylinder
        # print("DEBUG: intersects the cylinder")
        return True       

    def SolveCollision(self, position, velocity):
        best_t = np.inf
        colliding_monomer = None
        #broad phase chromatin
        potentiallyCollidingMonomers = self.BVHchroma.BroadPhase(position, position+velocity)
        #check every monomer for collision and keep the first collision
        for monomer in potentiallyCollidingMonomers:#TODO:does that break if list is empty?
            t = self.DetectChromaCollision(position, velocity, monomer)
            if t != 0:#should be walrus operator in python3.8+
                best_t = t
                colliding_monomer = monomer

        # if a collision has been detected, handle the absorption 
        # and return 3D coord to the update
        if colliding_monomer is not None:
            new_position, new_velocity = \
                self.AbsorbOnChroma(position, velocity, best_t, colliding_monomer)
            return new_position, new_velocity
        
        # if no collision with chromatin is detected, we check for containment
        else:
            faceList = self.BVHfaces.BroadPhase(position, position+velocity)
            if not faceList:
                return position+velocity, velocity
            new_position, new_velocity = self.NarrowPhase(position, velocity, faceList)
            return new_position, new_velocity
            

    def AbsorbOnChroma(self, position, velocity, best_t, colliding_monomer):
        """Puts (current particle, position_bp) on the monomer, 
        and returns position_3D to the update."""
        # Calculate position in bp based on intersection point
        intersectionPoint = position+best_t*velocity
        monomer_base = colliding_monomer.from_position
        d = colliding_monomer.direction
        v = intersectionPoint - monomer_base
        w=np.dot(v,d)/np.dot(d,d)
        # w is the projection of v on the directing vector of the polymer
        w=w*d
        d_norm = np.linalg.norm(d)
        w_norm = np.linalg.norm(w)
        # this is the scale factor, 
        # how far along the directing vector was the intersection point
        ratio = w_norm/d_norm
        position_bp = int(colliding_monomer.length_bp*ratio)

        # Now update the status of everyone involved
        position_3D = colliding_monomer.UpdateParticlePosition(self.currentParticle, position_bp)
        return position_3D, np.array([0,0,0])

    def DetectChromaCollision(self, position, velocity, monomer):
        """
        Returns 0 (False) if the particle doesn't cross the monomer
        Returns t the time of crossing if it does.
        """
        n = velocity
        d = monomer.direction
        m = position - monomer.from_position
        r = self.chromatin.captureRadius
        md = np.dot(m,d)
        nd = np.dot(n,d)
        dd = np.dot(d,d)
        nn = np.dot(n,n)
        mn = np.dot(m,n)
        a = dd*nn-nd*nd
        k = np.dot(m,m)-r*r
        c = dd*k-md*md
        #special case, displacement is parallel to cylinder
        if (abs(a)< 1e-11):
            #print("DEBUG: a=0")
            if c>0:
                #first pos is outside cylinder, thus no intersetion
                #print("DEBUG: c>0")
                return 0
            #endcap tests
            if md<0:
                #print("DEBUG: parallel displacement hit the P side")
                t = -mn/nn
                return t
            elif md>dd:
                #print("DEBUG: parallel displacement hit the Q side")
                t = (nd-mn)/nn
                return t
            else:
                #first pos is inside cylinder
                #print("DEBUG: first pos is inside cylinder")
                t = 0
                return t
        
        b = dd*mn-nd*md
        discr = b*b-a*c#it is NOT -4ac, this is not a mistake

        #no roots, no intersection
        if discr<0:
            #print("DEBUG: discriminant below 0.")
            return 0
        
        t = (-b-np.sqrt(discr))/a
        #intersection is outside segment
        if not 0<t<1:
            #print(f"DEBUG: No intersection, with t={t}.")
            return 0
        #intersection outside cylinder on p side
        if (md+t*nd<0):
            #print("DEBUG: intersects infinite cylinder on p side")
            #pointing away
            if (nd<0):
                #print("DEBUG: but does not cross the plane")
                return 0
            t = -md/nd
            #return True or False whether the intersection in
            #inside the endcap
            new=(position+t*n)-monomer.from_position#distance from intersect to center of face P
            cross = np.dot(new,new)#TODO:why the fuck doesn't the source code work ?
            if cross-r*r <= 0.0:
                #print("DEBUG: crosses the base")
                return t
            else:
                #print("DEBUG: doesn't cross the base")
                return 0
        #intersection outside cylinder on q side
        elif (md+t*nd>dd):
            #print("DEBUG: intersects infinite cylinder on q side")
            #pointing away
            if (nd>=0):
                #print("DEBUG: but does not cross the plane")
                return 0
            t=(dd-md)/nd
            #return True or False whether the intersection in
            #inside the endcap
            if k+dd-2*md+t*(2*(mn-nd)+t*nn)<=0:
                #print("DEBUG: crosses the base")
                return t
            else:
                #print("DEBUG: doesn't cross the base")
                return 0 
        #if we end up here, the segment intersect cyclinder
        #between the endcaps. We return t
        #print("DEBUG: intersects the cyinder")
        return t

    def NarrowPhase(self, position, velocity, faceList):
        """NarrowPhase"""
        new_position = position + velocity
        new_velocity = velocity
        # tracking variables to keep track of the current
        # closest colliding face and its time of crossing
        u = ctypes.c_double()
        v = ctypes.c_double()
        w = ctypes.c_double()
        t = ctypes.c_double()
        bestT = 1
        collidingFaceIndex  = None
        for i, f in enumerate(faceList):

            A, B, C = f.GetVerticesAsArray()

            #TODO: declare the arg types before to make this call 
            # less atrocious ??
            if (self.TriangleInteresect(
                ctypes.c_void_p(position.ctypes.data),
                ctypes.c_void_p(new_position.ctypes.data),
                ctypes.c_void_p(A.ctypes.data),
                ctypes.c_void_p(B.ctypes.data),
                ctypes.c_void_p(C.ctypes.data),
                ctypes.byref(u),
                ctypes.byref(v),
                ctypes.byref(w),
                ctypes.byref(t))):
                if t.value<bestT:
                    new_position = u.value*A+v.value*B+w.value*C
                    bestT = t.value
                    collidingFaceIndex = i
                    
        #compute new velocity vector if a collision happened
        if collidingFaceIndex is not None:
            alongNorm = 2*np.dot(velocity,faceList[collidingFaceIndex].normal)* \
                faceList[collidingFaceIndex].normal
            new_direction = velocity - alongNorm
            new_velocity = (1-bestT)*new_direction
            #recursively call until we don't cross any faces
            new_position, new_velocity = \
                self.SolveCollision(new_position, new_velocity)
        return new_position, new_velocity

    def Displace(self, diffusivity=10, fps=1000, mpp=1):
        # TODO:sort out the units in all of this
        # diffusivity should be hold either by particle itself or simulator
        # same thing for fps and mpp
        time_interval = 1 / fps
        displacement = np.random.normal(0,1,3).astype(ctypes.c_double)
        displacement = displacement * np.sqrt(2*diffusivity*time_interval) * mpp
        return displacement


class BVH():
    """
    Top down BVTree
    """
    def __init__(self, objectsArray):
        self.minObjPerLeaf = 2
        self.objectsArray = objectsArray# array of objects into 'soup'

        if isinstance(objectsArray[0], geometry.Monomer):
            self.PartitionObjects = getattr(self,"partitionChromatin")
            self.ComputeAABB = getattr(self,"AABBChromatin")
        elif isinstance(objectsArray[0], geometry.Face):
            self.PartitionObjects = getattr(self,"partitionFace")
            self.ComputeAABB = getattr(self,"AABBFace")

        self.nodeList = []
        self.TopDownTree(0, len(objectsArray)) #first call points to first object
        for i, node in enumerate(self.nodeList):
            node.index = i
        
    def TopDownTree(self, startingIndex, numObjects):
        # assert type?
        # create new node
        node = BVHNode()
        self.nodeList.append(node)
        #compute bounding volume based on the objects
        AABB = self.ComputeAABB(startingIndex, numObjects)
        node.AABB = AABB
        node.numObjects = numObjects
        node.pointerToFirstObject = startingIndex
        # if nb object <= min, node is leaf, terminate
        if numObjects <= self.minObjPerLeaf:
            node.leaf = True

        else:
            # else, node is not leaf,
            # partition objects
            axisToCut = np.argmax(AABB[1]) #0=x, 1=y, 2=z
            partitionPoint = self.PartitionObjects(startingIndex, numObjects, axisToCut)
            # recursively call function on the two partitions
            # keeping the current index of the node in the list
            # to assign the children
            leftNodeIndex = len(self.nodeList)
            numObjLeft = partitionPoint-startingIndex
            self.TopDownTree(startingIndex, numObjLeft)
            node.left = self.nodeList[leftNodeIndex]
            rightNodeIndex = len(self.nodeList)
            self.TopDownTree(startingIndex+numObjLeft, numObjects-numObjLeft)
            node.right = self.nodeList[rightNodeIndex]
            
    def AABBFace(self, startingIndex, numObjects):
        """
        Get vertices from face, then get the maximum and minimum along each axis
        Return AABB
        """
        subArray = self.objectsArray[startingIndex:startingIndex+numObjects]
        n = len(subArray)
        vertexArray = np.zeros((n*3,3))
        for i, face in enumerate(subArray):
            v1, v2, v3 = face.getFaceVertices()
            vertexArray[3*i] = v1.position
            vertexArray[3*i+1] = v2.position
            vertexArray[3*i+2] = v3.position

        min = np.min(vertexArray, axis=0)
        max = np.max(vertexArray, axis=0)
        return [min, max]

    def AABBChromatin(self, startingIndex, numObjects):
        """
        Get vertices from face, then get the maximum and minimum along each axis
        Return AABB
        """
        subArray = self.objectsArray[startingIndex:startingIndex+numObjects]
        n = len(subArray)
        vertexArray = np.zeros((n*2,3))
        for i, monomer in enumerate(subArray):
            vertexArray[2*i] = monomer.from_position
            vertexArray[2*i+1] = monomer.to_position

        min = np.min(vertexArray, axis=0)
        max = np.max(vertexArray, axis=0)
        return [min, max]

    def partitionFace(self, startingIndex, numObjects, axisToCut):
        """
        From a subset of all total objects, sorts them by their centroid
        along a specified axis (x, y,z) and update the object list
        """

        # get the subarray
        subArrayOfFace = self.objectsArray[startingIndex:startingIndex+numObjects]
        arrayOfCentroid = np.zeros((4, len(subArrayOfFace)))
        for i, face in enumerate(subArrayOfFace):
            centroid = face.center
            arrayOfCentroid[0,i]=centroid[0]
            arrayOfCentroid[1,i]=centroid[1]
            arrayOfCentroid[2,i]=centroid[2]
            arrayOfCentroid[3,i]=int(i)
        
        sortedArgs = arrayOfCentroid[axisToCut,:].argsort()
        # sort the object array along the index to cut
        arrayOfCentroid = arrayOfCentroid[:,sortedArgs]

        # if we reassign the sorted subarray,
        # we should be okay
        subArrayOfFace = subArrayOfFace[arrayOfCentroid[3,:].astype(int)]
        self.objectsArray[startingIndex:startingIndex+numObjects] = subArrayOfFace
        
        # even though we technically don't need it, 
        # I'll return a partition point since partition 
        # scheme can be other things than median

        median = int(len(subArrayOfFace)/2)
        return startingIndex+median

    def partitionChromatin(self, startingIndex, numObjects, axisToCut):
        """
        From a subset of all total objects, sorts them by their centroid
        along a specified axis (x, y,z) and update the object list
        """

        # get the subarray
        subArrayOfMonomer = self.objectsArray[startingIndex:startingIndex+numObjects]
        arrayOfCentroid = np.zeros((4, len(subArrayOfMonomer)))
        for i, monomer in enumerate(subArrayOfMonomer):
            centroid = (monomer.from_position+monomer.to_position)/2
            arrayOfCentroid[0,i]=centroid[0]
            arrayOfCentroid[1,i]=centroid[1]
            arrayOfCentroid[2,i]=centroid[2]
            arrayOfCentroid[3,i]=int(i)
        
        sortedArgs = arrayOfCentroid[axisToCut,:].argsort()
        # sort the object array along the index to cut
        arrayOfCentroid = arrayOfCentroid[:,sortedArgs]

        # if we reassign the sorted subarray,
        # we should be okay
        subArrayOfMonomer = subArrayOfMonomer[arrayOfCentroid[3,:].astype(int)]
        self.objectsArray[startingIndex:startingIndex+numObjects] = subArrayOfMonomer
        
        # even though we technically don't need it, 
        # I'll return a partition point since partition 
        # scheme can be other things than median

        median = int(len(subArrayOfMonomer)/2)
        return startingIndex+median
    
    def BroadPhase(self, p, q):
        """Collision logic"""
        particleAABB = [np.minimum(p,q), np.maximum(p,q)]
        collidingResults = []
        stack = [self.nodeList[0]]
        n_cycles = 0
        while len(stack)>0:
            n_cycles += 1
            current = stack[-1]
            if self.AABBOverlap(current.AABB, particleAABB):
                if current.leaf:
                    collidingResults.append(current)
                    del stack[-1]
                    continue
                else:
                    del stack[-1]
                    stack.append(current.left)
                    stack.append(current.right)
                    continue
            else:
                del stack[-1]
                continue
        
        #RETURN as a polygon soup of to be tested in narrow phase
        soup = []
        for res in collidingResults:
            pointer = res.pointerToFirstObject #TODO:shorten variable name
            for i in range(pointer, pointer+res.numObjects):
                soup.append(self.objectsArray[i])
        return soup
    
    def AABBOverlap(self, a, b):
        """If separated along one axis, there is no
        overlap"""
        if (a[1][0]<b[0][0] or a[0][0]>b[1][0]):
            return False
        if (a[1][1]<b[0][1] or a[0][1]>b[1][1]):
            return False
        if (a[1][2]<b[0][2] or a[0][2]>b[1][2]):
            return False
        return True

    def print_tree(self):
        """for debug purposes"""
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for node in self.nodeList:
            if node.leaf:
                node.printAABB()

    def treeToString(self, starting_node, depth = 0, line_stack = []):
        """Print out the tree in string form in the console
        for debug"""
        tree = self.nodeList[starting_node]
        line = "--" * (depth+1) + " "
        line = line+str(tree.index) + " "
        #line = line+ f"{tree.pointerToFirstObject};{tree.numObjects}"
        if tree.leaf:
            line += ">"
        #line +=f"min:({tree.AABB[0][0]:0.01f},{tree.AABB[0][1]:0.01f},{tree.AABB[0][2]:0.01f}) "\
        #    f"max:({tree.AABB[1][0]:0.01f},{tree.AABB[1][1]:0.01f},{tree.AABB[1][2]:0.01f})"
        line_stack.append(line)
        if not tree.leaf:
            depth +=1
            line_stack = self.treeToString(tree.left.index, depth = depth, line_stack = line_stack)
            line_stack = self.treeToString(tree.right.index, depth = depth, line_stack = line_stack)
        return line_stack


class BVHNode():
    def __init__(self):
        #for nodes
        self.index = 0
        self.AABB = []
        self.left = None
        self.right = None
        #if leaf, 
        self.leaf = False
        self.numObjects = None
        self.pointerToFirstObject = None

    def printAABB(self):
        """For debug"""
        ax = plt.gca()
        # draw cube
        r = [0, 1]
        min = self.AABB[0]
        max = self.AABB[1]
        scale = max-min
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            print(s,e)
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                s = s*scale+min
                e = e*scale+min
                ax.plot3D(*zip(s, e), color="b")


# np.random.seed(42)
# vertices, faces = geometry.parse_obj("./3Dmesh/icosphere.obj")
# sphere = geometry.Mesh(vertices, faces)
# sim = Simulator(sphere)
# print("starting")
# tic = time.perf_counter()
# sim.Simulate(10,10)
# toc = time.perf_counter()
# print(f"solving took {toc - tic:0.4f} seconds")
# len(faces)
# print(np.mean(sim.solver.perf))