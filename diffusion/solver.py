import numpy as np
import pandas as pd

class Solver():
    def __init__(self, mesh, particleList, chromatin):
        self.mesh = mesh
        self.particleList = particleList
        self.perf = []
        self.currentParticle = None
        self.chromatin = chromatin
        self.slidingTracker = 0
        self.currentFrame = 0
        #self.bindingSiteList = bindingSiteList
    
    def Update(self):
        # for all particles in particleList,
        # update their position, check for collision
        for particle in self.particleList:
            self.currentParticle = particle
            if particle.sliding:
                new_position, _ = self.HandleSliding()
            else:
                position = particle.positionList[-1]#all particles hold position or list of previous positions ?????
                dx, dy, dz = self._displace()#we could return the velocity ndarray directly...
                velocity = np.array([dx, dy, dz])
                new_position, _ = self.SolveCollision(position, velocity)

            particle.positionList.append(new_position)
            for monomer in self.chromatin.monomers:
                monomer.TrackOccupied(self.currentFrame)
            #Added for kon=koff
            particle.slidingList.append(particle.sliding)
        #ADDED in order to pass it monomers
        self.currentFrame += 1

    def HandleSliding(self):
        #draw for if we want to keep on sliding or not
        #maybe expose this somehow to the simulator's parameters
        if np.random.uniform(0,1)>0.9:
            dx, dy, dz = self._displace()
            velocity = np.array([dx, dy, dz])
            position = self.currentParticle.positionList[-1]
            monomer = self.currentParticle.slidingOn
            if self.CheckExitFromChroma(position, velocity, monomer):
                return self.SolveCollision(position, velocity)
            else:
                return self.HandleRecapture(position+velocity, monomer)
        else:
            # Slide in 1D because we haven't tried to escape
            return self.Handle1DDiffusion()
        
    def Handle1DDiffusion(self):
        """
        Basically just calls the sliding function that sits on the monomer.

        TODO: I don't know if that breaks the design flow to have the movement function,
        inside the geometry object. So far, this is where the position in bp is stored
        so it makes more sense to chenge it there.

        Returns:
            the new_position (in 3D coord) and a null velocity vector."""
        new_position = self.currentParticle.slidingOn.Slide(self.currentParticle)
        self.slidingTracker +=1 #DEBUG:added to figure out behavior
        return new_position, np.array([0,0,0])   

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
        new_position = monomer.BpTo3D(new_position_bp)
        monomer.SetParticlePosition(self.currentParticle, new_position_bp)
        return new_position, np.array([0,0,0])
    
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
        
        # this is ugly because the return type of DetectChromaColl
        # is not the same each time. If no collision occurs, it
        # returns 0, if it does it returns t, the time at which the collision
        # occurs
        # this is bad practise
        best_t = np.inf
        colliding_monomer = None
        #check every monomer for collision and keep the first collision
        for monomer in self.chromatin.monomers:
            t = self.DetectChromaCollision(position, velocity, monomer)
            if t != 0:
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
            new_position, new_velocity = self.ContainerCollision(position, velocity)
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
        colliding_monomer.Absorb(self.currentParticle, position_bp)
        self.currentParticle.sliding = True
        self.currentParticle.slidingOn = colliding_monomer

        # get the 3D position and pass it up to Update
        position_3D = colliding_monomer.BpTo3D(position_bp)
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

    def ContainerCollision(self, position, velocity):
        faceList = self.mesh.faces
        new_position = position + velocity
        new_velocity = velocity
        # tracking variables to keep track of the current
        # closest colliding face and its time of crossing
        bestT = 1
        collidingFaceIndex  = None
        for i, f in enumerate(faceList):
            t = self.TimeOfCrossing(position, new_position, f)
            if t<bestT:
                if self.InsideTriangle(position, new_position, f):
                    new_position = self.IntersectionPoint(position,new_position,f)
                    bestT = t
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

    def TimeOfCrossing(self, pos, new_pos, face):
        """
        Computes t the time along a displacement at which the particle
        crosses the plane of the face.
        Returns:
        np.inf if the crossing happens outside the [0,1] interval
        t (float) : time of crossing
        """
        a_idx = face.halfedge.from_vertex
        b_idx = face.halfedge.next.from_vertex
        c_idx = face.halfedge.next.next.from_vertex
        a = self.mesh.vertices[a_idx].position
        b = self.mesh.vertices[b_idx].position
        c = self.mesh.vertices[c_idx].position
        ab = b - a
        ac = c - a
        n = np.cross(ab, ac)
        qp = pos - new_pos #this is velocity

        # Compute denominator d. If d <= 0, segment is parallel to or points
        # away from triangle, so exit early
        d = np.dot(qp, n)
        if (d == 0):
            return np.inf
        
        ap = pos - a
        t = np.dot(ap, n)/d
        # t<0 : the triangle is behind, t>1 : triangle is too far
        if not 0<t<1:
            return np.inf
        return t

    def InsideTriangle(self, pos, new_pos, face):
        """
        Returns 
        """
        a_idx = face.halfedge.from_vertex
        b_idx = face.halfedge.next.from_vertex
        c_idx = face.halfedge.next.next.from_vertex
        a = self.mesh.vertices[a_idx].position
        b = self.mesh.vertices[b_idx].position
        c = self.mesh.vertices[c_idx].position
        ab = b - a
        ac = c - a
        n = np.cross(ab, ac)
        qp = pos - new_pos #this is velocity
        ap = pos - a
        # Compute denominator d. If d <= 0, segment is parallel to or points
        # away from triangle, so exit early
        e = np.cross(qp, ap)#shared calculation for v and w
        v = np.dot(ac, e)
        d = np.dot(qp, n)
        if v<0 or v>d:
            return False
        w = -np.dot(ab, e)
        if (v+w)**2>d**2 or w<0:
            return False
        return True

    def IntersectionPoint(self, pos, new_pos, face):
        """
        Returns the intersection point of the velocity segment and the face
        WARNING: DOES NOT CHECK FOR REACHABLE OR INSIDE
        ASSUMES THE CHECKS WERE ALREADY DONE
        """
        a_idx = face.halfedge.from_vertex
        b_idx = face.halfedge.next.from_vertex
        c_idx = face.halfedge.next.next.from_vertex
        a = self.mesh.vertices[a_idx].position
        b = self.mesh.vertices[b_idx].position
        c = self.mesh.vertices[c_idx].position
        ab = b - a
        ac = c - a
        n = np.cross(ab, ac)
        qp = pos - new_pos #this is velocity
        ap = pos - a
        # Compute denominator d. If d <= 0, segment is parallel to or points
        # away from triangle, so exit early
        e = np.cross(qp, ap)#shared calculation for v and w
        v = np.dot(ac, e)
        w = -np.dot(ab, e)
        ood = 1/np.dot(qp, n)
        v *= ood
        w *= ood
        u = 1 - v - w
        r = u*a + v*b + w*c
        return r

    def _displace(self, diffusivity=5, fps=1000, mpp=1):
        # TODO:sort out the units in all of this
        # diffusivity should be hold either by particle itself or simulator
        # same thing for fps and mpp
        time_interval = 1 / fps
        dx = np.random.normal(0, np.sqrt(2 * diffusivity * time_interval)) * mpp
        dy = np.random.normal(0, np.sqrt(2 * diffusivity * time_interval)) * mpp
        dz = np.random.normal(0, np.sqrt(2 * diffusivity * time_interval)) * mpp
        return dx, dy, dz

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