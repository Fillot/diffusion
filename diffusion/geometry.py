import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from itertools import product, combinations
      

class Vertex():
    def __init__(self, position, index):
        self.position = position
        self.x = position[0]
        self.y = position[1]
        self.z = position[2]
        self.outgoing_halfedge = None
        self.index = index

class Face():
    #TODO:add reference to mesh containing the face,
    # to get rid of the vertices arguments
    def __init__(self, halfedge, mesh):
        self.halfedge = halfedge
        self.center = None
        self.normal = np.array([0,0,0], dtype=float)
        self.mesh = mesh
    
    def calc_face_normal(self):
        if self.normal.all() != np.array([0,0,0]).all():
            self.normal = np.array([0,0,0], dtype=float)
        v1, v2, v3 = self.getFaceVertices()
        verts = [v1, v2, v3, v1]
        for i in range(3):
            self.normal[0] += (verts[i].y - verts[i+1].y) * (verts[i].z + verts[i+1].z)
            self.normal[1] += (verts[i].z - verts[i+1].z) * (verts[i].x + verts[i+1].x)
            self.normal[2] += (verts[i].x - verts[i+1].x) * (verts[i].y + verts[i+1].y)
        self.normal = self.normal / np.linalg.norm(self.normal)
    
    def getFaceVertices(self):
        v1 = self.mesh.vertices[self.halfedge.from_vertex]
        v2 = self.mesh.vertices[self.halfedge.next.from_vertex]
        v3 = self.mesh.vertices[self.halfedge.previous.from_vertex]
        return v1, v2, v3

    def GetVerticesAsArray(self):
        v1 = self.mesh.vertices[self.halfedge.from_vertex].position
        v2 = self.mesh.vertices[self.halfedge.next.from_vertex].position
        v3 = self.mesh.vertices[self.halfedge.previous.from_vertex].position
        return v1, v2, v3
    
    def calc_center(self):
        v1, v2, v3 = self.getFaceVertices()
        self.center = (v1.position + v2.position + v3.position)/3

class Halfedge():
    def __init__(self, from_vertex):
        self.next = None
        self.previous = None
        self.from_vertex = from_vertex
        self.opposite = None
        self.face = None
        self.id = None
    
    def __str__(self):
        return "from vert {} to {}, next HE is {}".format(self.from_vertex, self.next.from_vertex, self.next)
    
    def set_key(self):
        #might get useful to store this key in the object
        return str(self.from_vertex)+'-'+str(self.next.from_vertex)

class Edge():
    def __init__(self, halfedge):
        self.halfedge = halfedge
    
    def getTuple(self):
        v1 = self.halfedge.from_vertex
        v2 = self.halfedge.next.from_vertex
        if v1 < v2:
            return (v1, v2)
        else:
            return (v2, v1)

class Mesh():
    def __init__(self, vertices, faces):
        #init vertices
        self.vertices = []
        for idx, vert in enumerate(vertices):
            self.vertices.append(Vertex(vert, idx))
        
        self.faces = np.zeros(len(faces), dtype=object)
        self.halfedges = []
        self._init_halfedges(faces)
        for face in self.faces:
            face.calc_face_normal()
            face.calc_center()            
        self.setAABB(vertices)
    
    def setAABB(self, vertices):
        min = np.min(vertices, axis=0)
        max = np.max(vertices, axis=0)
        self.AABB = [min, max]
    
    def getAABB(self):
        """Returns minx, miny, minz, maxx, maxy, maxz"""
        return self.AABB[0][0],self.AABB[0][1],self.AABB[0][2],\
               self.AABB[1][0],self.AABB[1][1],self.AABB[1][2],
    
    def _init_halfedges(self, faces):
        #from Euler's formula, E = 3/2 * F.
        #the edge map will help us keep track of what edges have
        #already been created, and pairing.
        self.edgeMap = np.zeros((int(len(faces)*3/2), 2), dtype=object)
        edgeCounter = 0
        # TODO: here i pass in the index of the vextex rather than a
        # direct reference
        for index, face in enumerate(faces):
            v1 = face[0]
            v2 = face[1]
            v3 = face[2]
            #check if one of the edges has already been assigned
            conflict = self._checkEdgeMap(face[0], face[1], face[2], self.edgeMap)
            if (conflict):
                #flip order of face
                #TODO:correct the vertice order in face list ?
                v1 = face[1]
                v2 = face[0]
            self._AddTriangle(v1, v2, v3, index)
            edgeCounter = self._UpdateEdgeMap(edgeCounter)
    
    def _checkEdgeMap(self, v1_index, v2_index, v3_index, edgeMap):
        """For a trio of vertices, checks if any of the
        halfedges has already been assigned, as this might lead
        to conflict
        
        Returns True if one conflict is detected, meaning we 
        have to flip the vertices order"""

        #array form for easy loop
        array = [v1_index, v2_index, v3_index, v1_index]
        #boolean because i don't know how to code
        boolean = False

        for i in range(3):
            if (self._checkIfEdgeExists(array[i], array[i+1])):
                boolean = True
                break
        return boolean
    
    def _checkIfEdgeExists(self, v1, v2):
        # some advice to keep the doubly connected edge list
        # but here the edgemap is really just for keeping track
        # of whether or not the edge exists at all

        #order the vertex index in increasing order
        if v1 < v2:
            tup = (v1, v2)
        else:
            tup = (v2, v1)

        for edge in self.edgeMap:
            if edge[0] == tup:
                if edge[1].halfedge.from_vertex == v1:
                    return True
        return False

    def _UpdateEdgeMap(self, edgeCounter):
        """Coordinates the call AddEdge and passes the edgeCounter between them"""
        #self.edgeCOunter ????
        edgeCounter = self._AddEdge(self.halfedges[-3], edgeCounter)
        edgeCounter = self._AddEdge(self.halfedges[-2], edgeCounter)
        edgeCounter = self._AddEdge(self.halfedges[-1], edgeCounter)
        return edgeCounter
    
    def _AddEdge(self, he, edgeCounter):
        """Verifies if a similar edge has already been created,
        if not, adds a new Edge in the edge map and increases the edge counter
        if yes, pairs the two edges."""
        # TODO:seems to throw an erro for a single face => E = 3/2 F assumes
        # closed mesh
        v1 = he.from_vertex
        v2 = he.next.from_vertex
        if v1 < v2:
            tup = (v1, v2)
        else:
            tup = (v2, v1)
        found_edge = False
        for edge in self.edgeMap[:edgeCounter, :]:
            if edge[0] == tup:
                found_edge = True
                #assigns the opposite reference for each halfedge
                edge[1].halfedge.opposite = he
                he.opposite = edge[1].halfedge
        if not found_edge:
            #print("I didn't find an edge for {} -> {}. Counter is {}".format(v1, v2, edgeCounter))
            self.edgeMap[edgeCounter, 0] = tup
            self.edgeMap[edgeCounter, 1] = Edge(he)
            edgeCounter += 1
        #if found_edge:
            #print("Edge {} <-> {} already assigned".format(v1, v2))
        return edgeCounter

    def _AddTriangle(self, v1, v2, v3, index):
        #hopefully does the assignments sort of correctly
        self.halfedges.append(Halfedge(v1))
        self.halfedges.append(Halfedge(v2))
        self.halfedges.append(Halfedge(v3))
        #a face needs to have a reference to one halfedge
        self.faces[index] = Face(self.halfedges[-3], self)

        #he have references to next he, previous he, face, and have a unique id
        #first he
        self.halfedges[-3].next = self.halfedges[-2]
        self.halfedges[-3].previous = self.halfedges[-1]
        self.halfedges[-3].face = self.faces[index]
        self.halfedges[-3].id = int(index*3 + 0)
        #second
        self.halfedges[-2].next = self.halfedges[-1]
        self.halfedges[-2].previous = self.halfedges[-3]
        self.halfedges[-2].face = self.faces[index]
        self.halfedges[-2].id = int(index*3 + 1)
        #third
        self.halfedges[-1].next = self.halfedges[-3]
        self.halfedges[-2].previous = self.halfedges[-2]
        self.halfedges[-2].face = self.faces[index]
        self.halfedges[-2].id = int(index*3 + 2)
    
    def print_mesh(self):
        #try that one
        #https://stackoverflow.com/questions/56864378/how-to-light-and-shade-a-poly3dcollection
        fig = plt.figure()
        ax = Axes3D(fig)
        for id, face in enumerate(self.faces):
            v1, v2, v3 = face.getFaceVertices()
            x = [v1.x, v2.x, v3.x]
            y = [v1.y, v2.y, v3.y]
            z = [v1.z, v2.z, v3.z]
            verts = [list(zip(x, y, z))]
            ax.add_collection3d(Poly3DCollection(
                verts, 
                edgecolor='black',
                linewidths=2, 
                alpha=0))
            normal = face.normal
            center = (v1.position + v2.position + v3.position)/3
            ax.plot([center[0], center[0]+normal[0]], 
                [center[1], center[1]+normal[1]],
                [center[2], center[2]+normal[2]],
                color='yellow')
        plt.show()

    def contains(self, point):
        """
        Returns 
        -------
        True if the position of the point is inside the mesh or on the surface
        False if outside

        Notes
        ----
        Doesn't work for non-convex polygons
        """
        for face in self.faces:
            if np.dot((face.center - point), face.normal) > 0:
                return False
        return True

class Chromatin():
    def __init__(self, locusList, resolution, captureRadius, bindingSiteList = None):
        self.resolution = resolution
        self.captureRadius = captureRadius
        self.monomers = np.zeros(len(locusList)-1, dtype=object)
        self.bindingSiteList = np.sort(bindingSiteList)
        self._init_chain(locusList)
        self._setAABB(locusList)
        self.diffusivity = 100
        self.solver = None

    def _setAABB(self, locusList):
        min = np.min(locusList, axis=0)
        max = np.max(locusList, axis=0)
        self.AABB = [min, max]


    def _init_chain(self,locusList):
        """Instantiate every monomer from list of position,
        then links them together."""

        # initiate every monomer with correct 
        for index, locus_coord in enumerate(locusList[:-1]):
            direction = locusList[index+1]-locusList[index]
            self.monomers[index] = Monomer(locus_coord, direction, self.resolution)
        # link the monomers in a chain
        for index, monomer in enumerate(self.monomers):
            monomer.index = index
            monomer.chromatin = self
            # first monomer doesn't have previous
            if index==0:
                monomer.next = self.monomers[index+1]
                continue
            # last doesn't have next
            if index==len(self.monomers)-1:
                monomer.previous = self.monomers[index-1]
                continue
            # we aren't at an extremity
            monomer.previous = self.monomers[index-1]
            monomer.next = self.monomers[index+1]
            
        self.dispatchBindingSites()
    
    def dispatchBindingSites(self):
        """All adjustments needed because position in bp for each
        monomer starts at 0 and ends at the specified length."""

        if self.bindingSiteList is None:
            pass

        for i, monomer in enumerate(self.monomers):

            start = i*monomer.length_bp
            end = (i+1)*monomer.length_bp-1
            idx_start = np.searchsorted(self.bindingSiteList, start, 'left')
            idx_end = np.searchsorted(self.bindingSiteList, end, 'right')
            rng = np.arange(idx_start, idx_end)
            bs = self.bindingSiteList[rng] - start
            monomer.bindingSiteList = bs
    
    def TranslateArray(self, slidingArray):
        res = np.zeros((len(slidingArray), 3))
        for i, frame in enumerate(slidingArray):
            if not frame[0]:
                res[i,0]=False
                res[i,1]=None
                res[i,2]=False
                continue
            #index of the segment of chromatin
            res[i,0]=True
            #position in bp, absolute
            res[i,1]=frame[0].index*self.resolution+frame[1]
            #is it the position of a binding site?
            if res[i,1] in self.bindingSiteList:
                res[i,2]=True
            else:
                res[i,2]=False
        return res

            

            
class Monomer():
    def __init__(self, position, direction, resolution):
        #geometrical properties of the monomer
        self.from_position = position#ndarray for 3D position
        self.direction = direction
        self.to_position = position+direction
        
        self.length_bp = resolution
        self.index = 0

        # references to the neighbor monomers in the chain, 
        # initiated by the chromatin object
        self.next = None
        self.previous = None
        self.chromatin = None
        self.bindingSiteList = []

        # tracking variables
        self.occupiedList = []
        self.reached = False
        self.firstPassageTime = np.inf
    
    def Slide(self, particle):
        """Returns the new 3D coordinates of a given molecule after it
        has slide in 1D on the chromatin"""

        frame = self.chromatin.solver.currentFrame
        pos_bp = particle.slidingArray[frame, 1]
        random_move = int(np.random.normal()*self.chromatin.diffusivity)
        new_pos_bp = pos_bp+random_move

        #if we overflow
        if new_pos_bp>=self.length_bp:
            #check own binding sites from pos to end
            bp = self.CheckForBindingSite(pos_bp, self.length_bp)
            if bp:
                particle.bound = True
                return self.UpdateParticlePosition(particle, bp)
            #check if there is next
            if self.next:
                #move up with transfer(), which updates particle info
                new_position_3D = self.Transfer(\
                    self.next, particle, new_pos_bp-self.length_bp, True)
                return new_position_3D
            else:
                #no next monomer, cap displacement TODO:release???
                return self.UpdateParticlePosition(particle, self.length_bp)
        
        #if we underflow
        if new_pos_bp<0:
            #check own binding sites from pos to start
            bp = self.CheckForBindingSite(pos_bp, 0)
            if bp:
                particle.bound = True
                return self.UpdateParticlePosition(particle, bp)
            #check if there is previous
            if self.previous:
                #move down with transfer(), which updates particle info
                new_position_3D = self.Transfer(\
                    self.previous, particle, new_pos_bp+self.length_bp, False)
                return new_position_3D
            else:
                #update particle info, capping
                return self.UpdateParticlePosition(particle, 0)
        
        #if we didn't get out of bounds
        #check for binding sites between pos & new_pos
        bp = self.CheckForBindingSite(pos_bp, new_pos_bp)
        if bp:
            particle.bound = True
            return self.UpdateParticlePosition(particle, bp)

        return self.UpdateParticlePosition(particle, new_pos_bp)


    def UpdateParticlePosition(self, particle, bp):
        """
        Writes [thisMonomer, pos_bp] in particle sliding array
        for the next frame.
        Returns the right 3D position to the solver.
        """
        frame = self.chromatin.solver.currentFrame
        particle.slidingArray[frame+1,0] = self
        particle.slidingArray[frame+1,1] = bp
        position_3D = self.BpTo3D(bp)
        return position_3D
      
    def Transfer(self, to_monomer, particle, final_pos_bp, transferUp):
        """Transfers the molecule to another monomer in
        case of a particle overshooting,
        and returns the 3D position to the Slide function
        so it can pass it back to the Update."""

        #we need to know if the particle is sweeping in from the top or bottom        
        if (transferUp):
            pos_bp = to_monomer.CheckForBindingSite(0, final_pos_bp)
        else:
            pos_bp = to_monomer.CheckForBindingSite(self.length_bp, final_pos_bp)
        
        # if pos_bp isn't None, it contains the position
        # of the first binding site in the sweep
        if pos_bp:
            particle.bound = True
            return to_monomer.UpdateParticlePosition(particle, pos_bp)

        #pos_bp was none, we didn't encounter any binding sites
        return to_monomer.UpdateParticlePosition(particle, final_pos_bp)
    
    def CheckForBindingSite(self, start_pos, end_pos):
        """Checks if there are any binding site within
        the window of displacement from start to end pos.
        Returns:
            the one closest to start_pos if there are several
            None if there aren't any"""
        r = range(*sorted((start_pos, end_pos)))
        best = self.length_bp
        bp = None
        for bs in self.bindingSiteList:
            if bs in r:
                length = abs(bs - start_pos)
                if length<best:
                    bp = bs
        return bp

    def BpTo3D(self, position_bp):
        """Converts base pair coordinates into real world coordinates."""
        return self.from_position + (position_bp/self.length_bp)*self.direction


#read obj
def parse_obj(file):
    """Reads obj files to extract vertex and face list, so that it can be
    used to initialize a mesh
    """
    #TODO:take advantage of vertex normal ??
    with open(file, 'r') as obj:
        datos = obj.read()

    lineas = datos.splitlines()
    vertices = []
    faces = []

    for linea in lineas:
        elem = linea.split()
        if elem:
            if elem[0] == 'v':
                vertices.append([float(elem[1]), float(elem[2]), float(elem[3])])
            elif elem[0] == 'f':
                f = []
                for i in range(1,len(elem)):
                    vs = [int(e) for e in elem[i].replace('//','/').split('/')]
                    vs = [int(e) for e in elem[i].split('/') if e]
                    f.append(vs[0]-1)
                faces.append(f)
    return np.array(vertices), np.array(faces)

class BindingSite():
    def __init__(self, position, radius):
        #guard to force ndarrray
        if type(position).__module__ != np.__name__:
            self.position = np.array(position)
        else:
            self.position = position
        self.radius = radius
        self.Rsq = radius*radius
        self.occupied = False
        self.tracking = False

    def contains(self, point):
        if (np.sum(np.square(point-self.position))<self.Rsq):
            return True
        return False
    
    def track_occupency(self, t_step):
        self.occupency = np.zeros((2,t_step))
        self.occupency[0, :] = range(t_step)
        self.tracking = True
    
    def update_occupency(self,t):
        self.occupency[1,t]=self.occupied

