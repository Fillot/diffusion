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
    def __init__(self, locusList, resolution, captureRadius):
        self.resolution = resolution
        self.captureRadius = captureRadius
        # CORR:because the last point is the end of the chain,
        # it will not form a monomer
        self.monomers = np.zeros(len(locusList)-1, dtype=object)
        self._init_chain(locusList)
        self._setAABB(locusList)

    def _setAABB(self, locusList):
        min = np.min(locusList, axis=0)
        max = np.max(locusList, axis=0)
        self.AABB = [min, max]


    def _init_chain(self,locusList):
        """Instantiate every monomer from list of position,
        then links them together."""
        # TODO:Handle circular chromatin ???

        # initiate every monomer with correct 
        for index, locus_coord in enumerate(locusList[:-1]):
            direction = locusList[index+1]-locusList[index]
            self.monomers[index] = Monomer(locus_coord, direction, self.resolution)
        # link the monomers in a chain
        for index, monomer in enumerate(self.monomers):
            # TODO: There are probably better way to handle these 2 edge cases
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
            
class Monomer():
    def __init__(self, position, direction, resolution):
        #geometrical properties of the monomer
        self.from_position = position#ndarray for 3D position
        self.direction = direction
        self.to_position = position+direction
        # the length interval is treated as [0,5000[
        self.length_bp = resolution

        # references to the neighbor monomers in the chain, 
        # initiated by the chromatin object
        self.next = None
        self.previous = None

        # tracking variables
        self.particleList = []
        self.occupiedList = []
        self.reached = False
        self.firstPassageTime = np.inf
    
    def Slide(self, particleREF):
        """Returns the new 3D coordinates of a given molecule after it
        has slide in 1D on the chromatin"""
        #TODO:DANGER! if particleID is not found in particleList, the function returns nothing
        #find the particle inside the store (particle,position) pairs
        for i, (particle, position_bp) in enumerate(self.particleList):
            if particle == particleREF:
                #draw a possible move TODO:expose?
                random_move = int(np.random.normal()*100)
                new_position_bp = position_bp+random_move
                #if we are overflowing to neighboring monomers of the chromatin chain
                if new_position_bp>=self.length_bp:
                    #need to check if we are the last monomer
                    if self.next is None:
                        new_position_bp = self.length_bp
                        new_position_3D = self.from_position+self.direction
                    else:
                        new_position_3D = self.Transfer(\
                            self.next, particle, new_position_bp-self.length_bp)
                elif new_position_bp<0:
                    #need to check if we are the first monomer
                    if self.previous is None:
                        new_position_bp = 0
                        new_position_3D = self.from_position
                    else:
                        new_position_3D = self.Transfer(\
                            self.previous, particle, new_position_bp+self.length_bp)
                else:
                    new_position_3D = self.BpTo3D(new_position_bp)
                    self.particleList[i][1] = new_position_bp
                return new_position_3D
                
    def Transfer(self, to_monomer, particle, position):
        """Transfers the molecule to another monomer in
        case of a particle overshooting,
        and returns the 3D position to the Slide function
        so it can pass it back to the Update."""
        # TODO: this need to be asserted so fucking much, cause this
        # is a potential place where particle references get mixed up.

        #add to the other monomer and change monomer ref in particle
        #CORR:append(list) instead of append(tuple)
        to_monomer.particleList.append([particle, position])
        particle.slidingOn = to_monomer
        particle.sliding = True#CORR:added for robustness

        #erase particle id from own list
        self.particleList = [i for i in self.particleList if i[0]!=particle]
        return to_monomer.BpTo3D(position)
    
    def BpTo3D(self, position_bp):
        """Converts base pair coordinates into real world coordinates."""
        return self.from_position + (position_bp/self.length_bp)*self.direction

    def TrackOccupied(self, frame): 
        """Update a list keeping track of which locus where occupied.
        To be called each frame by the Solver."""
        #should mark the time of the first position in bp to be listed
        if len(self.particleList) != 0:
            #ADDED for first passage time
            if self.reached == False:
                self.firstPassageTime = frame
            for (_, position_bp) in self.particleList:
                #print(f"occL:{type(self.occupiedList)}, pbp:{type(position_bp)}, len:{len(self.particleList)}")
                self.occupiedList.append(position_bp)

    def SetParticlePosition(self, particleREF, position_bp):
        """Sets the position in bp of a given particle"""
        for i, (particle, _) in enumerate(self.particleList):
            if particle == particleREF:
                self.particleList[i][1] = position_bp
    
    def Absorb(self, particle, position_bp):
        #CORR:list instead of tuple, since tuple cannot be value assigned
        self.particleList.append([particle, position_bp])

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

class BVH():
    """
    Top down BVTree, for mesh only (not chromatin)
    It is probable that I'll end up with a different tree
    for chromatin, especially if chromatin ends up moving
    """
    def __init__(self, objectsArray):
        self.minObjPerLeaf = 2
        self.objectsArray = objectsArray# array of objects into 'soup'
        self.nodeList = []
        self.TopDownTree(0, len(objectsArray)) #first call points to first object
        
    
    def TopDownTree(self, startingIndex, numObjects):
        # assert type?
        # create new node
        node = BVHNode()
        self.nodeList.append(node)
        #compute bounding volume based on the objects
        AABB = self.ComputeAABB(startingIndex, numObjects)
        node.AABB = AABB
        # if nb object <= min, node is leaf, terminate
        if numObjects <= self.minObjPerLeaf:
            node.leaf = True
            node.numObjects = numObjects
            node.pointerToFirstObject = startingIndex
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
            
    
    def ComputeAABB(self, startingIndex, numObjects):
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


    def PartitionObjects(self, startingIndex, numObjects, axisToCut):
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

    def print_tree(self):
        """for debug purposes"""
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for node in self.nodeList:
            if node.leaf:
                node.printAABB()



class BVHNode():
    def __init__(self):
        self.leaf = False
        self.AABB = []
        self.numObjects = 1
        self.pointerToFirstObject = 0
        self.left = None
        self.right = None

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