import numpy as np
import pymesh

#### TODO: Redo with pytorch3d

def load_mesh(path_mesh):
    """
    Because pymesh.read() couldn't read the G6-meshes from Mariam.

    """

    with open(path_mesh) as file:
        for i, line in enumerate(file):
            if 'element vertex' in line:
                n_vertices = int(line.split('element vertex')[-1])
            if 'element face' in line:
                n_faces = int(line.split('element face')[-1])
            if 'end_header' in line:
                n_lines_in_header = i+1
                break

    vertices = []
    faces = []

    with open(path_mesh) as file:
        for i, line in enumerate(file):
            if (i >= n_lines_in_header) and (i < n_lines_in_header+n_vertices):

                x, y, z = line.strip().split(' ')
                vertices.append([float(x), float(y), float(z)])

            elif (i >= n_lines_in_header+n_vertices):

                a, b, c, d = line.strip().split(' ')
                faces.append([float(b), float(c), float(d)])

    vertices = np.array(vertices)
    faces = np.array(faces)

    mesh = pymesh.meshio.form_mesh(vertices, faces, voxels=None)

    return mesh



def get_hole(edges_bad_unsorted, holes):
    """
    Adapted from MATLAB code by Mariam Andersson (maande@dtu.dk).

    TBW.
    """

    # keep track of changes in edges_bad_unsorted
    len_0 = len(edges_bad_unsorted)
    something_changed = True

    # while we still find edges associated with the current hole of interest, keep iterating
    while something_changed and len(edges_bad_unsorted)>0:

        idxs_to_delete = []

        for idx, edge_oi in enumerate(edges_bad_unsorted):

            if len(set(np.ravel(holes[-1])).intersection(set(edge_oi))) > 0:

                holes[-1] = np.vstack((holes[-1], edge_oi))

                idxs_to_delete.append(idx)
            else:
                continue

        edges_bad_unsorted = np.delete(edges_bad_unsorted, idxs_to_delete, axis=0)

        if len(edges_bad_unsorted) == len_0:
            something_changed = False
        else:
            len_0 = len(edges_bad_unsorted)

    return edges_bad_unsorted, holes



def get_face_normals(vertices, faces):
    """


    """

    # create mesh-obj in order to utilize pymesh functions
    mesh = pymesh.form_mesh(vertices, faces)

    # compute face normals
    mesh.add_attribute('face_normal')

    # extract face normals
    face_normals = mesh.get_attribute('face_normal').reshape(len(faces), 3)

    return face_normals



def close_holes(mesh):
    """
    Adapted from MATLAB code by Mariam Andersson (maande@dtu.dk).

    Not perfect.

    Works for round and not too weird holes that points away from the center of
    mass of the original mesh vertices.

    Reasoning:
        - In a closed mesh, each edge will occur twice - in two different faces.
        - If an edge occurs only once, it must be on the edge of a hole.

    Method:
        - Get all edges.
        - Find the edges that only occur once.
        - Group those edges w.r.t. the holes they are associated with.
        - For each hole:
            - Compute the center of mass, and add this point as a vertex.
            - Generate new faces by combining all associated edges with the new
              vertex.
            - Make sure the face normals are ok.
        - Enjoy your new closed mesh.

    Parameters
    ----------
    mesh : pymesh.Mesh.Mesh
        Mesh for which holes are to be closed.

    Returns
    -------
    mesh : pymesh.Mesh.Mesh
        Mesh for which holes has been closed.

    """

    # # can't set attribute of mesh-obj. new obj must be created.
    vertices = mesh.vertices
    faces = mesh.faces

    # get edges
    edges = np.sort(np.concatenate((mesh.faces[:, :2], mesh.faces[:, 1:], mesh.faces[:, [0, 2]]), axis=0), axis=-1)
    assert len(edges) == 3*len(mesh.faces), 'Number of edges were computed wrongly.' # for triangular mesh
    edges_unique, edges_idxs_unique, edges_counts_unique = np.unique(edges, return_index=True, return_counts=True, axis=0)

    # check if any holes exist
    if edges.shape[0] == 2*edges_unique.shape[0]:
        # every edge is used twice... consistent with a closed manifold mesh
        print('No problem! No holes here.')
        return mesh
    else:
        pass

    # find bad edges, i.e. the edges associated witht the hole
    edges_bad_idxs = edges_idxs_unique[edges_counts_unique == 1]
    edges_bad = edges[edges_bad_idxs, :]
    edges_bad_unsorted = np.copy(edges_bad)

    #### group the edges for each hole
    holes = [] # list to contain an np.array of the involved edges per hole

    while len(edges_bad_unsorted) > 0:

        holes.append(edges_bad_unsorted[0, :]) #move first edge to hole

        edges_bad_unsorted = np.delete(edges_bad_unsorted, 0, axis=0) #delete first edge from edges

        edges_bad_unsorted, holes = get_hole(edges_bad_unsorted, holes) #get all edges associated with that hole

    #### Inform
    print(f'Closing {len(holes)} holes.')

    #### Generate patches
    center_all = np.mean(mesh.vertices, axis=0)
    center_all = center_all / np.linalg.norm(center_all) #normalize

    # For each hole
    for hole in holes:

        hole = np.atleast_2d(hole)

        #### generate new vertice
        # works for circular holes. WON'T WORK FOR ODDLY SHAPED HOLES.
        # extract vertices
        vertices_oi = mesh.vertices[np.unique(hole.ravel()), :]
        # compute new vertice as center of mass.
        vertice_new = np.mean(vertices_oi, axis=0)

        # add new vertice to vertices
        vertices = np.vstack((vertices, vertice_new))

        #### generate new faces
        # for each edge in the hole, generate a new face by connecting to center of mass
        vertice_new_idx = len(vertices) - 1
        faces_new = np.hstack((hole, np.ones((hole.shape[0], 1))*vertice_new_idx)).astype(np.int)

        #### ensure that the normal of each new face points outwards
        # compute reference vector
        vector_hole_to_center = vertice_new - center_all
        vector_hole_to_center = vector_hole_to_center / np.linalg.norm(vector_hole_to_center)

        # compute normals
        normals_new = get_face_normals(vertices, faces_new)

        # compute angles w.r.t. reference vector
        #angles = np.arccos(np.dot(center_all[np.newaxis, :], normals_new.T)) ####
        angles = np.arccos(np.dot(vector_hole_to_center[np.newaxis, :], normals_new.T)) ####

        # flip bad normals
        #faces_new[(angles < np.pi/2)[0, :], :] = np.fliplr(faces_new[(angles < np.pi/2)[0, :], :]) ####
        faces_new[(angles > np.pi/2)[0, :], :] = np.fliplr(faces_new[(angles > np.pi/2)[0, :], :]) ####

        #### add new faces to faces
        faces = np.vstack((faces, faces_new))

    mesh = pymesh.form_mesh(vertices, faces)

    return mesh
