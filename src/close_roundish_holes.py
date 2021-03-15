import numpy as np
import torch
import pytorch3d



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
    Get face normals from vertices and faces by exploiting the 
    pytorch3d.structures.Meshes class.
    
    Parameters
    ----------
    vertices : numpy.ndarray, [n_vertices, n_dimensions]
    faces : numpy.ndarray, [n_faces, n_dimensions]
    
    Returns
    -------
    face_normals : numpy.ndarray, [n_faces, n_dimensions]

    """
    vertices = torch.Tensor(vertices)
    faces = torch.Tensor(faces)

    # create pytorch3d.structures.Meshes object in order to utilize functions
    mesh = pytorch3d.structures.Meshes(verts=[vertices], faces=[faces])

    # extract face normals
    face_normals = mesh.faces_normals_list()[0]
    
    # convert back to numpy.ndarray (redundant)
    face_normals = face_normals.numpy()

    return face_normals



def close_holes(vertices, faces):
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

    Done in an awkward mix of numpy and torch due to historic reasons. But it 
    works!
    
    Parameters
    ----------
    vertices : torch.Tensor, [n_vertices, n_dimensions]
        Vertices of the mesh which holes has to be closed.
    faces : torch.Tensor, [n_faces, n_dimensions]
        Faces of the mesh which holes has to be closed.

    Returns
    -------
    _vertices : torch.Tensor, [n_vertices, n_dimensions]
        Vertices of the mesh which holes has been closed.
    _faces : torch.Tensor, [n_faces, n_dimensions]
        Faces of the mesh which holes has been closed.

    """

    # convert to numpy.ndarrays
    vertices = vertices.numpy()
    faces = faces.numpy()
    
    # for storing the new vertices and faces 
    _vertices = np.copy(vertices)
    _faces = np.copy(faces)

    # get edges
    edges = np.sort(np.concatenate((faces[:, :2], faces[:, 1:], faces[:, [0, 2]]), axis=0), axis=-1)
    assert len(edges) == 3*len(faces), 'Number of edges were computed wrongly.' # for triangular mesh
    edges_unique, edges_idxs_unique, edges_counts_unique = np.unique(edges, return_index=True, return_counts=True, axis=0)

    # check if any holes exist
    if edges.shape[0] == 2*edges_unique.shape[0]:
        # every edge is used twice... consistent with a closed manifold mesh
        print('No problem! No holes here.')
        return vertices, faces
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
    center_all = np.mean(vertices, axis=0)
    center_all = center_all / np.linalg.norm(center_all) #normalize

    # For each hole
    for hole in holes:

        hole = np.atleast_2d(hole)

        #### generate new vertice
        # works for circular holes. WON'T WORK FOR ODDLY SHAPED HOLES.
        # extract vertices
        vertices_oi = vertices[np.unique(hole.ravel()), :]
        # compute new vertice as center of mass.
        vertice_new = np.mean(vertices_oi, axis=0)

        # add new vertice to vertices
        _vertices = np.vstack((_vertices, vertice_new))

        #### generate new faces
        # for each edge in the hole, generate a new face by connecting to center of mass
        vertice_new_idx = len(_vertices) - 1
        faces_new = np.hstack((hole, np.ones((hole.shape[0], 1))*vertice_new_idx)).astype(np.int)

        #### ensure that the normal of each new face points outwards
        # compute reference vector
        vector_hole_to_center = vertice_new - center_all
        vector_hole_to_center = vector_hole_to_center / np.linalg.norm(vector_hole_to_center)

        # compute normals
        normals_new = get_face_normals(_vertices, faces_new)

        # compute angles w.r.t. reference vector
        angles = np.arccos(np.dot(vector_hole_to_center[np.newaxis, :], normals_new.T)) ####

        # flip bad normals
        faces_new[(angles > np.pi/2)[0, :], :] = np.fliplr(faces_new[(angles > np.pi/2)[0, :], :]) ####

        #### add new faces to faces
        _faces = np.vstack((_faces, faces_new))

    _vertices = torch.Tensor(_vertices)
    _faces = torch.Tensor(_faces).type(torch.long)
    
    return _vertices, _faces
