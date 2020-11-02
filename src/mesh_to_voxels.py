import torch
import pytorch3d as torch3d
from pytorch3d import io
import math
from tqdm.auto import tqdm
import numpy as np



def check_ray_triangle_intersection(ray_origins, ray_direction, triangle, epsilon=1e-6):
    """
    Optimized to work for:
        >1 ray_origins
        1 ray_direction multiplied to match the dimension of ray_origins
        1 triangle

    Based on: Answer by BrunoLevy at
    https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d
    Thank you!

    Parameters
    ----------
    ray_origin : torch.Tensor, (n_rays, n_dimensions), (x, 3)
    ray_directions : torch.Tensor, (n_rays, n_dimensions), (1, x)
    triangle : torch.Tensor, (n_points, n_dimensions), (3, 3)

    Return
    ------
    intersection : boolean (n_rays,)

    Test
    ----
    triangle = torch.Tensor([[0., 0., 0.],
                             [1., 0., 0.],
                             [0., 1., 0.],
                            ]).to(device)

    ray_origins = torch.Tensor([[0.5, 0.25, 0.25],
                                [5.0, 0.25, 0.25],
                               ]).to(device)

    ray_origins = torch.rand((10000, 3)).to(device)

    ray_direction = torch.Tensor([[0., 0., -10.],]).to(device)
    #ray_direction = torch.Tensor([[0., 0., 10.],]).to(device)

    ray_direction = ray_directions.repeat(ray_origins.shape[0], 1)

    check_ray_triangle_intersection(ray_origins, ray_direction, triangle)
    """

    E1 = triangle[1] - triangle[0] #vector of edge 1 on triangle
    E2 = triangle[2] - triangle[0] #vector of edge 2 on triangle
    N = torch.cross(E1, E2) # normal to E1 and E2

    invdet = 1. / -torch.einsum('ji, i -> j', ray_direction, N) # inverse determinant

    A0 = ray_origins - triangle[0]
    # print('A0.shape: ', A0.shape)
    # print('ray_direction.shape: ', ray_direction.shape)
    DA0 = torch.cross(A0, ray_direction)

    u = torch.einsum('ji, i -> j', DA0, E2) * invdet
    v = -torch.einsum('ji, i -> j', DA0, E1) * invdet
    t = torch.einsum('ji, i -> j', A0, N) * invdet

    intersection = (t >= 0.0) * (u >= 0.0) * (v >= 0.0) * ((u + v) <= 1.0)

    return intersection



def generate_voxel_grid(bbox, res):
    """
    Generates mesh grids where the points correspond to the corners of voxels.

    Parameters
    ----------
    bbox : tuple/list
        Content: xmin, xmax, ymin, ymax, zmin, zmax.
    res : float
        Resolution of the grid. Voxel-side-length.

    Return
    ------
    grid_x : torch.Tensor
    grid_y : torch.Tensor
    grid_z : torch.Tensor

    """

    xmin, xmax, ymin, ymax, zmin, zmax = bbox

    #### keep while fixing line artifact.
    xs = torch.arange(xmin, xmax+res, res)
    ys = torch.arange(ymin, ymax+res, res)
    zs = torch.arange(zmin, zmax+res, res)
    #### should be this
    #xs = torch.arange(xmin-res, xmax+res, res)
    #ys = torch.arange(ymin-res, ymax+res, res)
    #zs = torch.arange(zmin-res, zmax+res, res)

    grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs)

    return grid_x, grid_y, grid_z



def get_mesh_bbox(vertices):

    (xmin, ymin, zmin), _ = torch.min(vertices, dim=0)
    (xmax, ymax, zmax), _ = torch.max(vertices, dim=0)

    bbox = xmin, xmax, ymin, ymax, zmin, zmax
    # bbox = xmin, xmax, ymin, ymax, 0, 37.5 ####

    return bbox



def get_ray_origins(vertices, res=0.1):

    device = vertices.device

    # get bbox of mesh
    bbox = get_mesh_bbox(vertices)

    # generate grid
    grid_x, grid_y, grid_z = generate_voxel_grid(bbox, res)

    shape_grid = grid_x.shape

    # flatten for optimized computation
    grid_x_flat = grid_x.reshape(grid_x.numel(), 1)
    grid_y_flat = grid_y.reshape(grid_y.numel(), 1)
    grid_z_flat = grid_z.reshape(grid_z.numel(), 1)

    ray_origins = torch.cat((grid_x_flat, grid_y_flat, grid_z_flat), dim=1).to(device)

    return ray_origins, shape_grid



def dep_get_ray_direction(ray_direction, shape, device):

    ray_direction = torch.Tensor([ray_direction,]).to(device)

    ray_direction = ray_direction.repeat(shape, 1)

    return ray_direction



def get_ray_directions(ray_directions_singles, n_ray_origins, device):
    """
    Parameters
    ----------
    ray_directions_singles : list
    """

    #ray_direction = torch.Tensor(ray_directions_singles).to(device)

    #print('ray_direction.shape', ray_direction.shape)

    ray_direction = ray_directions_singles.repeat_interleave(n_ray_origins, dim=0)

    return ray_direction


def get_faces_associated_with_given_ray_origins(faces, vertices, ray_origins, buffer_z_lower=0.0, buffer_z_higher=0.0, epsilon=0.0):
    """

    """

    #### no need to check faces that are not within the range of of the ray_origins
    # doing batches along z
    # For a triangle to be associated with the slab of interest, one of the two conditions must be
    # fulfilled.
    #     1. >=1 of the vertice's z-coordinate must be >=slab_lower_z AND <= slab_upper_z
    # OR
    #     2. >=1 of the vertice's z-coordinate must be >=B_z AND >=1 point must be <=A_z

    (ray_origins_xmin, ray_origins_ymin, ray_origins_zmin), _ = torch.min(ray_origins, dim=0)
    (ray_origins_xmax, ray_origins_ymax, ray_origins_zmax), _ = torch.max(ray_origins, dim=0)
    # print('ray_origins_zmin: ', ray_origins_zmin)
    # print('ray_origins_zmax: ', ray_origins_zmax)

    # condition 1, if any vertex is contained by the slab
    mask_contained = (vertices[:, 2] >= (ray_origins_zmin - abs(buffer_z_lower) - epsilon)) * (vertices[:, 2] <= (ray_origins_zmax + buffer_z_higher + epsilon))
    # mask_z = mask_contained

    # mask_vertices_inside_range_oi = mask_contained
    # print('mask_contained: ', mask_contained.shape)

    mask_contained = torch.sum(mask_contained[faces], dim=1) > 0

    # condition 2, if one face is associated with vertices both above and below the slab
    condition_a = (vertices[:, 2] <= (ray_origins_zmin))
    condition_b = (vertices[:, 2] >= (ray_origins_zmax))

    mask_a = torch.sum(condition_a[faces], dim=1) > 0
    mask_b = torch.sum(condition_b[faces], dim=1) > 0

    mask_exceeding = mask_a * mask_b

    # final mask
    mask_faces_inside_range_oi = mask_contained + mask_exceeding

    faces_oi = faces[mask_faces_inside_range_oi]

    return faces_oi


def get_segmentation_from_mask_voxel_corners_inside(mask_inside):
    """
    Only if all 8 corners of a voxel are inside the mesh, the voxel is
    considered as being inside the mesh.

    """

    segmentation = mask_inside[1:, :-1, :-1] * mask_inside[:-1, 1:, :-1] * mask_inside[:-1, :-1, 1:] * \
                   mask_inside[1:, 1:, :-1] * mask_inside[1:, :-1, 1:] * mask_inside[:-1, 1:, 1:] * \
                   mask_inside[1:, 1:, 1:] * mask_inside[:-1, :-1, :-1]

    return segmentation



def batch(iterable, n=1):
    """
    Returns batch of size n.

    Parameters
    ----------
    iterable : iterable
        What ever you want batched.
    n : int
        Batch size.

    """

    l = len(iterable)

    for i in range(0, l, n):
        yield iterable[i:min(i + n, l)]



def get_angles_between_ray_directions_and_z(ray_directions):
    """
    Returns angles between ray_directions and z in radians.

    Parameters
    ----------
    ray_directions : torch.Tensor [n_ray_directions, n_dimensions]
       Ray directions.

    Returns
    -------
    theta_zs : torch.Tensor [n_ray_directions]
        Angle with respect to z for each vector in ray_directions.

    """

    normed = ray_directions / torch.norm(ray_directions, dim=1)

    theta_zs = torch.asin(normed[:, -1])

    return theta_zs



def get_buffer_lengths_z(theta_zs, shape_grid, res):
    """
    When ray_directions are not perpendicular with z, they might intersect with
    the mesh outside of the slab defined by the ray_origins. Hence, a buffer
    must be added to the area of the interval along z for which the faces are
    being checked.

    Parameters
    ----------
    theta_zs : torch.Tensor
        Angle with respect to z for each vector in ray_directions.
    shape_grid : tuple
        Shape of the grid of points that is being checked for being inside the
        mesh.
    res : float
        Spatial resolution of the grid of points.

    Returns
    -------
    buffer_lengths_z : torch.Tensor
        The z-projection for each theta_z over the maximal distance achievable
        (the diagonal) over the grid of points.

    """

    # buffer_lengths_z = torch.sin(theta_zs) * max(shape_grid[:2]) * res
    buffer_lengths_z = torch.sin(theta_zs) * (math.sqrt(shape_grid[0]**2 + shape_grid[1]**2)) * res

    return buffer_lengths_z



def check_if_voxel_corners_are_inside_mesh(path_mesh, res, device, inspection_mode=False):
    """
    Main function. Combines
    """

    #### load mesh
    vertices, faces = torch3d.io.load_ply(path_mesh)
    vertices = vertices.to(device)
    faces = faces.to(device)

    #### get tensor containing a point for each corner of each voxel
    ray_origins_all, shape_grid = get_ray_origins(vertices, res=res)
    print(f'The points to be segmented are organized in a grid of shape {shape_grid}. That is {shape_grid.numel()} points in total.')

    #### sort ray origins with respect their z-position
    mask_ray_origins_all_z_sorted = torch.argsort(ray_origins_all[:, 2])

    #### estimate suitable slab height
    # currently based on the most occuring z-length of all faces.
    # alternatives could be mean, median, max, ...
    z_lengths_faces = torch.max(vertices[faces][:, :, -1], dim=1)[0] - torch.min(vertices[faces][:, :, -1], dim=1)[0]
    z_length_most_occuring = torch.mode(z_lengths_faces)[0]

    n_z_slices_per_batch = int((z_length_most_occuring // res) / 1) #int(z_length_most_occuring // res)
    print('n_z_slices_per_batch: ', n_z_slices_per_batch)

    # number of points per batch
    n = shape_grid[0] * shape_grid[1] * n_z_slices_per_batch

    #### for progress bar (tqdm)
    total = int(np.ceil(len(mask_ray_origins_all_z_sorted) / n))

    #### initialize
    intersections = torch.zeros(ray_origins_all.shape).to(device)
    ray_directions_singles = torch.Tensor([[10., 9., 0.], [10., 9., 1], [10., 9., -1]]).to(device) #### TODO: Parameter?

    #### get slab-buffer specs
    theta_zs = get_angles_between_ray_directions_and_z(ray_directions_singles)
    buffer_lengths_z = get_buffer_lengths_z(theta_zs, shape_grid, res)

    #### voting policy
    # TODO: OPTIMIZE!
    # minimum number of rays per point to be classified as being inside the mesh
    # in order for that point to be considered as inside the mesh
    votes_min = np.ceil(len(ray_directions_singles)/2)

    #### do
    for idxs_oi in tqdm(batch(mask_ray_origins_all_z_sorted, n=n), total=total):

        ray_origins = ray_origins_all[idxs_oi, :]

        # get faces of interest
        faces_oi = get_faces_associated_with_given_ray_origins(faces, vertices,
                                                               ray_origins,
                                                               buffer_z_lower=min(buffer_lengths_z), ####
                                                               buffer_z_higher=max(buffer_lengths_z), ####
                                                               epsilon=0.0, ####
                                                              )

        print(f'\t Checking {len(faces_oi)}/{len(faces)} faces.')

        if len(faces_oi) == 0:
            continue

        # get ray origins within an xy-boundary box of the mesh.
        xmin_vertices = torch.min(vertices[faces_oi][:, :, 0])
        ymin_vertices = torch.min(vertices[faces_oi][:, :, 1])
        xmax_vertices = torch.max(vertices[faces_oi][:, :, 0])
        ymax_vertices = torch.max(vertices[faces_oi][:, :, 1])

        mask_mesh_bbox = (ray_origins[:, 0] >= xmin_vertices) * (ray_origins[:, 0] <= xmax_vertices) * \
                         (ray_origins[:, 1] >= ymin_vertices) * (ray_origins[:, 1] <= ymax_vertices)

        # how many ray_origins are ignored
        print(f'\t Ignoring {np.round(1 - torch.sum(mask_mesh_bbox).cpu().numpy()/len(mask_mesh_bbox), 2)} ray_origins in this slab.')

        ray_origins = ray_origins[mask_mesh_bbox, :]

        # repeat for linalg stuff
        ray_origins = ray_origins.repeat(len(ray_directions_singles), 1)

        # get ray direction
        # TODO: might want to use two rays that are perpendicular to each other. both in the plane.
        ray_directions = get_ray_directions(ray_directions_singles,
                                            n_ray_origins=ray_origins.shape[0]//len(ray_directions_singles),
                                            device=device)

        # go through all faces_oi
        for face in faces_oi:

            # get the corresponding vertices
            triangle = vertices[face].double()#.T

            # get votes of intersections
            votes = check_ray_triangle_intersection(ray_origins, ray_directions, triangle).double()

            votes = votes.reshape(len(ray_directions_singles), ray_origins.shape[0]//len(ray_directions_singles)).T #stupid work around becaused reshape ordering cannot be defined (as in np.reshape).

            intersections[idxs_oi[mask_mesh_bbox], :] += votes

    del votes
    torch.cuda.empty_cache()

    #### get boolean mask of whether points are inside or outside mesh
    mask_voxel_corners_inside = torch.sum((intersections % 2 == 1), dim=1) >= votes_min

    #### reshape
    mask_voxel_corners_inside = mask_voxel_corners_inside.reshape(shape_grid)

    #### return
    if inspection_mode:

        #reshape
        intersections = intersections.reshape(shape_grid + (len(ray_directions_singles),))
        intersections = [intersections[:, :, :, i] for i in range(intersections.shape[-1])]

        return mask_voxel_corners_inside, intersections
    else:
        return mask_voxel_corners_inside
