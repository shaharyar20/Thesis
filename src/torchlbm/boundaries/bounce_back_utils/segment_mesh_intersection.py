import numpy as np
import trimesh

from torchlbm.exceptions import TorchlbmError


def segment_mesh_intersection(mesh: trimesh.Trimesh, p0, p1, eps=1e-9):
    """
    Compute intersection point(s) of a segment [p0, p1] with an STL mesh.

    Args:
        mesh: trimesh.Trimesh object
        p0, p1: (3,) numpy arrays, endpoints of the segment
        eps: tolerance for floating-point comparisons

    Returns:
        intersections: list of intersection points (each (3,))
    """
    # print(p0, p1)
    # Direction vector
    direction = p1 - p0
    length = np.linalg.norm(direction)
    if length < eps:
        return []
    direction /= length
    # print(direction, length)

    # Cast ray
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=[p0],
        ray_directions=[direction]
    )
    # print("locations:", locations)
    # print(a)

    intersections = []
    for loc in locations:
        # Distance along the segment
        t = np.dot(loc - p0, direction)
        # print("loc:", loc)
        # print("t:", t)
        if -eps <= t <= length + eps:
            intersections.append(t/length)
    # print("intersections:", intersections)
    # print(a)
    if len(intersections) > 1:
        raise TorchlbmError("Multiple intersections found, which is not supported yet.")
    

    return intersections[0]
