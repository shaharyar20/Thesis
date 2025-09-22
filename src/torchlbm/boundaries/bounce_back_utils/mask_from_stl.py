import torch
import trimesh
import numpy as np

def generate_bounce_back_mask_from_stl(stl_path: str, meshgrid: torch.Tensor, num_halo_cells: int) -> torch.Tensor:
    """
    Generate a binary mask (0 = fluid, 1 = solid) for a structured grid.

    Args:
        stl_path: Path to STL file.
        cell_centers: torch tensor of shape (N, 3), coordinates of cell centers.

    Returns:
        mask: torch tensor of shape (N,), 0 = fluid, 1 = solid
    """
    # Load STL mesh
    mesh = trimesh.load_mesh(stl_path)
    if not mesh.is_watertight:
        print("Warning: STL mesh is not watertight, point containment may fail.")
    
    mins, maxs = mesh.bounds
    print(mins, maxs)
    mesh.apply_translation([-mins[0], -mins[1]+0.1, 0.0])

    new_mins, new_maxs = mesh.bounds
    print(new_mins, new_maxs)
    # print(a)

    mask = torch.ones_like(meshgrid[0])

    X = meshgrid[0][num_halo_cells:-num_halo_cells, num_halo_cells:-num_halo_cells, :]
    Y = meshgrid[1][num_halo_cells:-num_halo_cells, num_halo_cells:-num_halo_cells, :]
    Z = meshgrid[2][num_halo_cells:-num_halo_cells, num_halo_cells:-num_halo_cells, :]
    points = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=-1).numpy()  # shape (N, 3)
    # Convert to numpy
    # points = cell_centers.numpy()  # shape (N, 3)
    print(points.shape)
    # print(a)

    # trimesh.contains returns a boolean array (True if inside)
    inside = mesh.contains(points)   # shape (N,)

    # Inside → fluid (0), Outside → solid (1)
    mask[num_halo_cells:-num_halo_cells, num_halo_cells:-num_halo_cells, :] = torch.from_numpy(~inside).long().reshape(X.shape)
    
    mask[0:num_halo_cells, :, :] = mask[num_halo_cells, :, :].unsqueeze(0)
    mask[-num_halo_cells:, :, :] = mask[-num_halo_cells-1, :, :].unsqueeze(0)
    mask[:, 0:num_halo_cells, :] = mask[:, num_halo_cells, :].unsqueeze(1)
    mask[:, -num_halo_cells:, :] = mask[:, -num_halo_cells-1, :].unsqueeze(1)
    print(mask.shape)
    print(mask.sum().item(), mask.numel())
    # print(a)

    return mask, mesh
