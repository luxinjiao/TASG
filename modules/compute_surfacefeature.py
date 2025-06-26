import torch
import numpy as np

def compute_normal(triangle_points, offsets, random_flip=False, is_grouped=False):
    """
    Calculate unit normal vectors for triangles defined by three points.

    Args:
        triangle_points (Tensor): Shape [..., 3, 3], where the last dimension represents triangle vertices.
        offsets (Tensor): Batch offset tensor.
        random_flip (bool): Apply random inversion of normals per batch.
        is_grouped (bool): Whether input tensor is grouped, affects broadcasting.

    Returns:
        Tensor: Unit normal vectors, oriented and optionally randomly flipped.
    """
    vec1 = triangle_points[..., 1, :] - triangle_points[..., 0, :]
    vec2 = triangle_points[..., 2, :] - triangle_points[..., 0, :]
    normals = torch.cross(vec1, vec2, dim=-1)
    normals = normals / torch.norm(normals, dim=-1, keepdim=True)

    if is_grouped:
        sign = (normals[..., 0:1, 0] > 0).float() * 2. - 1.
    else:
        sign = (normals[..., 0] > 0).float() * 2. - 1.
    normals *= sign.unsqueeze(-1)

    if random_flip:
        num_batches = offsets.shape[0]
        boundaries = [0] + list(offsets.cpu().numpy())
        flip_choices = (np.random.rand(num_batches) < 0.5)
        flip_mask = [torch.full((boundaries[i+1] - boundaries[i], 1), 1.0 if flip_choices[i] else -1.0) 
                      for i in range(num_batches)]
        flip_mask = torch.cat(flip_mask, dim=0).to(normals.device)

        normals *= flip_mask.unsqueeze(-1) if is_grouped else flip_mask

    return normals

def compute_centroids(triangle_points):
    return triangle_points.mean(dim=-2)

def compute_distances(triangle_points):
    p1, p2 = triangle_points[..., 1, :], triangle_points[..., 2, :]
    d1 = torch.norm(p1, dim=-1, keepdim=True)
    d2 = torch.norm(p2, dim=-1, keepdim=True)
    return torch.cat([d1, d2], dim=-1)

def compute_areas(triangle_points):
    p1, p2 = triangle_points[..., 1, :], triangle_points[..., 2, :]
    cross_product = torch.cross(p1, p2, dim=-1)
    return 0.5 * torch.norm(cross_product, dim=-1, keepdim=True)

def compute_inner_product(normals, centers, normalize=True):
    constants = torch.sum(normals * centers, dim=-1, keepdim=True)
    if normalize:
        constants /= torch.sqrt(torch.tensor(3.0, device=normals.device))
    return constants

def sanitize_nan(normals, centers, positions=None):
    N = normals.shape[0]
    nan_mask = torch.isnan(normals).any(dim=-1)
    first_valid = torch.argmax((~nan_mask).int(), dim=-1)

    valid_normals = normals[None, first_valid].expand(N, -1)
    normals[nan_mask] = valid_normals[nan_mask]

    valid_centers = centers[None, first_valid].expand(N, -1)
    centers[nan_mask] = valid_centers[nan_mask]

    if positions is not None:
        valid_positions = positions[None, first_valid].expand(N, -1)
        positions[nan_mask] = valid_positions[nan_mask]
        return normals, centers, positions
    return normals, centers

def sanitize_nan_grouped(normals, centers, positions=None, fallback_normal=None):
    N, G, _ = normals.shape
    if fallback_normal is None:
        fallback_normal = torch.tensor([0.0, 0.0, 1.0], device=normals.device).view(1, 1, 3)

    nan_mask = torch.isnan(normals).any(dim=-1)
    has_valid = ~nan_mask.all(dim=-1)
    first_valid = torch.argmax((~nan_mask).int(), dim=-1)

    normal_fill = torch.where(
        has_valid.view(N, 1, 1),
        normals[torch.arange(N), None, first_valid].expand(-1, G, -1),
        fallback_normal.expand(N, G, -1)
    )
    normals[nan_mask] = normal_fill[nan_mask]

    center_fill = torch.where(
        has_valid.view(N, 1, 1),
        centers[torch.arange(N), None, first_valid].expand(-1, G, -1),
        torch.zeros_like(centers)
    )
    centers[nan_mask] = center_fill[nan_mask]

    if positions is not None:
        position_fill = torch.where(
            has_valid.view(N, 1, 1),
            positions[torch.arange(N), None, first_valid].expand(-1, G, -1),
            torch.zeros_like(positions)
        )
        positions[nan_mask] = position_fill[nan_mask]
        return normals, centers, positions

    return normals, centers
