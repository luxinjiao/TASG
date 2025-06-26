import torch
import torch.nn as nn
from modules.polar_utils import xyz2sphere
from modules.compute_surfacefeature import compute_inner_product, compute_normal, compute_centroids, sanitize_nan_grouped,compute_distances,compute_areas
from modules.pointops.functions import pointops

def resort(points, idx):

    device = points.device
    N, G, _ = points.shape
    batch_indices = torch.arange(N, device=device).view(N, 1).repeat(1, G)
    return points[batch_indices, idx]


def _fixed_rotate(xyz):

    rot_mat = torch.tensor(
        [[0.5, -0.5, 0.7071],
         [0.7071, 0.7071, 0.],
         [-0.5, 0.5, 0.7071]],
        dtype=xyz.dtype, device=xyz.device
    )
    return xyz @ rot_mat


def group(xyz, new_xyz, offset, new_offset, k=3):

    group_idx, _ = pointops.knnquery(k, xyz, new_xyz, offset, new_offset)  # [M, K]
    group_xyz = xyz[group_idx.view(-1), :].view(new_xyz.shape[0], k, 3)     # [M, K, 3]
    group_xyz_norm = group_xyz - new_xyz.unsqueeze(1)                       # [M, K, 3]

    
    group_phi = xyz2sphere(_fixed_rotate(group_xyz_norm))[..., 2]          # [M, K]
    sort_idx = group_phi.argsort(dim=-1)                                   # [M, K]

    sorted_group_xyz = resort(group_xyz_norm, sort_idx).unsqueeze(-2)  # [M, K, 1, 3]
    sorted_group_xyz_roll = torch.roll(sorted_group_xyz, shifts=-1, dims=-3)  # [M, K, 1, 3]

    group_centroid = torch.zeros_like(sorted_group_xyz)                       # [M, K, 1, 3]
    final_group_xyz = torch.cat(
        [group_centroid, sorted_group_xyz, sorted_group_xyz_roll], dim=-2
    )  # [M, K, 3, 3]

    return final_group_xyz



class SFE(nn.Module):
    def __init__(self, k, in_channel, out_channel, random_inv=True):
        super(SFE, self).__init__()
        self.k = k
        self.random_inv = random_inv
        self.mlps = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 1, bias=True),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(True),
            nn.Conv1d(out_channel, out_channel, 1, bias=True),
        )
        self.sort_func = group
    def forward(self, center, offset):
        
        group_xyz = self.sort_func(center, center, offset, offset, k=self.k)  # [N, K-1, 3, 3]
        group_normal = compute_normal(group_xyz, offset, random_flip=self.random_inv, is_grouped=True)
        group_center = compute_centroids(group_xyz)
        group_polar = xyz2sphere(group_center)
        group_pos = compute_inner_product(group_normal, group_center)
        group_dist = compute_distances(group_xyz)  # [N, K-1, 2]
        group_area = compute_areas(group_xyz)      # [N, K-1, 1]
        group_normal, group_center, group_pos = sanitize_nan_grouped(group_normal, group_center, group_pos)
        new_feature = torch.cat([
            group_normal,      
            #group_polar,       
            group_pos,         
            group_center,     
            group_dist,        
            group_area         
        ], dim=-1)   
        new_feature = new_feature.transpose(1, 2).contiguous()  # [N, C, K-1]
        new_feature = self.mlps(new_feature)
        new_feature = torch.sum(new_feature, dim=2)             # [N, C]
    
        return new_feature
