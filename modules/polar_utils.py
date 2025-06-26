import torch
import numpy as np
def xyz2sphere(xyz: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Convert Cartesian coordinates to spherical coordinates.
    
    Args:
        xyz: Tensor of shape [..., 3], in (x, y, z) format.
        normalize: Whether to normalize theta and phi to [0, 1].

    Returns:
        Tensor of shape [..., 3]: (rho, theta, phi)
    """
    rho = torch.norm(xyz, dim=-1, keepdim=True)  # [..., 1]
    rho = torch.clamp(rho, min=1e-8)  # avoid division by zero

    z = xyz[..., 2:3]
    theta = torch.acos(torch.clamp(z / rho, -1.0, 1.0))  # polar angle: [0, pi]
    phi = torch.atan2(xyz[..., 1:2], xyz[..., 0:1])       # azimuthal angle: [-pi, pi]

    if normalize:
        theta = theta / np.pi             # [0, 1]
        phi = phi / (2 * np.pi) + 0.5     # [0, 1]

    return torch.cat([rho, theta, phi], dim=-1)


