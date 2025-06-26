import torch
from typing import Tuple

def sample_points_along_rays( ray_origins, ray_directions, near, far, num_samples ):
    """
    Sample points uniformly along the rays between near and far bounds.

    Parameters:
    --------

    ray_origins: torch.Tensor
        - Origins of the rays.

    ray_directions: torch.Tensor
        - Directions of the rays.

    near: float
        - Near bound for ray sampling.
    
    far: float
        - Far bound for ray sampling.
    
    num_samples: int
        - Number of samples to take along each ray.

    Returns:
    --------
    samples: torch.Tensor
        - The sampled points along the rays.
    """

    t_vals = torch.linspace( near, far, num_samples )

    sample_points = ray_origins.unsqueeze( 1 ) + t_vals( -1 ) * ray_directions.unsqueeze( 1 )

    return sample_points

def stratified_samples(
    ray_d: torch.Tensor,
    num_samples: int,
    near: float,
    far: float,
    rand: bool = True, 
) -> torch.Tensor:
    """
    Uniform (stratefied) sampling along rays.

    Parameters
    ----------

    ray_d       : (N_rays, 3) torch.Tensor
                Only needed for device / dtype; directions themselves aare not used here.
    num_samples   : int
                Numbers of strata / coarse samples.
    near, far   : float
                Depth bounds along the ray (in world units).
    rand        : bool, default = True
                - True -> random point in each stratum (original NeRF)
                - False -> center of each stratum (deterministic).

    Returns
    -------
    t_vals      : (N_rays, n_samples) torch.Tensor
                Sample depths along each ray, ascending.
    """

    device = ray_d.device

    # Stratum edges: shape (n_samples + 1)
    t_edges = torch.linspace( near, far, steps=num_samples + 1, device= device )

    # Random offset inside each bin
    if rand:
        # Uniform noise in [0,1) for each ray and each bin
        noise = torch.rand( (*ray_d.shape[:-1], num_samples), device= device )
    else:
        noise = 0.5 # Center of the bin
    
    # Interpolate inside each stratum
    #   t = t_left + u * (t_right - t_left)
    t_left = t_edges[ :-1 ]     # (num_samples,)
    t_right = t_edges[ 1: ]     # (num_samples,)

    # Reshape to broadcast over rays
    t_left, t_right = [ x.expand( *ray_d.shape[ :-1 ], num_samples ) for x in (t_left, t_right ) ]

    t_vals = t_left + noise * ( t_right - t_left ) # (N_rays, num_samples )
    return t_vals
