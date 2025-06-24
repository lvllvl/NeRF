import torch

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


    # TODO: stratified sampling 
    # TODO: PDF sampling utitlities ?
    