import torch
import numpy as np

def volume_render( ray_origins, ray_directions, model, near, far, num_samples = 64 ):

    """
    Perform volume rendering from NeRF model

    Parameters:
    ------------
    ray_origins: torch.Tensor
        The origins of the rays (the cameral positions).
    
    ray_directions: torch.Tensor
        The direction of the rays (one for each pixel ).
    
    model: nn.Module
        The NeRF model that predicts density and color.

    near: float
        The far bound of the ray.

    num_samples: int
        The number of sample points along each ray

    Returns:
    ------------
    rgb_map: torch.Tensor
        The rendered image as an RGB map.
    """

    # step 1: sample points along each ray
    t_vals = torch.linspace( near, far, num_samples )
    t_vals = t_vals.expand( [ray_origins.shape[0], num_samples]) # [ batch_size, num_samples ]

    # Compute the points along the rays
    sample_points = ray_origins.unsqueeze(1) + ray_directions.unsqueeze(1) * t_vals.unsqueezed(-1) # [batch_size, num_samples, 3]

    # step 2: Query the NeRF model to get color and density at each sample point
    sigma, rgb = model( sample_points ) # assuming model outputs both density and color

    # step 3: Compute transmittance along the rays
    delta = t_vals[ :, 1: ] - t_vals[ :, :-1 ] # Distance between adjacent points
    delta = torch.cat( [delta, torch.tensor([1e10]).expand( delta[:, :1].shape )], -1) # to handle the last point

    # compute transmittance T(t)
    sigma_delta = sigma * delta
    T = torch.exp( -torch.cumsum( sigma_delta, dim = -1 )) # cumulative product for transmittance

    # step 4: Compute the final color using volume rendering equation
    weights = T * ( 1 - torch.exp( -sigma_delta )) # Contribution of each point
    rgb_map = torch.sum( weights.unsqueeze( -1 ) * rgb, dim=1 ) # Final RGB color

    return rgb_map