import numpy as np
import torch
from nerf.render import generate_rays, volume_rendering, hierarchical_sampling

def test_generate_rays_basic():

    height = 4
    width = 6
    focal = 2.0

    # Identity camera pose, camera at origin (no rotation)
    c2w = np.eye(4)

    rays_o, rays_d = generate_rays( height, width, focal, c2w )

    # Check shapes
    assert rays_o.shape == (height, width, 3 )
    assert rays_d.shape == (height, width, 3 )

    # Ray origins should all be same ( broadcasted)
    unique_origins = np.unique( rays_o.reshape(-1, 3), axis=0 )
    assert unique_origins.shape[0] == 1
    np.testing.assert_allclose( unique_origins[0], np.array([0.0, 0.0, 0.0]))

    # Centered ray should point down -Z
    center_ray = rays_d[ height // 2, width // 2 ]
    expected_dir = np.array( [ 0.0, 0.0, -1.0 ] )
    dot = np.dot( center_ray, expected_dir )

    assert dot > 0.95 # nearly pointing in the same direction

    rays_o2, rays_d2 = generate_rays(height, width, focal, c2w)
    np.testing.assert_allclose(rays_o, rays_o2)
    np.testing.assert_allclose(rays_d, rays_d2)

def test_volume_rendering_shapes():
    N_rays, N_samples = 2, 5
    rgb   = torch.rand(N_rays, N_samples, 3)
    sigma = torch.rand(N_rays, N_samples)
    t     = torch.linspace(0.1, 1.0, steps=N_samples).expand(N_rays, N_samples)

    out = volume_rendering(rgb, sigma, t)
    assert out["rgb"].shape   == (N_rays, 3)
    assert out["depth"].shape == (N_rays,)
    assert out["weights"].shape == (N_rays, N_samples)
    assert torch.all(out["weights"] >= 0)

def test_hierarchical_sampling_basic():
    N_rays, N_bins, N_fine = 3, 4, 10
    bins = torch.linspace(0.0, 1.0, steps=N_bins).expand(N_rays, N_bins)
    # skewed weights → concentrate near last bin
    weights = torch.tensor([[0.1,0.1,0.1,0.7]]).expand(N_rays, N_bins)
    t_fine = hierarchical_sampling(bins, weights, N_fine, rand=False)
    assert t_fine.shape == (N_rays, N_fine)
    # All fine samples should lie within [0,1]
    assert torch.all(t_fine >= 0.0) and torch.all(t_fine <= 1.0)