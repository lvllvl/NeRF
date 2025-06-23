import numpy as np
from nerf.render import generate_rays

# validate generate_rays() function
# what cases should we expect? include edge cases. 
# if we 

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