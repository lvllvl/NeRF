import numpy as np

# Volume Rendering
def generate_rays( height, width, focal, c2w ):
    
    """
    Generate rays from camera to world.

    Args:

        - height: Image height in pixels, scalar
        - width: Image width in pixels, scalar
        - focal: Focal length in pixels, scalar
        - c2w: Camera-to-world transform, shape (4, 4), a homogeneous matrix

    Returns:
        - rays_o: (H, W, 3) Ray origins in world space 
        - rays_d: (H, W, 3) Ray directions in world sapce (normalized)
    
    Random notes:
        - c2w[ :3, :3 ] --> rotation matrix ( shape (3,3) )
        - c2w[ :3, 3 ] --> translation vector ( shape (3,) )
    """

    # Create a grid of pixel coordinates 
    i, j = np.meshgrid( np.arange( width ), np.arange( height ), indexing='xy' )

    # Convert image space --> camera space 
    x_cam = (i - width/2) / focal 
    y_cam = -(j - height/2) / focal 
    z_cam = -np.ones_like(i)

    # Stack into direciton vector
    negative_one = -1 # standard convention: camera looks down -Z axis in camera space
    dirs = np.stack( [x_cam, y_cam, z_cam], negative_one)  # shape (H, W, 3)

    # Rotate ray directions from camera to world
    rays_d = dirs @ c2w[ :3, :3 ].T # Rotate ray directions, this is where the ray starts!
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)  # normalize

    # Ray origins are the same for all pixels  - the camera center
    rays_o = np.broadcast_to( c2w[:3, 3], rays_d.shape )

    return rays_o, rays_d # Bothh shapes should be (H, W, 3)