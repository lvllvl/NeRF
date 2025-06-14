import numpy as np

def generate_rays( camera_pose, image_size ):
    """
    Generate rays from a given camera pose and image size.

    Parameters:
    --------

    camera_pose: np.ndarray
        - The 4x4 camera transformation matrix.

    image_size: tuple
        - The size of the output image (width, height).
    
    Returns:
    --------
    rays_o: np.ndarray
        - Ray origins
    rays_d: np.ndarray
        - Ray directions.
    """
    H, W = image_size # unpack this tuple
    
    # create a grid of pixel coordinates for the image.
    # i, j represnt the pixel indices in the x (horizontal) and y (vertical) dimensions
    # this grid is used to generate directions for each pixel, bc we need to cast a ray through EACH pixel to understand the image
    # indexing='xy' is to make sure that i(horizonatal), j (vertical) coordinates are aligned with image coordinates
    i, j = np.meshgrid( np.arange(W), np.arange(H), indexing='xy' ) 

    # Computation
    # computing the normallized direction from the camera for each point
    # (i-W/2) / 2: normalize x-coordinate (horizontal offset) for each pixel
    # -(j - H / 2) / H: normalize y-coordinate (vertical offset) for each pixel and flips it (so positive y points UP)
    # -np.ones_lik(i): adds a constant z-coordinate of -1 to all directions. This assumes camera is pointing in negative z-direction ( a common convention ), the camera is pointing forward
    # purpose: this gives us a direction for each pixel in 3d space, pointing away from the camera  
    directions = np.stack( [(i - W / 2) / W, -(j - H / 2) / H, -np.ones_like(i)], -1 )

    rays_d = directions @ camera_pose[ :3, :3 ].T
    rays_o = np.broadcast_to( camera_pose[ :3, 3 ], rays_d.shape )

    return rays_o, rays_d


