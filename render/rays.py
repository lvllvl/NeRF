import numpy as np
import torch 

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
    camera_pose = camera_pose.squeeze()
    if camera_pose.shape[0] == 3 and camera_pose.shape[1] == 4:
        R = camera_pose[:, :3] # Rotation matrix: (3,3)
        t = camera_pose[:, 3] # Translation vector: (3,)
    elif camera_pose.shape[0] == 4 and camera_pose.shape[1] == 4:
        # otherwise, assume it's a full 4x4 matrix
        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]
    else:
        raise ValueError("Unexpected shape for camera_pose: {}".format(camera_pose.shape))
    

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
    # Create the normalized directions with fixed z = -1.
    directions = np.stack([(i - W / 2) / W, -(j - H / 2) / H, -np.ones_like(i)], axis=-1)
    directions = torch.from_numpy(directions).float().to(camera_pose.device)

    # mat mul for each pixel, rotate direction by camera's rotation
    # here camera_pose[:3, :3] is a 2D tensor of shape [3,3].
    rays_d = torch.matmul( directions, R.T )

    # Broadcast the camera translation to all pixels
    rays_o = t.expand_as( rays_d )

    return rays_o, rays_d