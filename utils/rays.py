import numpy as np

def generate_rays(camera_pose, image_size):
    """
    Generate rays from a given camera pose and image size.

    Parameters:
    -----------
    camera_pose: np.ndarray
        The 4x4 camera transformation matrix (camera's position and orientation in the world).
    image_size: tuple
        The size of the output image (height, width).

    Returns:
    --------
    rays_o: np.ndarray
        Ray origins (the camera position for each ray).
    rays_d: np.ndarray
        Ray directions (the direction of each ray passing through a pixel).
    """
    H, W = image_size  # Unpack the image size
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    # Normalize pixel coordinates (i, j) to [-1, 1] space (screen space)
    directions = np.stack([(i - W * 0.5) / W, -(j - H * 0.5) / H, -np.ones_like(i)], axis=-1)

    # Apply the camera rotation to ray directions
    rays_d = directions @ camera_pose[:3, :3].T  # Rotate ray directions into world space

    # The origin of all rays is the camera's position, which is the translation component of the camera pose
    rays_o = np.broadcast_to(camera_pose[:3, 3], rays_d.shape)

    return rays_o, rays_d
