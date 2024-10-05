from models.nerf_model import NeRF
from render.volume_render import volume_render
from utils.rays import generate_rays

def render_image( camera_pose, image_size, model, near, far, num_samples=64 ):

    """
    Render an image from a given camera pose using the trained NeRF model.

    Parameters:
    --------
    camera_pose: np.ndarray
        - The camera's position and orientation

    image_size: tuple
        - The size of the output image (width, height).

    model: NeRF
        - The trained NeRF model.

    near: float
        - The near bound: the min depth along the ray where sampling starts (igonores objects closer than this). 

    far: float
        - The far bound: the max depth along the ray where sampling stops (igonores objects further than this).

    num_samples: int
        - Number of sample points along each ray.

    Returns:
    --------
    rendered_image: np.ndarray
        - The final rendered image as an array of RGB values.
    """

    rays_o, rays_d = generate_rays( camera_pose, image_size ) # Generate rays for the given camera pose
    rgb_map = volume_render( 
        rays_o,
        rays_d,
        model,
        near,
        far,
        num_samples ) # returns rgb_map: torch.Tensor - The rendered image as an RGB map.

    return rgb_map