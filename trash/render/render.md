This should involve RAY TRACING and volume rendering. The rendering loop will: 
 - Generate rays from the camera into the scene
 - Integrate NeRF output (color and density) along each ray to determine the final color of each pixel.


## Potential additions

1. `render_video.py`
    - render animations or videos, render multiple frames and switch them into a video, e.g., .mp4

2. `depth_render.py`
    - Render depth maps along with RGB images
    - This file could handle extracting depth information (i.e., how far each point is from the camera along the ray).
    - depth maps are for vizualizing 3D strucutres

3. `ray_visualizations.py`
    - for debugging, this could be for visualizing the rays being cast into the scene
    - this helps see how rays are interacting with 3D space and verifying correctness

4. `config_render.py`
    - config file to specify rendering settings ( e.g., resolution, near and far bounds, number of samples ). This allows for easy tweaking of render parameters without changing the main code.

5. `render_utils.py`
    - Additional helper functions for common tasks, e.g., adjust camera parameters, calculate camera matrices, post-processing