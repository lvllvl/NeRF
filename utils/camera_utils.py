import numpy as np

def get_camera_matrix( fov, aspect_ratio, near, far ):
    """
    Generate a camera projection matrix.

    Parameters:
    --------
    fov: float
        - Field of view of the camera in degrees.
    aspect_ratio: float
        - Aspect ratio of the image ( width / height ).
    near: float
        - Near clipping plane.
    far: float
        - Far clipping plane.

    Returns:
    --------
    np.ndarray:
        - The camera projection matrix.
    """
    f = 1.0 / np.tan( np.radians( fov ) / 2.0 )

    return np.array( [[f / aspect_ratio, 0, 0, 0 ],
                      [0, f, 0, 0],
                      [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far) ],
                      [0, 0, -1, 0]
                      ])