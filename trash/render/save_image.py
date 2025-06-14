from PIL import Image
import numpy as np

def save_image( rgb_array, file_path ):
    """
    Save an RGB array as an image.

    Parameters:
    --------
    rgb_array: np.ndarray
        - The RBG values as an array (shape: H x W x 3)
    file_path: str
        - The path where the image will be saved.
    """

    rgb_image = (rgb_array * 255).astype( np.uint8 ) # Convert to 8-bit format
    image = Image.fromarray( rgb_image )
    image.save( file_path )