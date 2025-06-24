import os
import imageio
import numpy as np

def load_data(data_dir):
    """
    Load images and corresponding camera poses from the data directory.

    Parameters:
    -----------
    data_dir: str
        Path to the directory containing images and camera pose files.

    Returns:
    --------
    images: list of np.ndarray
        List of loaded image arrays.
    poses: list of np.ndarray
        List of 4x4 camera pose matrices corresponding to each image.
    """
    # Load images
    image_files = sorted(os.listdir(os.path.join(data_dir, 'images')))
    images = [imageio.imread(os.path.join(data_dir, 'images', f)) for f in image_files]

    # Load camera poses
    pose_files = sorted(os.listdir(os.path.join(data_dir, 'poses')))
    poses = [np.loadtxt(os.path.join(data_dir, 'poses', f)) for f in pose_files]
    
    return images, poses