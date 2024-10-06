import matplotlib.pyplot as plt
import numpy as np
import imageio

def display_image(image, title=None):
    """Display an RGB image using matplotlib."""
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')  # Remove axis for clarity
    plt.show()

def save_image(image, filepath):
    """Save an image to disk using imageio."""
    image = np.clip(image, 0, 1)  # Ensure values are in the range [0, 1]
    image = (image * 255).astype(np.uint8)  # Convert to 8-bit image
    imageio.imwrite(filepath, image)

def display_depth_map(depth_map, title=None, cmap='plasma'):
    """Display a depth map using matplotlib."""
    plt.imshow(depth_map, cmap=cmap)
    if title:
        plt.title(title)
    plt.colorbar(label='Depth')
    plt.axis('off')
    plt.show()

def save_depth_map(depth_map, filepath, cmap='plasma'):
    """Save a depth map to disk as an image."""
    plt.imsave(filepath, depth_map, cmap=cmap)

def compare_images(image1, image2, title1="Ground Truth", title2="Prediction"):
    """
    Display two images side by side for comparison.

    Parameters:
    -----------
    image1: np.ndarray
        The first image (e.g., ground truth).
    image2: np.ndarray
        The second image (e.g., prediction).
    title1: str
        Title for the first image.
    title2: str
        Title for the second image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image1)
    axes[0].set_title(title1)
    axes[0].axis('off')

    axes[1].imshow(image2)
    axes[1].set_title(title2)
    axes[1].axis('off')

    plt.show()

def save_comparison(image1, image2, filepath, title1="Ground Truth", title2="Prediction"):
    """
    Save a side-by-side comparison of two images as a single image file.

    Parameters:
    -----------
    image1: np.ndarray
        The first image (e.g., ground truth).
    image2: np.ndarray
        The second image (e.g., prediction).
    filepath: str
        Path where the comparison image will be saved.
    title1: str
        Title for the first image.
    title2: str
        Title for the second image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image1)
    axes[0].set_title(title1)
    axes[0].axis('off')

    axes[1].imshow(image2)
    axes[1].set_title(title2)
    axes[1].axis('off')

    plt.savefig(filepath)
    plt.close()
