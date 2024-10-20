import os
import shutil

def organize_data(image_sets_dir, output_dir):
    """
    Organize data from image_sets directory into depth, images, and poses subdirectories.
    
    Parameters:
    -----------
    image_sets_dir : str
        Path to the downloaded image_sets directory.
    output_dir : str
        Path to the output data directory (with subfolders for depth, images, and poses).
    """
    
    # Ensure output subdirectories exist
    depth_dir = os.path.join(output_dir, 'depth')
    images_dir = os.path.join(output_dir, 'images')
    poses_dir = os.path.join(output_dir, 'poses')

    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(poses_dir, exist_ok=True)
    
    # Example logic for moving or copying files
    for filename in os.listdir(image_sets_dir):
        if filename.endswith(".png"):  # Assuming images are PNG files
            shutil.move(os.path.join(image_sets_dir, filename), os.path.join(images_dir, filename))
        elif filename.endswith(".txt"):  # Assuming poses are in TXT format
            shutil.move(os.path.join(image_sets_dir, filename), os.path.join(poses_dir, filename))
        # Handle depth maps if they exist (you can adjust file extensions accordingly)
        elif filename.endswith("_depth.png"):
            shutil.move(os.path.join(image_sets_dir, filename), os.path.join(depth_dir, filename))

    print(f"Data has been organized into {output_dir}.")
