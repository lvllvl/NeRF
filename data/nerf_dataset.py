import os
import numpy as np
from PIL import Image
import torch

class NeRFDataset:
    def __init__(self, data_dir, load_without_pose=False):
        self.data_dir = data_dir
        self.image_paths = []
        self.poses = [] 
        self.load_data(load_without_pose)

    def load_data(self, load_without_pose):
        # Check for 'train', 'val', or 'test' directories
        subdirs = ['train', 'val', 'test']
        found_dir = None
        print("Looking for dataset folders in:", self.data_dir)

        # Locate subdirectory
        for subdir in subdirs:
            subdir_path = os.path.join(self.data_dir, subdir)
            if os.path.exists(subdir_path):
                found_dir = subdir_path
                break
        if not found_dir:
            raise FileNotFoundError(f"No 'train', 'val', or 'test' directories found in {self.data_dir}")
        
        # Load images and poses
        print(f"Found directory: {found_dir}")
        for image_file in sorted(os.listdir(found_dir)):
            image_path = os.path.join(found_dir, image_file)
            pose_file = image_file.replace('.png', '.npy')
            pose_path = os.path.join(self.data_dir, 'poses', pose_file)
            
            if os.path.exists(pose_path):
                self.image_paths.append(image_path)
                self.poses.append(np.load(pose_path))
            elif load_without_pose:
                print(f"Pose file missing for {image_file}, loading image only.")
                self.image_paths.append(image_path)
                self.poses.append(None)
            else:
                print(f"Pose file missing for {image_file}, skipping.")

        print(f"Loaded {len(self.image_paths)} image-pose pairs.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # Convert to (C, H, W) for PyTorch

        pose = self.poses[idx]
        if pose is not None:
            pose = torch.from_numpy(pose).float()

        return image, pose
