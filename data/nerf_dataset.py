import os
import numpy as np

class NeRFDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images = []
        self.poses = []
        self.load_data()

    def load_data(self):
        images_dir = os.path.join(self.data_dir, 'images')
        poses_dir = os.path.join(self.data_dir, 'poses')

        # Load images
        for image_file in sorted(os.listdir(images_dir)):
            image = np.load(os.path.join(images_dir, image_file))  # Assuming images are saved as .npy files
            self.images.append(image)

        # Load poses
        for pose_file in sorted(os.listdir(poses_dir)):
            pose = np.load(os.path.join(poses_dir, pose_file))  # Assuming poses are saved as .npy files
            self.poses.append(pose)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.poses[idx]
