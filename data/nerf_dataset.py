import os
import numpy as np
from PIL import Image

class NeRFDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = []
        self.poses = [] 
        self.load_data()

    def load_data(self):
        # check if 'train', 'val', or 'test' directories exist
        subdirs = ['train', 'val', 'test' ]
        found_dir = None

        # Look for the correct subdirectory to load data from
        for subdir in subdirs:
            subdir_path = os.path.join( self.data_dir, subdir )
            if os.path.exists( subdir_path ):
                found_dir = subdir_path
                break
        if not found_dir:
            raise FileNotFoundError(f"None of the directories 'train', 'val', or 'test' were found in {self.data_dir}")
        
        # Load images from the found directory (train, val, or test)

        for image_file in sorted( os.listdir( found_dir )):
            image_path = os.path.join( found_dir, image_file )
            self.image_paths.append( image_path )
        
            pose_file = image_file.replace('.png', '.npy' )
            pose_path = os.path.join( self.data_dir, 'poses', pose_file )

            if os.path.exists( pose_path ):
                pose = np.load( pose_path )
                self.poses.append( pose )
            else:
                self.poses.append( None ) # Handle missing pose files

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image at the specified index
        image_path = self.image_paths[ idx ]
        image = Image.open( image_path )
        image = np.array( image ) # convert to numpy array, if needed

        # Get the corresponding pose (if available )
        pose = self.poses[ idx ]

        return image, pose 
