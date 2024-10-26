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

        print( "Looking for dataset folders in: ", self.data_dir )

        # Look for the correct subdirectory to load data from
        for subdir in subdirs:
            subdir_path = os.path.join( self.data_dir, subdir )
            if os.path.exists( subdir_path ):
                found_dir = subdir_path
                break
        if not found_dir:
            raise FileNotFoundError(f"None of the directories 'train', 'val', or 'test' were found in {self.data_dir}")
        
        # Load images from the found directory (train, val, or test)
        print(f"Found directory: {found_dir}")
        for image_file in sorted( os.listdir( found_dir )):
            image_path = os.path.join( found_dir, image_file )
            print( f"Image path: {image_path}") # Confirming each image path

            self.image_paths.append( image_path )
        
            pose_file = image_file.replace('.png', '.npy' )
            pose_path = os.path.join( self.data_dir, 'poses', pose_file )
            print(f"Pose path: {pose_path}") # Confirming each pose path

            if os.path.exists( pose_path ):

                pose = np.load( pose_path )
                self.poses.append( pose ) 
                
            else:
                print( f"Pose file missing for {image_file}")
                self.poses.append( None ) # Handle missing pose files

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image at the specified index
        image_path = self.image_paths[ idx ]
        image = Image.open( image_path )
        # Convert to numpy array  and normalize [0,1] range
        image = np.array( image ).astype( np.float32 ) / 255.0

        # Get the corresponding pose (if available )
        pose = self.poses[ idx ]

        return image, pose 
