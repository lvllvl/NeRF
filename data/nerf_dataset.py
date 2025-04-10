import os
import numpy as np
from PIL import Image
import torch

class NeRFDataset:
    def __init__(self, image_dir, pose_dir, load_without_pose=False):

        # Define where Poses and IMages are stored
        self.image_dir = image_dir
        self.pose_dir = pose_dir
        
        self.image_paths = []
        self.poses = [] 
        self.load_data( load_without_pose )

    def load_data(self, load_without_pose):
        print( "Loading data from:", self.image_dir )
        print( "Loading data from:", self.pose_dir )

        if not os.path.exists( self.image_dir ):
            raise FileNotFoundError( f"Image directory not found: {self.image_dir}" )
        if not os.path.exists( self.pose_dir ):
            print(f"Warning: Pose directory not found: {self.pose_dir}. All poses will be default (identity)")

        # Load images and poses
        for image_file in sorted(os.listdir( self.image_dir )):
            image_path = os.path.join(self.image_dir, image_file)
            
            # print( "image_file type:", type( image_file ), image_file )
            # Use os.path.splitext to work with any extension 
            base_name = os.path.splitext( str(image_file) )[0]
            # print( "base_name type:", type( base_name ), base_name )
            
            pose_file = base_name + '.npy'
            pose_path = os.path.join( self.pose_dir, pose_file )

            self.image_paths.append( image_path )
             
            if os.path.exists(pose_path):
                self.poses.append(np.load(pose_path))
            elif load_without_pose:
                print(f"Pose file missing for {image_file}, loading image only.")
                self.poses.append( None ) 
            else:
                print(f"Pose file missing for {image_file}, skipping.")
        
        print(f"Loaded {len(self.image_paths)} image-pose pairs.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Open image, convert to RGB
        image = Image.open(image_path).convert('RGB')

        # Resize image to a fixed size, e.g., 256x256
        target_size = (256, 256) # width, height
        image = image.resize( target_size, Image.BILINEAR )

        # Convert to numpy array and normalize to [0, 1]
        image = np.array(image).astype(np.float32) / 255.0

        # convert to torch tensor and re-arrange dimensions to (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)

        pose = self.poses[idx]
        if pose is not None:
            pose = torch.from_numpy(pose).float()
        else:
            # Provide a default pose or placeholder (e.g., identity matrix)
            pose = torch.eye( 4 ) 
        
        return image, pose
