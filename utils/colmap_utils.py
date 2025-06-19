import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat # impport from COLMAP file
import numpy as np
from PIL import Image
import json
import os

def list_colmap_images(path):
    '''
    List all images registered in COLMAP's reconstruction.
    '''
    images_path = os.path.join( path, 'sparse/0/images.bin' )
    images = read_images_binary( images_path )
    return [ img.name for img in images.values() ]

def get_image_info( path, image_name ):
    '''
    Return camera pose and intrinsics for a given image.
    '''
    images = read_images_binary( os.path.join( path, 'sparse/0/images.bin' )) 
    cameras = read_cameras_binary( os.path.join( path, 'sparse/0/cameras.bin' ))

    for img in images.values():

        if img.name == image_name:
            cam = cameras[ img.camera_id ]
            return {
                "image_id": img.image_id,
                "qvec": img.qvec, # Rotation quaternion
                "tvec": img.tvec, # translation
                "intrinsics": {
                    "model": cam.model,
                    "params": cam.params.tolist(),
                    "width": cam.width,
                    "height": cam.height
                }
            }
    raise ValueError(f"Image {image_name} not found.")

def print_colmap_metadata_summary(path):
    '''
    Print all camera IDs, image names, intrinsics, poses
    '''
    images = read_images_binary( os.path.join( path, 'sparse/0/images.bin' )) 
    cameras = read_cameras_binary( os.path.join( path, 'sparse/0/cameras.bin' ))

    for img in images.values():
        cam = cameras[ img.camera_id ]
        print( f"Image: {img.name}" )
        print( f" -> ID: {img.id}" )
        print( f" -> qvec: {img.qvec}" )
        print( f" -> tvec: {img.tvec}" )
        print( f" -> Intrinsics: {cam.model} {cam.params}" )
        print()

def visualize_camera_poses_3d(path):
    '''
    Visualize the camera poses
    '''
    images = read_images_binary( os.path.join( path, 'sparse/0/images.bin') )

    fig = plt.figure( figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    for img in images.values():
        R = qvec2rotmat(img.qvec)  # (3, 3)
        t = img.tvec.reshape(3, 1) # (3, 1)

        # COLMAP world-to-camera, so invert to get camera-to-world:
        C = -R.T @ t  # camera center in world coords
        direction = R.T @ np.array([[0, 0, 1]]).T  # camera looks down +z

        ax.scatter(C[0], C[1], C[2], c='blue', marker='o')
        ax.quiver(
            C[0], C[1], C[2],
            direction[0], direction[1], direction[2],
            length=0.2, color='red'
        )

        ax.text(C[0], C[1], C[2], img.name, fontsize=8)

    ax.set_title("Camera Poses (COLMAP)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=10, azim=-90)
    plt.show()


def colmap_to_transforms_json(path, image_dir="images", output_file="transforms.json"):
    '''
    Read binary files, decode usings COLMAP's data spec
    Map files to JSON format, with camera poses, focal length, image paths, etc. 
    '''
    images = read_images_binary(os.path.join(path, 'sparse/0/images.bin'))
    cameras = read_cameras_binary(os.path.join(path, 'sparse/0/cameras.bin'))
    
    cam = next(iter(cameras.values()))
    fx = cam.params[0]
    h = cam.height
    w = cam.width
    camera_angle_x = 2 * np.arctan(w / (2 * fx))

    frames = []
    for img in images.values():
        R = qvec2rotmat(img.qvec)
        t = img.tvec.reshape(3, 1)
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3:] = -R.T @ t
    
        # Validation: Verify that COLMAP process was accurate
        img_path = os.path.join( "data", image_dir, img.name ) # pull image from disk
        print("Verifying image path:", img_path)
        
        with Image.open( img_path ) as im:
            img_w, img_h = im.size # grab data from raw image
        if (img_w != cam.width) or (img_h != cam.height): # check for any discrepancies between COLMAP info and raw disk image info
            break
            print(f"Warning: Disk image {img.name} size ({img_w}, {img_h}) does not match COLMAP size ({cam.width}, {cam.height})")

        frame = {
            "file_path": f"{image_dir}/{img.name}",
            "transform_matrix": c2w.tolist()
        }
        frames.append(frame)

    transforms = {
        "camera_angle_x": camera_angle_x,
        "height": h,
        "width": w,
        "frames": frames
    }

    with open(os.path.join(path, output_file), 'w') as f:
        json.dump(transforms, f, indent=4)

    print(f"✅ transforms.json saved to {output_file}")
