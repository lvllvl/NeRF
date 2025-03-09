import numpy as np
import os

def quaternion_to_rotation_matrix(q):
    # q = [qw, qx, qy, qz]
    qw, qx, qy, qz = q
    # Normalize the quaternion
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R

def load_colmap_images_txt(file_path):
    poses = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Skip comments or empty lines
        if line.startswith("#") or len(line) == 0:
            i += 1
            continue

        parts = line.split()
        # Expecting at least 10 parts: IMAGE_ID, qw, qx, qy, qz, tx, ty, tz, CAMERA_ID, image_name
        if len(parts) < 10:
            i += 1
            continue

        # Extract the pose parameters
        image_id = parts[0]
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        image_name = parts[9]
        
        R = quaternion_to_rotation_matrix([qw, qx, qy, qz])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        poses[image_name] = T

        # Skip the next line which contains 2D-3D correspondences
        i += 2
    return poses

if __name__ == '__main__':
    colmap_images_txt = 'sparse/0_txt/images.txt'
    poses = load_colmap_images_txt(colmap_images_txt)
    output_dir = 'poses'
    os.makedirs(output_dir, exist_ok=True)
    for image_name, pose in poses.items():
        # Change extension to .npy
        pose_file = os.path.splitext(image_name)[0] + '.npy'
        np.save(os.path.join(output_dir, pose_file), pose)
