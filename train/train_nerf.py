import torch
import numpy as np
import os
from models.nerf_model import NeRF
from utils.data_loader import load_data
from utils.rays import generate_rays
from utils.metrics import compute_psnr
from utils.visualization_utils import display_image, save_image, display_depth_map, save_depth_map, compare_images, save_comparison
from train.losses import mse_loss
from train_config import config
from scheduler import get_scheduler

# Initialize model, optimizer, and loss function using config values
model = NeRF(num_freqs=config['num_freqs'])
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
scheduler = get_scheduler(optimizer, config)
criterion = mse_loss

# Initialize model, optimizer, and loss function using config values
model = NeRF(num_freqs=config['num_freqs'])
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
scheduler = get_scheduler(optimizer, config)
criterion = mse_loss

# Create a checkpoint directory if it does not exist
if not os.path.exists( config['checkpoint_dir']):
    os.makedirs['checkpoint_dir']

def train_nerf(data_dir, epochs=config['epochs']):
    """
    Train the NeRF model and visualize/save images and depth maps after each epoch.

    Parameters:
    -----------
    data_dir: str
        Path to the dataset containing images and poses.
    epochs: int
        Number of epochs to train (default is pulled from config).
    """

    # Load the data (images and camera poses)
    images, poses = load_data(data_dir)

    for epoch in range(epochs):
        total_loss = 0

        for image, pose in zip(images, poses):

            # Sample rays from the image and cast them into the scene
            rays_o, rays_d = generate_rays(pose, image.shape[:2])

            # Forward pass through NeRF
            rgb_pred, density = model(rays_o)

            # Compute the loss (comparing predicted colors with actual image colors)
            loss = criterion(rgb_pred, image)
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update the learning rate using the scheduler
        scheduler.step()

        # Print training progress and PSNR
        avg_loss = total_loss / len( images )
        psnr_value = compute_psnr( rgb_pred, image )
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, PSNR: {psnr_value:.2f}")

        # Optionally save model checkpoints
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint_path = f'{config["checkpoint_dir"]}nerf_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at: {checkpoint_path}")

        # Visualize and save rendered image 
        display_image(rgb_pred, title=f"Rendered Image Epoch {epoch + 1}")
        save_image(rgb_pred, f'output/rendered_image_epoch_{epoch + 1}.png')

        # Compare rendered image with the ground truth
        ground_truth_image = image  # This is the ground truth image from the dataset
        compare_images(ground_truth_image, rgb_pred, title1="Ground Truth", title2="Prediction")
        save_comparison(ground_truth_image, rgb_pred, f'output/comparison_epoch_{epoch + 1}.png')

        # Visualize and save the depth map
        display_depth_map( density, title=f"Depth Map Epoch {epoch + 1}")
        save_depth_map( density, f'output/depth_map_epoch_{epoch + 1}.png')

    print("Training is done!")
