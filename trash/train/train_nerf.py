import torch
import os
from utils.rays import generate_rays
from utils.metrics import compute_psnr
from utils.visualization_utils import display_image, save_image, display_depth_map, save_depth_map
from train.losses import mse_loss
from train.train_config import config

# Ensure checkpoint and output directory exist
if not os.path.exists(config['checkpoint_dir']):
    os.makedirs(config['checkpoint_dir'])

output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def train_nerf(model, dataloader, optimizer, epoch, scheduler=None, save_intervals=5):
    """
    Train the NeRF model and visualize/save images and depth maps after each epoch.

    Parameters:
    -----------
    model: torch.nn.Module
        The NeRF model to train.
    dataloader: torch.utils.data.DataLoader
        Dataloader providing batches of images and poses.
    optimizer: torch.optim.Optimizer
        Optimizer for updating model parameters.
    epoch: int
        The current epoch number.
    scheduler: torch.optim.lr_scheduler (optional)
        Scheduler for adjusting the learning rate.
    save_intervals: int
        Interval at which to save model checkpoints.
    """
    model.train()
    total_loss = 0

    for batch_idx, (image, pose) in enumerate(dataloader):
        # Prepare rays and forward pass
        rays_o, rays_d = generate_rays(pose, image.shape[:2])
        rgb_pred, density = model(rays_o)

        # Compute the loss
        loss = mse_loss(rgb_pred, image)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Scheduler step
    if scheduler:
        scheduler.step()

    # Logging and checkpointing
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    if (epoch + 1) % save_intervals == 0:
        checkpoint_path = f'{config["checkpoint_dir"]}/nerf_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at: {checkpoint_path}")
    
    # Visualization and PSNR calculation
    with torch.no_grad():
        for image, pose in dataloader:
            rays_o, rays_d = generate_rays(pose, image.shape[:2])
            rgb_pred, density = model(rays_o)
            psnr_value = compute_psnr(rgb_pred, image)

            print(f"Epoch {epoch + 1}, PSNR: {psnr_value:.2f}")

            display_image(rgb_pred, title=f"Rendered Image Epoch {epoch + 1}")
            save_image(rgb_pred, f'{output_dir}/rendered_image_epoch_{epoch + 1}.png')

            # Visualize and save depth map
            display_depth_map(density, title=f"Depth Map Epoch {epoch + 1}")
            save_depth_map(density, f'{output_dir}/depth_map_epoch_{epoch + 1}.png')
    
    print("Training for epoch completed.")