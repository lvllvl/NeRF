import torch
import torch.nn as nn

# Basic Mean Squared Error (MSE) Loss
def mse_loss( pred_rgb, target_rgb ):
    """
    Compute Mean Squared Error (MSE) between the predicted and target
    RGB values.

    Parameters:
    --------

    pred_rgb: torch.Tensor
        - Predicted RGB values from the model.
    
    target_rgb: torch.Tensor
        - Ground truth RGB values.

    Returns:
    --------
    loss: torch.Tensor
        - The MSE loss.
    """

    criterion = nn.MSELoss()

    return criterion( pred_rgb, target_rgb )

# PSNR Metric (for evaluation)
def psnr( pred_rgb, target_rgb ):
    """
    Compute Peak Signal-to-Noise Ration (PSNR) between the predicted and target
    RGB values.

    Parameters:
    --------
    pred_rgb: torch.Tensor
        - Predicted RGB values from the model.

    target_rgb: torch.Tensor
        - Ground truth RGB values.

    Returns:
    --------
    psnr_value: float
        - The computed PSNR value. 
    """
    mse = torch.mean(( pred_rgb - target_rgb ) ** 2 )
    psnr_value = -10 * torch.log10( mse )

    return psnr_value

# Custom Loss Example (optional)
def custom_loss( pred_rgb, target_rgb, pred_density, target_density ):

    """
    Compute a custom loss combining RGB and density.

    Parameters:
    --------

    pred_rgb: torch.Tensor
        - Predicted RGB values from the model.
    
    target_rgb: torch.Tensor
        - Ground truth RGB values.

    pred_density: torch.Tensor
        - Predicted density values from the model.

    target_density: torch.Tensor
        - Ground truth density values.

    Returns:
    --------
    loss: torch.Tensor
        - The combined loss.
    """
    rgb_loss = mse_loss( pred_rgb, target_rgb )
    density_loss = torch.mean(( pred_density - target_density )**2 )
    
    total_loss = rgb_loss + density_loss * 0.1 # Example weighting factor

    return total_loss