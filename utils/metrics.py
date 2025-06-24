import torch

def compute_psnr( pred, target ):

    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between predicted and target images.

    Parameters:
    --------
    pred: torch.Tensor
        - The predicted image.
    
    target: torch.Tensor
        - The ground truth image.

    Returns:
    --------
    psnr: float
        - The PSNR value. 
    """

    mse = torch.mean( (pred - target) ** 2 )
    psnr = -10 * torch.log10( mse )

    return psnr

# TODO: SSSIM, MSE, LPIPS funcitons here