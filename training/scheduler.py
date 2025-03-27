import torch

def get_scheduler(optimizer, config):
    """
    Get the learning rate scheduler based on the configuration.

    Parameters:
    -----------
    optimizer: torch.optim.Optimizer
        The optimizer for which the scheduler is applied.
    config: dict
        The training configuration dictionary (from train_config.py).

    Returns:
    --------
    scheduler: torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler.
    """
    
    if config['lr_scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    
    elif config['lr_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['T_max'], eta_min=config['eta_min'])
    
    elif config['lr_scheduler'] == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'])
    
    else:
        raise ValueError(f"Unknown scheduler type: {config['lr_scheduler']}")
    
    return scheduler
