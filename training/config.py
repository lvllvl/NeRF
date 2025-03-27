config = {
    # Paths
    'data_dir': 'data/',
    'checkpoint_dir': 'checkpoints/',

    # Training parameters
    'epochs': 100,
    'batch_size': 1,
    'learning_rate': 5e-4,

    # NeRF model parameters
    'num_freqs': 10, # Number of frequencies for positional encoding
    # This should be adjusted, as needed 

    # Scheduler type (choose 'step', 'cosine', or 'exponential')
    'lr_scheduler': 'cosine',

    # StepLR parameters
    'step_size': 30,  # Number of epochs after which to reduce LR
    'gamma': 0.5,     # Multiplicative factor to reduce LR by

    # CosineAnnealingLR parameters
    'T_max': 50,      # Number of epochs for one cycle
    'eta_min': 1e-6,  # Minimum learning rate

    # Save intervals
    'save_interval': 10,
}
