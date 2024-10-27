import torch
import argparse
from models.nerf_model import NeRF
from data.nerf_dataset import NeRFDataset
from train.train_nerf import train_nerf  
from train.train_config import config 

def get_args():
    parser = argparse.ArgumentParser(description="Train NeRF model")
    parser.add_argument('--dataset', type=str, default='synthetic', help='Dataset type (e.g. synthetic, llff)')
    parser.add_argument('--data_dir', type=str, default='data/synthetic/lego', help='Directory of dataset')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    args = parser.parse_args()
    return args

def get_data_loaders(args):
    dataset = NeRFDataset(args.data_dir, load_without_pose=True )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return dataloader

def get_model_and_optimizer(args, config):
    model = NeRF(num_freqs=config['num_freqs']) 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR( optimizer, step_size=config.get('scheduler_step_size', 10), gamma=config.get('scheduler_gamma', 0.9) ) # Optional scheduler
    return model, optimizer, scheduler

def train_model(args, config):
    dataloader = get_data_loaders(args)
    model, optimizer, scheduler = get_model_and_optimizer(args, config)

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        train_nerf(model, dataloader, optimizer, epoch, scheduler=scheduler, save_intervals=config.get('save_interval', 5))

if __name__ == "__main__":
    args = get_args()
    print( f"data_dir after get_args(): {args.data_dir}")
    
    print(f"Training with {args.dataset} dataset located at {args.data_dir}")
    train_model(args, config)
