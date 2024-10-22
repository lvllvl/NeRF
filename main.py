import torch
import argparse
import json
from models.nerf_model import NeRF
from data.nerf_dataset import NeRFDataset
from train.train_nerf import train_nerf  

def get_args():
    parser = argparse.ArgumentParser(description="Train NeRF model")
    parser.add_argument('--config', type=str, default="configs/nerf_config.json", help='Path to config file')
    parser.add_argument('--dataset', type=str, default='synthetic', help='Dataset type (e.g. synthetic, llff)')
    parser.add_argument('--data_dir', type=str, default='data/synthetic/lego', help='Directory of dataset')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def get_data_loaders(args):
    dataset = NeRFDataset(args.data_dir )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return dataloader

def get_model_and_optimizer(args, config):
    model = NeRF(**config['model'])  # Load model based on configuration
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    return model, optimizer

def train_model(args, config):
    dataloader = get_data_loaders(args)
    model, optimizer = get_model_and_optimizer(args, config)

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        train_nerf(model, dataloader, optimizer, epoch)
        # Add saving models, logging, etc., if needed

if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config)
    
    print(f"Training with {args.dataset} dataset located at {args.data_dir}")
    train_model(args, config)
