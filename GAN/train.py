import torch
from dataloader import Data_simple
from torch.utils.data import DataLoader
import numpy as np

def print_stats(dataset):
    imgs = np.array([img.numpy() for img, _ in dataset])
    print(f'shape: {imgs.shape}')
    
    if imgs.shape[1] == 3:
        min_r = np.min(imgs, axis=(2, 3))[:, 0].min()
        min_g = np.min(imgs, axis=(2, 3))[:, 1].min()
        min_b = np.min(imgs, axis=(2, 3))[:, 2].min()

        max_r = np.max(imgs, axis=(2, 3))[:, 0].max()
        max_g = np.max(imgs, axis=(2, 3))[:, 1].max()
        max_b = np.max(imgs, axis=(2, 3))[:, 2].max()

        mean_r = np.mean(imgs, axis=(2, 3))[:, 0].mean()
        mean_g = np.mean(imgs, axis=(2, 3))[:, 1].mean()
        mean_b = np.mean(imgs, axis=(2, 3))[:, 2].mean()

        std_r = np.std(imgs, axis=(2, 3))[:, 0].std()
        std_g = np.std(imgs, axis=(2, 3))[:, 1].std()
        std_b = np.std(imgs, axis=(2, 3))[:, 2].std()
        
        print(f'min: {min_r, min_g, min_b}')
        print(f'max: {max_r, max_g, max_b}')
        print(f'mean: {mean_r, mean_g, mean_b}')
        print(f'std: {std_r, std_g, std_b}')
    else:
        min_val = np.min(imgs, axis=(2, 3))[:].min()
        max_val = np.max(imgs, axis=(2, 3))[:].max()
        mean_val = np.mean(imgs, axis=(2, 3))[:].mean()
        std_val = np.std(imgs, axis=(2, 3))[:].std()
        
        print(f'min: {min_val}')
        print(f'max: {max_val}')
        print(f'mean: {mean_val}')
        print(f'std: {std_val}')
        
train_dataset = Data_simple(True, "mnist")
train_loader = DataLoader(train_dataset, batch_size=4, num_workers=4, shuffle=True)

# print_stats(train_dataset)

