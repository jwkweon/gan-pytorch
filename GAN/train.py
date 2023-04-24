import os
import random

import torch
import torch.nn as nn
from gan import Generator, Discriminator
from dataloader import Data_simple
from torch.utils.data import DataLoader

import torch.optim as optim
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


def make_noise(n, z_dim=100):
    return torch.randn(n, z_dim)

def make_ones(size):
    return torch.ones(size, 1)

def make_zeros(size):
    return torch.zeros(size, 1)


device = 'cuda'

data = 'mnist' # ["mnist", "fashion", "cifar10"]
n_batch = 128
n_epochs = 500
n_samples = 36
z_dim = 100


train_dataset = Data_simple(True, data)
train_loader = DataLoader(train_dataset, batch_size=n_batch, num_workers=4, shuffle=True, drop_last=True)

# print_stats(train_dataset)

generator = Generator(data)
discriminator = Discriminator(data)

generator.to(device)
discriminator.to(device)

g_optim = optim.SGD(generator.parameters(), lr=0.01, momentum=0.9)
d_optim = optim.SGD(discriminator.parameters(), lr=0.01, momentum=0.9)

# TO-DO : use wandb to log
g_losses = []
d_losses = []

loss_fn = nn.BCELoss()

fixed_z = make_noise(n_samples, z_dim).to(device)

real_label = 1.
fake_label = 0.

print(f"Starting Training...")

for epoch in range(1):
    for i, data in enumerate(train_loader):
        imgs, _ = data
        imgs = imgs.to(device)
        
        # update D : max log(D(x)) + log(1-D(G(z)))
        d_optim.zero_grad()
        
        label = make_ones(n_batch).to(device)
        print(imgs.shape)
        output = discriminator(imgs)
        
        D_loss_real = loss_fn(output, label)
        D_loss_real.backward()
        D_x = output.mean().item()
        
        z = make_noise(n_batch, z_dim).to(device)
        fake = generator(z)
        
        label = make_zeros(n_batch).to(device)
        output = discriminator(fake.detach())
        
        D_loss_fake = loss_fn(output, label)
        D_loss_fake.backward()
        D_G_z = output.mean().item()
        
        D_loss = D_loss_real + D_loss_fake
        
        d_optim.step()
        
        print(f'loss D : {D_loss}')
