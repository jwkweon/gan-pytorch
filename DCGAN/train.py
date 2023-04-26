import os
import random
import time
import math
import numpy as np
import imageio

import torch
import torch.nn as nn
from dcgan import Generator, Discriminator, Generator_cifar10, Discriminator_cifar10

from dataloader import Data_simple
from torch.utils.data import DataLoader
import torch.optim as optim

from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage


def make_noise(n, z_dim=100):
    return torch.randn(n, z_dim)


def make_ones(size):
    return torch.ones(size, 1, 1, 1)


def make_zeros(size):
    return torch.zeros(size, 1, 1, 1)


def make_grids(n_samples, samples):
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    
    num_cols = int(math.sqrt(n_samples))
    num_rows = int(math.ceil(n_samples / num_cols))
    
    grid_image = make_grid(samples, nrow=num_cols, padding=2, pad_value=1)
    
    return grid_image


def save_results(n_samples, samples, epoch, data):
    path_save = 'results/' + f'{data}/'
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    
    grid_image = make_grids(n_samples, samples)
    
    # Save the grid image
    save_image(grid_image, path_save + f'{epoch:04d}_results_{data}.png')
    

def save_gifs(n_samples, images, num_epochs, data):
    path_save = 'results/' + f'{data}/'
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        
    imgs = [np.array(to_image(i)) for i in images]
    imageio.mimsave(path_save + f'{num_epochs}_gif_results_{data}.gif', imgs)

device = "cuda:0"

dataset_name = 'cifar10' 
n_batch = 1024
n_epochs = 1000
n_samples = 36
z_dim = 100
to_image = ToPILImage()

if dataset_name == 'cifar10':
    train_dataset = Data_simple(True, dataset_name)
    generator = Generator_cifar10()
    discriminator = Discriminator_cifar10()
else: # 'lsun', 'imagenet'
    raise Exception('Dataset is not prepared!')

train_loader = DataLoader(train_dataset, batch_size=n_batch, num_workers=4, shuffle=True, drop_last=True)


generator.to(device)
discriminator.to(device)

g_optim = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optim = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# TO-DO : use wandb to log
g_losses = []
d_losses = []

img_for_gif = []

loss_fn = nn.BCELoss()

fixed_z = make_noise(n_samples, z_dim).to(device)

print(f"Starting Training...")
start_time = time.time()

for epoch in range(n_epochs):
    for i, data in enumerate(train_loader):
        imgs, _ = data
        imgs = imgs.to(device)
        
        # update D : max log(D(x)) + log(1-D(G(z)))
        d_optim.zero_grad()
        
        label = make_ones(n_batch).to(device)
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
        
        # update G : max log(D(G(z)))
        g_optim.zero_grad()
        
        label = make_ones(n_batch).to(device)
        output = discriminator(fake)

        G_loss = loss_fn(output, label)
        G_loss.backward()
        
        g_optim.step()
        
        if i % 20 == 0:
            print(f"[{epoch+1}/{n_epochs}][{i}/{len(train_loader)}][{time.time()-start_time:.4f}s]\
                    Loss_D: {D_loss:.4f}, Loss_G: {G_loss:.4f},\
                    D(x): {D_x:.4f}, D(G(x)): {D_G_z:.4f}")
    
    samples = generator(fixed_z)
    img_for_gif.append(make_grids(n_samples, samples.detach()))
    
    if (epoch+1) % 10 == 0:
        save_results(n_samples, samples.detach(), epoch+1, dataset_name)
    
save_gifs(n_samples, img_for_gif, n_epochs, dataset_name)