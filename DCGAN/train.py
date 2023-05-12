import os
import math
import numpy as np
import imageio
import argparse
import wandb
from tqdm import tqdm

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
    

def save_gifs(images, num_epochs, data):
    path_save = 'results/' + f'{data}/'
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        
    imgs = [np.array(to_image(i)) for i in images]
    imageio.mimsave(path_save + f'{num_epochs}_gif_results_{data}.gif', imgs)

def train():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--is_ckpt', action='store_true')
    parser.add_argument('--dataset_name', type=str, help='["cifar10", "lsun", "imagenet"]', default='cifar10')
    parser.add_argument('--dataset_path', type=str, default='./datasets')
    parser.add_argument('--n_batch', type=int, default='1024', help='num of batch_size')
    parser.add_argument('--n_epochs', type=int, default='500', help='num of epochs to train')
    parser.add_argument('--n_samples', type=int, default='36', help='num to generate samples')
    parser.add_argument('--z_dim', type=int, default='100', help='lenght of latent vector')
    parser.add_argument('--lr', type=float, default='0.0002', help='learning rate')
    parser.add_argument('--num_workers', type=int, default='4', help='num of loader workers')
    parser.add_argument('--wandb_log_iters', type=int, default=50, help='logging iters')
    args = parser.parse_args()
    
    # wandb init
    project = "DCGAN"
    proj_name = project + '-' + args.dataset_name
    wandb.init(project=project, name = proj_name,  settings = wandb.Settings(code_dir="."))
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    to_image = ToPILImage()
    
    train_dataset = Data_simple(True, args=args)
    train_loader = DataLoader(train_dataset, batch_size=args.n_batch, \
                        num_workers=args.num_workers, shuffle=True, drop_last=True)
    
    if args.dataset_name == 'cifar10':
        train_dataset = Data_simple(True, args=args)
        generator = Generator_cifar10()
        discriminator = Discriminator_cifar10()
    else: # 'lsun', 'imagenet'
        raise Exception('Dataset is not prepared!')

    generator.to(device)
    discriminator.to(device)

    g_optim = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optim = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    if args.is_ckpt:   # train from checkpoint
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        if not os.path.isfile(args.save_path+'/last.pt'):
            raise FileNotFoundError
        
        else:  
            if args.ckpt and os.path.isfile(args.save_path+'/last.pt'):
                ckpt_model = torch.load(args.save_path+'/last.pt')
                generator.load_state_dict(ckpt_model['model_G'])
                discriminator.load_state_dict(ckpt_model['model_D'])

                g_optim.load_state_dict(ckpt_model['optimizer_G'])
                d_optim.load_state_dict(ckpt_model['optimizer_D'])
                print('Model Loaded Successfully')

    # TO-DO : use wandb to log
    g_losses = []
    d_losses = []

    img_for_gif = []

    loss_fn = nn.BCELoss()

    fixed_z = make_noise(args.n_samples, args.z_dim).to(device)

    loader_len = train_loader.__len__()
    
    print(f"Starting Training...")

    for epoch in range(args.n_epochs):
        print ('#Epoch - '+str(epoch))
        
        for i, data in enumerate(tqdm(train_loader)):
            imgs, _ = data
            imgs = imgs.to(device)
            
            # update D : max log(D(x)) + log(1-D(G(z)))
            d_optim.zero_grad()
            
            label = make_ones(args.n_batch).to(device)
            output = discriminator(imgs)
            
            D_loss_real = loss_fn(output, label)
            D_loss_real.backward()
            D_x = output.mean().item()
            
            z = make_noise(args.n_batch, args.z_dim).to(device)
            fake = generator(z)
            
            label = make_zeros(args.n_batch).to(device)
            output = discriminator(fake.detach())
            
            D_loss_fake = loss_fn(output, label)
            D_loss_fake.backward()
            D_G_z = output.mean().item()
            
            D_loss = D_loss_real + D_loss_fake
            
            d_optim.step()
            
            # update G : max log(D(G(z)))
            g_optim.zero_grad()
            
            label = make_ones(args.n_batch).to(device)
            output = discriminator(fake)

            G_loss = loss_fn(output, label)
            G_loss.backward()
            
            g_optim.step()
            
            if i % args.wandb_log_iters == 0:
                wandb.log({'loss_D': D_loss,
                            'loss_G': G_loss,
                            'D(x)': D_x,
                            'D(G(z))': D_G_z,
                            'epoch':epoch,'steps':i+(loader_len*epoch)})
        
        samples = generator(fixed_z)
        img_for_gif.append(make_grids(args.n_samples, samples.detach()))
        
        if (epoch+1) % 10 == 0:
            save_results(args.n_samples, samples.detach(), epoch+1, args.dataset_name)
        
    save_gifs(img_for_gif, n_epochs, args.dataset_name)
    
    torch.save(
        {
            "model_D": discriminator.state_dict(),
            "model_G": generator.state_dict(),
            "optimizer_D": d_optim.state_dict(),
            "optimizer_G": g_optim.state_dict(),
        },
        args.save_path + '/last.pt'
    )
    print (f'Model Saved Successfully for #epoch {epoch}')