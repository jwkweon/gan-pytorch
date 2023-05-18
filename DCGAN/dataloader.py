import os
import numpy as np
from PIL import Image
from glob import glob

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, LSUN, ImageFolder
from torchvision import transforms

class Data_simple(Dataset):
    """Data_simple

    This class is a simple dataset (32 x 32) for training small generative models. 
    Default setting is MNIST, but you can also choose FashionMNIST and CIFAR10.

    Args:
        train: if train, then True / else validation, then False
        dataset: default is "MNIST". Select ["mnist", "fashion", "cifar10"]
        
    """
    def __init__(self, train, args):
        # Define the directory for save or load.
        PATH = args.dataset_path
        dataset = args.dataset_name

        # Define the transform to apply to the data : range of data -> [-1, 1]
        if dataset in ["mnist", "fashion"]:
            transform = transforms.Compose([
                transforms.Pad(2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        elif dataset == "cifar10":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        datasets = {
            "mnist": MNIST,
            "fashion": FashionMNIST,
            "cifar10": CIFAR10,
        }

        # Load the datasets
        self.train_dataset = datasets[dataset](root=PATH, train=train, download=True, transform=transform)        
        
        self.dataset_len = len(self.train_dataset.data)
        self.img_size = 32        
        self.train_data = self.train_dataset

    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        return self.train_data[idx]

    
class ImageNetDataset(ImageFolder):
    def __init__(self, args, transform=None):
        self.root_dir = os.path.join(args.dataset_path, 'image_net/train')
        
        self.norm = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))

        # Train Set
        self.transform = transforms.Compose([transforms.RandomResizedCrop(size=64),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            self.norm])
        
        self.image_paths = self.get_image_paths()
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image

    def get_image_paths(self):
        image_paths = []
        for root, dirs, _ in os.walk(self.root_dir):
            for dir in dirs:
                dir = os.path.join(root, dir)
                for _root, _, files in os.walk(dir):
                    for file in files:
                        if file.endswith('.jpg') or file.endswith('.png') \
                                or file.endswith('.JPEG'):
                            image_path = os.path.join(_root, file)
                            image_paths.append(image_path)

        return image_paths