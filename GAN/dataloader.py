import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
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
                transforms.Normalize((0.5, ), (0.5, ))
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
        
