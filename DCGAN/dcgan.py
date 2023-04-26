import torch
import torch.nn as nn

class Generator_cifar10(nn.Module):
    def __init__(self):
        super(Generator_cifar10, self).__init__()
        self.z_dim = 100
        
        self.proj0 = nn.Linear(self.z_dim, 512*4*4)
        
        self.conv1 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        self.conv2 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
        self.conv3 = nn.Sequential(
                nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
                nn.Tanh()
            )

    def forward(self, x):
        x = self.proj0(x)
        x = x.view(-1, 512, 4, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
    
        return x
    
class Discriminator_cifar10(nn.Module):
    def __init__(self):
        super(Discriminator_cifar10, self).__init__()
        
        self.n_in = 3
        self.n_out = 1

        self.conv1 = nn.Sequential(
                    nn.Conv2d(self.n_in, 128, 4, 2, 1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(),
                    nn.Dropout(0.3)
                    )
        self.conv2 = nn.Sequential(
                    nn.Conv2d(128, 256, 4, 2, 1),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(),
                    nn.Dropout(0.3)
                    )
        self.conv3 = nn.Sequential(
                    nn.Conv2d(256, 512, 4, 2, 1),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(),
                    nn.Dropout(0.3)
                    )
        self.conv4 = nn.Sequential(
                    nn.Conv2d(512, self.n_out, 4, 1, 0),
                    nn.Sigmoid()
                    )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        return x



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.z_dim = 100
        
        self.proj0 = nn.Linear(self.z_dim, 1024*4*4)
        
        self.conv1 = nn.Sequential(
                nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        self.conv2 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        self.conv3 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
        self.conv4 = nn.Sequential(
                nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
                nn.Tanh()
            )

    def forward(self, x):
        x = self.proj0(x)
        x = x.view(-1, 1024, 4, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
    
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.n_in = 3
        self.n_out = 1

        self.conv1 = nn.Sequential(
                    nn.Conv2d(self.n_in, 128, 4, 2, 1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(),
                    nn.Dropout(0.3)
                    )
        self.conv2 = nn.Sequential(
                    nn.Conv2d(128, 256, 4, 2, 1),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(),
                    nn.Dropout(0.3)
                    )
        self.conv3 = nn.Sequential(
                    nn.Conv2d(256, 512, 4, 2, 1),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(),
                    nn.Dropout(0.3)
                    )
        self.conv4 = nn.Sequential(
                    nn.Conv2d(512, 1024, 4, 2, 1),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(),
                    nn.Dropout(0.3)
                    )
        self.conv5 = nn.Sequential(
                    nn.Conv2d(1024, self.n_out, 4, 1, 0),
                    nn.Sigmoid()
                    )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x