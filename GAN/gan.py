import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, dataset='mnist'):
        super(Generator, self).__init__()
        self.z_dim = 100
        if dataset == 'mnist' or dataset == 'fashion':
            self.ch = 1
        else:
            self.ch = 3
        
        self.n_out = 32*32*self.ch
        
        self.fc0 = nn.Sequential(
                nn.Linear(self.z_dim, 256),
                nn.ReLU()
            )
        self.fc1 = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU()
            )
        self.fc2 = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU()
            )
        self.fc3 = nn.Sequential(
                nn.Linear(1024, self.n_out),
                nn.Tanh()
            )

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        x = x.view(-1, self.ch, 32, 32)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, dataset='mnist'):
        super(Discriminator, self).__init__()
        
        if dataset == 'mnist' or dataset == 'fashion':
            self.ch = 1
        else:
            self.ch = 3
            
        self.n_in = 32*32*self.ch
        self.n_out = 1
        self.fc0 = nn.Sequential(
                    nn.Linear(self.n_in, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                    )
        self.fc1 = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                    )
        self.fc2 = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                    )
        self.fc3 = nn.Sequential(
                    nn.Linear(256, self.n_out),
                    nn.Sigmoid()
                    )
    
    def forward(self, x):
        x = x.view(-1, self.n_in)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x