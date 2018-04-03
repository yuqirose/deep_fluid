from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch.autograd import Variable

class AutoEncoder(nn.Module):
    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
        #     nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
        #     nn.Tanh()
        # ) 
        conv_dim=4
        self.args = args
        c_dim = self.args.c_dim
        
        self.encoder =  nn.Sequential(
            nn.Conv2d(c_dim, conv_dim, 4, 2, 1),#in_channels, out_channels, kernel, stride, padding
            nn.BatchNorm2d(conv_dim),
            nn.Conv2d(conv_dim, conv_dim*2, 4, 2, 1),
            nn.BatchNorm2d(conv_dim*2),
            nn.Conv2d(conv_dim*2, conv_dim*4, 4, 2, 1),
            nn.BatchNorm2d(conv_dim*4),
            nn.Conv2d(conv_dim*4, conv_dim*8, 4, 2, 1),
        )
        

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(conv_dim*8, conv_dim*4, 4, 2,1),
            nn.BatchNorm2d(conv_dim*4),
            nn.ConvTranspose2d(conv_dim*4, conv_dim*2, 4,2,1),
            nn.BatchNorm2d(conv_dim*2),
            nn.ConvTranspose2d(conv_dim*2, conv_dim, 4,2,1),
            nn.BatchNorm2d(conv_dim),
            nn.ConvTranspose2d(conv_dim,c_dim, 4,2,1)
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
