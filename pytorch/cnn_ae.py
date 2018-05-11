from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch.autograd import Variable

class Conv2dAE(nn.Module):
    def __init__(self, args):
        super(Conv2dAE, self).__init__()
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
        d_dim = self.args.d_dim
        
        self.encoder =  nn.Sequential(
            nn.Conv2d(d_dim, conv_dim, 4, 2, 1),#in_channels, out_channels, kernel, stride, padding
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
            nn.ConvTranspose2d(conv_dim,d_dim, 4,2,1)
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Conv2dAESeq(nn.Module):
    def __init__(self, args):
        super(Conv2dAESeq, self).__init__()
        conv_dim = 4
        self.conv_dim= conv_dim
        self.args = args
        d_dim = self.args.d_dim
        
        self.encoder =  nn.Sequential(
            nn.Conv2d(d_dim, conv_dim, 4, 2, 1),#in_channels, out_channels, kernel, stride, padding
            nn.BatchNorm2d(conv_dim),
            nn.Conv2d(conv_dim, conv_dim*2, 4, 2, 1),
            nn.BatchNorm2d(conv_dim*2),
            nn.Conv2d(conv_dim*2, conv_dim*4, 4, 2, 1),
            nn.BatchNorm2d(conv_dim*4),
            nn.Conv2d(conv_dim*4, conv_dim*8, 4, 2, 1),
            nn.BatchNorm2d(conv_dim*8),
            nn.Conv2d(conv_dim*8, conv_dim*16, 4, 2, 1),
            nn.BatchNorm2d(conv_dim*16),
            nn.Conv2d(conv_dim*16, conv_dim*32, 4, 2, 1)
        )

        channel_sz  = 128
        
        self.fc = nn.Linear(self.args.input_len*channel_sz, self.args.output_len*channel_sz)

        self.decoder = nn.Sequential(
           nn.ConvTranspose2d(conv_dim*32, conv_dim*16, 4, 2,1),
            nn.BatchNorm2d(conv_dim*16),
            nn.ConvTranspose2d(conv_dim*16, conv_dim*8, 4, 2,1),
            nn.BatchNorm2d(conv_dim*8),
            nn.ConvTranspose2d(conv_dim*8, conv_dim*4, 4, 2,1),
            nn.BatchNorm2d(conv_dim*4),
            nn.ConvTranspose2d(conv_dim*4, conv_dim*2, 4,2,1),
            nn.BatchNorm2d(conv_dim*2),
            nn.ConvTranspose2d(conv_dim*2, conv_dim, 4,2,1),
            nn.BatchNorm2d(conv_dim),
            nn.ConvTranspose2d(conv_dim,d_dim, 4,2,1)
        )


    def forward(self, x):
        bt_sz = x.size(0)
        enc = []
        x = x.permute(1,0,2,3,4)
        for t in range(self.args.input_len):
            enc_t = self.encoder(x[t])
            enc += [enc_t.unsqueeze(1)]
        enc = torch.cat(enc, dim=1)
        enc = enc.view(bt_sz,-1)

        dec = self.fc(enc)
        dec = dec.view(bt_sz, self.args.output_len, 128, 1, 1)

        y = []
        dec = dec.permute(1,0,2,3,4)
        for t in range(self.args.output_len):
            dec_t = self.decoder(dec[t])
            y += [dec_t.unsqueeze(1)]
        y = torch.cat(y, dim=1)
        return y

class Conv2dLSTM(nn.Module):
    """2d Convolution with LSTM:
    next frame prediction
    
    Attributes:
        args (TYPE): Description
        decoder (TYPE): Description
        encoder (TYPE): Description
    """
    
    def __init__(self, args):
        super(Conv2dLSTM, self).__init__()
        self.args = args
        conv_dim=4
        d_dim = self.args.d_dim

        # input size: N x C x D x H x W
        self.encoder =  nn.Sequential(
            nn.Conv2d(d_dim, conv_dim, 4, 2, 1),#in_channels, out_channels, kernel, stride, padding
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
            nn.ConvTranspose2d(conv_dim,d_dim, 4,2,1)
        )
        self.fc = nn.Linear(input_size, self.hidden_size)
        #input of shape (seq_len, batch, input_size)
        self.gru = nn.GRU(conv_dim*8+2, self.args.h_dim, self.args.n_layers)
        
    def forward(self, x):
        x = x.transpose(1,2) #swap C and D
        print('input x', x.shape)

        x = self.encoder(x)
        print('cnn x', x.shape)
        # (batch, input_size, seq_len) -> (batch, seq_len, input_size)
        y = x.view(x.size()[0], 1, -1).contiguous()
        # (batch, seq_len, input_size) -> (seq_len, batch, input_size)
        y = y.transpose(0, 1).contiguous()
        print('lstm y', y.shape) # dim
        y = self.gru(y)
        x = y.view(x.shape)
        x = self.decoder(x)
        return x

