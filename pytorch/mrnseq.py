from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch.autograd import Variable
import numpy as np
import random

EOS_token = 1

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, n_layers=1, args=None):
        super(EncoderRNN, self).__init__()
        self.args = args
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        c_dim = self.args.c_dim
        conv_dim = 4

        self.cnn =  nn.Sequential(
            nn.Conv2d(c_dim, conv_dim, 4, 2, 1),#in_channels, out_channels, kernel, stride, padding
            nn.BatchNorm2d(conv_dim),
            nn.Conv2d(conv_dim, conv_dim*2, 4, 2, 1),
            nn.BatchNorm2d(conv_dim*2),
            nn.Conv2d(conv_dim*2, conv_dim*4, 4, 2, 1),
            nn.BatchNorm2d(conv_dim*4),
            nn.Conv2d(conv_dim*4, conv_dim*8, 4, 2, 1)
        )

        # input_size = int(self.args.x_dim * self.args.y_dim/(2**3) ) #tbd:calculate
        input_size = self.args.x_dim*self.args.y_dim*self.args.c_dim 
        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers)
        self.fc2 = nn.Linear(self.hidden_size, input_size)

    def forward(self, x, h):
        # x = self.cnn(x) #channel_in=1
        x = x.contiguous().view(x.size(0), -1)        
        x = self.fc1(x)
        x = torch.unsqueeze(x, 0) # T=1, TxBxD
        x, h = self.gru(x, h)
        x = self.fc2(x.squeeze(0))
        x = x.view(1, self.args.batch_size, self.args.x_dim, self.args.y_dim)
        return x, h

    def initHidden(self):
        h = Variable(torch.zeros(self.n_layers, self.args.batch_size, self.hidden_size))
        if self.args.cuda:
            return h.cuda()
        else:
            return h

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, n_layers=1, args=None):
        super(DecoderRNN, self).__init__()
        self.args = args

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        conv_dim = 4
        c_dim = self.args.c_dim

        # input_size = int(self.args.x_dim * self.args.y_dim/(2**3))
        input_size = self.args.x_dim*self.args.y_dim*self.args.c_dim 

        self.cnn_enc =  nn.Sequential(
            nn.Conv2d(c_dim, conv_dim, 4, 2, 1),#in_channels, out_channels, kernel, stride, padding
            nn.BatchNorm2d(conv_dim),
            nn.Conv2d(conv_dim, conv_dim*2, 4, 2, 1),
            nn.BatchNorm2d(conv_dim*2),
            nn.Conv2d(conv_dim*2, conv_dim*4, 4, 2, 1),
            nn.BatchNorm2d(conv_dim*4),
            nn.Conv2d(conv_dim*4, conv_dim*8, 4, 2, 1)
        )
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.fc2 = nn.Linear(hidden_size, input_size)

        self.cnn_dec =  nn.Sequential(
            nn.ConvTranspose2d(conv_dim*8, conv_dim*4, 4, 2,1),
            nn.BatchNorm2d(conv_dim*4),
            nn.ConvTranspose2d(conv_dim*4, conv_dim*2, 4,2,1),
            nn.BatchNorm2d(conv_dim*2),
            nn.ConvTranspose2d(conv_dim*2, conv_dim, 4,2,1),
            nn.BatchNorm2d(conv_dim),
            nn.ConvTranspose2d(conv_dim,c_dim, 4,2,1)
        )

    def forward(self, x, h):
        # x = self.cnn_enc(x) 
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.unsqueeze(x, 0) # T=1, TxBxD
        x, h = self.gru(x, h) #TxBxD
        # x = torch.squeeze(x, 0)
        x = self.fc2(x.squeeze(0))
        # x = x.view(x.size(0), 32, 4, 4) # channel up, kernel down with more convs
        # x = self.cnn_dec(x)
        x = x.view(1,  self.args.batch_size, self.args.x_dim, self.args.y_dim)
        return x, h

    def initHidden(self):
        h = Variable(torch.zeros(self.n_layers, self.args.batch_size, self.hidden_size))
        if self.args.cuda:
            return h.cuda()
        else:
            return h



class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, n_layers=1, args=None):
        super(AttnDecoderRNN, self).__init__()
        self.args = args

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

        self.attn = nn.Linear(self.hidden_size * 2, self.args.output_len)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.out = nn.Linear(self.hidden_size * 2, output_size)

        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden, encoder_outputs):
        # input  B x H
        # hidden 1 x B x H
        # output B x H

        embedded = F.relu(self.embedding(input))

        # Calculate attention weights and apply to encoder outputs

        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden.squeeze(0)), 1))) # B x T

        print(attn_weights.size(), "attn_weights", encoder_outputs.size(), "encoder_outputs")

        # B x 1 x T * B x T x H = B x 1 x H = B x H
        context = torch.bmm(attn_weights.unsqueeze(1),
                            encoder_outputs.transpose(0, 1)).squeeze(1)

        # Combine embedded input word and attended context, run through RNN
        rnn_input = self.attn_combine(torch.cat((embedded, context), 1))
        rnn_input = rnn_input.unsqueeze(0)

        output, hidden = self.gru(rnn_input, hidden)

        output = self.softmax(self.out( torch.cat((output.squeeze(0), context), 1) ))

        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers, self.args.batch_size, self.hidden_size))
        if self.args.cuda:
            return result.cuda()
        else:
            return result

class FocDecoderRNN(nn.Module):
    def __init__(self, hidden_size, n_layers=1, args=None):
        super(FocDecoderRNN, self).__init__()
        self.args = args

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        input_size = self.args.x_dim*self.args.y_dim*self.args.c_dim 
        self.scale = 8

        self.embed1 = nn.Linear(input_size, hidden_size)
        self.gru1 = nn.GRU(hidden_size, hidden_size, n_layers)
        self.out1 = nn.Linear(hidden_size, 64)

        self.embed2 = nn.Linear(self.scale*self.scale, hidden_size)
        self.gru2 = nn.GRU(hidden_size, hidden_size, n_layers)
        self.out2 = nn.Linear(hidden_size, self.scale*self.scale)

    def forward(self, x, h, focal_area):
        # x:  B x 1 x H x W
        # h: 1 x B x H

        # low-res predict
        x1 = x.view(x.size(0), -1) 
        x1 = self.embed1(x1)
        x1 = torch.unsqueeze(x1, 0) # T=1, TxBxD
        x1, h1 = self.gru1(x1, h)
        x1 = self.out1(x1.squeeze(0))
        x1 = x1.view(self.args.batch_size, 8, 8)
        # print('x1.shape', x1.shape)

        # predict refine area: center, width 
        # c_x, c_y, c_w = focal_area.split(1, dim=2)
        # x_start = c_x-c_w+1
        # x_end =  c_x+c_w-1
        # y_start = c_y-c_y+1
        # y_end = c_y+c_w-1
        # x2 = x.narrow(1, x_start, x_end)
        # x2 = x2.narrow(2, y_start,y_end)

        # predict refine area: masking
        # focal_area(BxD)
        cell_list = (focal_area > 0.5).nonzero()
        # combine outputs
        y = x1.view(x.size(0), -1, 1).repeat(1, 1, 64).view(x.size(0), 64, 64)
        y_h = h1
        if len(cell_list):
            for cell in cell_list:
                # high-res predict
                xs = cell[0].data.cpu()[0]* self.scale
                xe = xs+ self.scale

                ys = cell[1].data.cpu()[0]* self.scale 
                ye = ys+ self.scale
                # print('focus range:', xs, xe, '|', ys, ye)
           
                x2 = x[:,:,xs:xe, ys:ye].contiguous().view(x.size(0),-1)
                x2 = self.embed2(x2)
                x2 = torch.unsqueeze(x2, 0) # T=1, TxBxD
                x2, h2 = self.gru2(x2, h)
                x2 = self.out2(x2.squeeze(0))
                x2 = x2.view(self.args.batch_size, 8, 8)
                y[:,xs:xe,ys:ye] = y[:,xs:xe,ys:ye]+x2

                y_h = torch.cat((y_h, h2))
        

        # combine hidden 
        y = y.unsqueeze(0)  
        return y, y_h

    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers, self.args.batch_size, self.hidden_size))
        if self.args.cuda:
            return result.cuda()
        else:
            return result


class Seq2Seq(nn.Module):
    def __init__(self, args):
        super(Seq2Seq, self).__init__()

        self.args = args
        T = torch.cuda if self.args.cuda else torch
     
        self.encoder = EncoderRNN(self.args.h_dim, self.args.n_layers, args=args)

        focus_size = 3
        self.focus = nn.Sequential(
            nn.Linear(self.args.h_dim, focus_size),
            nn.Sigmoid()
        )

        self.use_focus = True
        self.use_attn = False
        if self.use_attn:
            self.decoder = AttnDecoderRNN(self.args.h_dim, args=args)
        if self.use_focus:
            self.decoder = FocDecoderRNN(self.args.h_dim, args = args)
        else:
            self.decoder = DecoderRNN(self.args.h_dim, self.args.n_layers, args=args)

        self.teacher_forcing_ratio = 0.5


    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def forward(self, x, y):
        # update batch size
        self.args.batch_size  = x.size(0)
        # encoder forward pass
        encoder_hidden = self.encoder.initHidden()
        encoder_outputs = []
        # B x T x C x H x W -> T x B x C x H x W
        encoder_inputs = x.permute(1,0,2,3,4)
        for t in range(self.args.input_len):
            # TBD: different batch size
            encoder_output, encoder_hidden = self.encoder(encoder_inputs[t], encoder_hidden)
            encoder_outputs += [encoder_output]
        encoder_outputs = torch.cat(encoder_outputs, 0)

        # decoder forward pass
        decoder_input = Variable(torch.zeros(x.size(0), self.args.c_dim, self.args.x_dim, self.args.y_dim))
        decoder_input = decoder_input.cuda() if self.args.cuda else decoder_input
        # initialize decoder with encoder final state
        decoder_hidden = encoder_hidden

        target_variable = y.permute(1,0,2,3,4)
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        decoder_outputs = []
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(self.args.output_len):
                if self.use_attn:
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                if self.use_focus:
                    # predict focus area (center and width)
                    focal_area = self.focus(decoder_hidden.transpose(0, 1).squeeze(1))
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden, focal_area)
                else: 
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden)
                decoder_input = target_variable[di]  # Teacher forcing
                decoder_outputs += [decoder_output]
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(self.args.output_len):
                if self.use_attn:
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs)

                if self.use_focus:
                    # predict focus area (center and width)
                    focal_area = self.focus(decoder_hidden.transpose(0, 1).squeeze(1))
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden, focal_area)
                else: 
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden)
                decoder_input = decoder_output
                decoder_outputs += [decoder_output]

        decoder_outputs_with_t = [torch.unsqueeze(y.permute(1,0,2,3), 1) for y in decoder_outputs]
        output = torch.cat(decoder_outputs_with_t, dim=1)
        return output