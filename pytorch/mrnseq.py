from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch.autograd import Variable
import numpy as np
import random
from utils import mean_pool, mean_unpool

EOS_token = 1

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, args=None):
        super(EncoderRNN, self).__init__()
        self.args = args
        self.n_layers = args.n_layers
        self.hidden_size = hidden_size
        d_dim = self.args.d_dim
        conv_dim = 4

        self.cnn =  nn.Sequential(
            nn.Conv2d(d_dim, conv_dim, 4, 2, 1),#in_channels, out_channels, kernel, stride, padding
            nn.BatchNorm2d(conv_dim),
            nn.Conv2d(conv_dim, conv_dim*2, 4, 2, 1),
            nn.BatchNorm2d(conv_dim*2),
            nn.Conv2d(conv_dim*2, conv_dim*4, 4, 2, 1),
            nn.BatchNorm2d(conv_dim*4),
            nn.Conv2d(conv_dim*4, conv_dim*8, 4, 2, 1)
        )

        # input_size = int(self.args.x_dim * self.args.y_dim/(2**3) ) #tbd:calculate
        input_size = self.args.x_dim*self.args.y_dim*self.args.d_dim 
        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers)
        self.fc2 = nn.Linear(self.hidden_size, input_size)

    def forward(self, x, h):
        # x = self.cnn(x) #channel_in=1
        x = x.contiguous().view(x.size(0), -1)        
        x = F.relu(self.fc1(x))
        x = torch.unsqueeze(x, 0) # T=1, TxBxD
        x, h = self.gru(x, h)
        x = F.tanh(self.fc2(x.squeeze(0)))
        x = x.view(1, self.args.batch_size, self.args.d_dim, self.args.x_dim, self.args.y_dim)
        return x, h

    def initHidden(self):
        h = Variable(torch.zeros(self.n_layers, self.args.batch_size, self.hidden_size))
        if self.args.cuda:
            return h.cuda()
        else:
            return h

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, args=None):
        super(DecoderRNN, self).__init__()
        self.args = args

        self.n_layers = args.n_layers
        self.hidden_size = hidden_size
        conv_dim = 4
        d_dim = self.args.d_dim

        # input_size = int(self.args.x_dim * self.args.y_dim/(2**3))
        input_size = self.args.x_dim*self.args.y_dim*self.args.d_dim 

        self.cnn_enc =  nn.Sequential(
            nn.Conv2d(d_dim, conv_dim, 4, 2, 1),#in_channels, out_channels, kernel, stride, padding
            nn.BatchNorm2d(conv_dim),
            nn.Conv2d(conv_dim, conv_dim*2, 4, 2, 1),
            nn.BatchNorm2d(conv_dim*2),
            nn.Conv2d(conv_dim*2, conv_dim*4, 4, 2, 1),
            nn.BatchNorm2d(conv_dim*4),
            nn.Conv2d(conv_dim*4, conv_dim*8, 4, 2, 1)
        )
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, self.n_layers)
        self.fc2 = nn.Linear(hidden_size, input_size)

        self.cnn_dec =  nn.Sequential(
            nn.ConvTranspose2d(conv_dim*8, conv_dim*4, 4, 2,1),
            nn.BatchNorm2d(conv_dim*4),
            nn.ConvTranspose2d(conv_dim*4, conv_dim*2, 4,2,1),
            nn.BatchNorm2d(conv_dim*2),
            nn.ConvTranspose2d(conv_dim*2, conv_dim, 4,2,1),
            nn.BatchNorm2d(conv_dim),
            nn.ConvTranspose2d(conv_dim,d_dim, 4,2,1)
        )

    def forward(self, x, h):
        # x = self.cnn_enc(x) 
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.unsqueeze(x, 0) # T=1, TxBxD
        x, h = self.gru(x, h) #TxBxD
        # x = torch.squeeze(x, 0)
        x = F.tanh(self.fc2(x.squeeze(0)))
        # x = x.view(x.size(0), 32, 4, 4) # channel up, kernel down with more convs
        # x = self.cnn_dec(x)
        x = x.view(self.args.batch_size, self.args.d_dim, self.args.x_dim, self.args.y_dim)
        x = x.unsqueeze(0)
        return x, h

    def initHidden(self):
        h = Variable(torch.zeros(self.n_layers, self.args.batch_size, self.hidden_size))
        if self.args.cuda:
            return h.cuda()
        else:
            return h



class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, args=None):
        super(AttnDecoderRNN, self).__init__()
        self.args = args

        self.n_layers = args.n_layers
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
    def __init__(self, hidden_size, args=None):
        super(FocDecoderRNN, self).__init__()
        self.args = args

        self.n_layers = args.n_layers
        self.kernel_sz =8
        input_size = self.args.d_dim * self.args.x_dim*self.args.y_dim
        low_input_size = int(input_size/(self.kernel_sz**2))

        # low res
        self.embed1 = nn.Linear(low_input_size, hidden_size)
        self.gru1 = nn.GRU(hidden_size, hidden_size, self.n_layers)
        self.out1 = nn.Linear(hidden_size, low_input_size)

        # focal area
        focus_size = int(self.args.x_dim*self.args.y_dim/(self.kernel_sz**2))
        self.focus = nn.Sequential(
            nn.Linear(self.args.n_layers*hidden_size, focus_size),
            nn.Sigmoid()
        )

        # high res
        self.embed2 = nn.Linear(input_size, hidden_size)
        self.gru2 = nn.GRU(hidden_size, hidden_size, self.n_layers)
        self.out2 = nn.Linear(hidden_size,input_size)

    def forward(self, x, h):
        # x:  B x C x H x W, C = 1
        # h: 1 x B x H
        # low-res predict
        x1 = mean_pool(x, self.kernel_sz, cuda=self.args.cuda)# compute mean behavior
        x1 = x1.view(x1.size(0), -1) 
        x1 = F.relu(self.embed1(x1))
        x1 = torch.unsqueeze(x1, 0) # T=1, TxBxD
        x1, h1 = self.gru1(x1, h)
        x1 = F.tanh(self.out1(x1.squeeze(0)))
        x1 = x1.view(self.args.batch_size, self.args.d_dim, int(self.args.x_dim/self.kernel_sz), int(self.args.y_dim/self.kernel_sz))
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
        context = h.view(self.args.batch_size,-1)
        focal_area = self.focus(context)

        focal_mask = (focal_area > 0.5).view((-1,1, int(self.args.x_dim/self.kernel_sz), int(self.args.y_dim/self.kernel_sz)))
        
        y1 = mean_unpool(x1, self.kernel_sz)
        focal_mask = mean_unpool(focal_mask, self.kernel_sz)
    
        y2 = y1.clone()

        # if len(cell_list):
        #     for cell in cell_list:
        #         # high-res predict
        #         xs = cell[0].data.cpu()[0]* self.kernel_sz
        #         xe = xs+ self.kernel_sz

        #         ys = cell[1].data.cpu()[0]* self.kernel_sz 
        #         ye = ys+ self.kernel_sz
        #         # print('focus range:', xs, xe, '|', ys, ye)
        #         x2 = x[:,:,xs:xe, ys:ye].contiguous().view(x.size(0),-1)
        x2 = x.view(x.size(0), -1) 
        x2 = F.relu(self.embed2(x2))
        x2 = torch.unsqueeze(x2, 0) # T=1, TxBxD
        x2, h2 = self.gru2(x2, h)
        x2 = F.relu(self.out2(x2.squeeze(0)))
        x2 = x2.view(self.args.batch_size, self.args.d_dim, self.args.x_dim, self.args.y_dim)
        y2 = y2 + x2.masked_scatter_(focal_mask, x2)

                # y2[:,:, xs:xe,ys:ye] = y2[:,:, xs:xe,ys:ye]+x2

                # need to change the aggregation
                # y_h = y_h + h2 #doesn't matter for T=1

                # print('h shape', y_h.shape)
        

        # combine outputs,combine hidden
        y1 = y1.unsqueeze(0)
        y2 = y2.unsqueeze(0)
        y_h = h1
        return (y1, y2, focal_mask), y_h

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
     
        self.encoder = EncoderRNN(self.args.h_dim, args=args)
   
        self.use_focus = False
        self.use_attn = False
        if self.use_attn:
            self.decoder = AttnDecoderRNN(self.args.h_dim, args=args)
        if self.use_focus:
            self.decoder = FocDecoderRNN(self.args.h_dim, args = args)
        else:
            self.decoder = DecoderRNN(self.args.h_dim,  args=args)
        if self.args.cuda: self.decoder = self.decoder.cuda()

        self.teacher_forcing_ratio = 0.5


    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def forward(self, x, y):
        # update batch size
        self.args.batch_size  = x.size(0)
        # encoder forward pass
        encoder_hidden = self.encoder.initHidden()
        encoder_outputs = []
        # B x T x D x H x W -> T x B x D x H x W
        encoder_inputs = x.permute(1,0,2,3,4)
        for t in range(self.args.input_len):
            # TBD: different batch size
            encoder_output, encoder_hidden = self.encoder(encoder_inputs[t], encoder_hidden)
            encoder_outputs += [encoder_output]
        encoder_outputs = torch.cat(encoder_outputs, 0)

        # decoder forward pass
        decoder_input = Variable(torch.zeros(x.size(0), self.args.d_dim, self.args.x_dim, self.args.y_dim))
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

                else:
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden)
                decoder_input = decoder_output
                decoder_outputs += [decoder_output]

        if self.use_focus:
            decoder_outputs_with_t = [y[0].permute(1,0,2,3,4) for y in decoder_outputs]
            output1 = torch.cat(decoder_outputs_with_t, dim=1)

            decoder_outputs_with_t = [y[1].permute(1,0,2,3,4) for y in decoder_outputs]
            output2 = torch.cat(decoder_outputs_with_t, dim=1)

            decoder_outputs_with_t = [y[2].permute(1,0,2,3) for y in decoder_outputs]
            focal_areas = torch.cat(decoder_outputs_with_t, dim=1)
            return output1, output2, focal_areas

        else:
            decoder_outputs_with_t = [y.permute(1,0,2,3,4) for y in decoder_outputs]
            output = torch.cat(decoder_outputs_with_t, dim=1)
            return output

