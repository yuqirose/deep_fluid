import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init

from model_utils import *

def num_trainable_params(model):
    total = 0
    for p in model.parameters():
        count = 1
        for s in p.size():
            count *= s
        total += count
    return total

class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.params = params

        self.hidden_dim = params['discrim_rnn_dim']
        self.action_dim = params['y_dim']
        self.state_dim = params['y_dim']
        self.gpu = params['cuda']
        self.num_layers = params['discrim_num_layers']

        self.gru = nn.GRU(self.state_dim, self.hidden_dim, self.num_layers)
        self.dense1 = nn.Linear(self.hidden_dim + self.action_dim, self.hidden_dim)
        self.dense2 = nn.Linear(self.hidden_dim, 1)

    def forward(self, x, a, h=None):  # x: seq * batch * 10, a: seq * batch * 10
        p, hidden = self.gru(x, h)   # p: seq * batch * 10
        p = torch.cat([p, a], 2)   # p: seq * batch * 20
        prob = F.sigmoid(self.dense2(F.relu(self.dense1(p))))    # prob: seq * batch * 1
        return prob

    def init_hidden(self, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_dim))

class PROG_RNN(nn.Module):

    def __init__(self, params):
        super(PROG_RNN, self).__init__()

        self.params = params
        y_dim = params['y_dim']
        h_dim = params['h_dim']
        rnn16_dim = params['rnn16_dim']
        rnn8_dim = params['rnn8_dim']
        rnn4_dim = params['rnn4_dim']
        rnn2_dim = params['rnn2_dim']
        rnn1_dim = params['rnn1_dim']
        n_layers = params['n_layers']

        self.gru16 = nn.GRU(y_dim, rnn16_dim, n_layers)
        self.dec16 = nn.Sequential(
            nn.Linear(rnn16_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec16_mean = nn.Linear(h_dim, y_dim)
        self.dec16_std = nn.Sequential(
            nn.Linear(h_dim, y_dim),
            nn.Softplus())

        self.gru8 = nn.GRU(y_dim, rnn8_dim, n_layers)
        self.dec8 = nn.Sequential(
            nn.Linear(rnn8_dim + y_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec8_mean = nn.Linear(h_dim, y_dim)
        self.dec8_std = nn.Sequential(
            nn.Linear(h_dim, y_dim),
            nn.Softplus())

        self.gru4 = nn.GRU(y_dim, rnn4_dim, n_layers)
        self.dec4 = nn.Sequential(
            nn.Linear(rnn4_dim + y_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec4_mean = nn.Linear(h_dim, y_dim)
        self.dec4_std = nn.Sequential(
            nn.Linear(h_dim, y_dim),
            nn.Softplus())

        self.gru2 = nn.GRU(y_dim, rnn2_dim, n_layers)
        self.dec2 = nn.Sequential(
            nn.Linear(rnn2_dim + y_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec2_mean = nn.Linear(h_dim, y_dim)
        self.dec2_std = nn.Sequential(
            nn.Linear(h_dim, y_dim),
            nn.Softplus())

        self.gru1 = nn.GRU(y_dim, rnn1_dim, n_layers)
        self.dec1 = nn.Sequential(
            nn.Linear(rnn1_dim + y_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec1_mean = nn.Linear(h_dim, y_dim)
        self.dec1_std = nn.Sequential(
            nn.Linear(h_dim, y_dim),
            nn.Softplus())

    def forward(self, data, step_size):
        # data: seq_length * batch * 10
        loss = 0
        count = 0

        if step_size == 16:
            h16 = Variable(torch.zeros(self.params['n_layers'], data.size(1), self.params['rnn16_dim']))
            if self.params['cuda']:
                h16 = h16.cuda()
            for t in range(data.shape[0] - 1):
                state_t = data[t].clone()
                next_t = data[t+1].clone()

                _, h16 = self.gru16(state_t.unsqueeze(0), h16)
                dec_t = self.dec16(h16[-1])
                dec_mean_t = self.dec16_mean(dec_t)
                dec_std_t = self.dec16_std(dec_t)

                loss += nll_gauss(dec_mean_t, dec_std_t, next_t)
                count += 1

        if step_size == 8:
            h8 = Variable(torch.zeros(self.params['n_layers'], data.size(1), self.params['rnn8_dim']))
            if self.params['cuda']:
                h8 = h8.cuda()
            for t in range(0, data.shape[0] - 2, 2):
                state_t = data[t].clone()
                next_t = data[t+1].clone()
                macro_t = data[t+2].clone()

                _, h8 = self.gru8(state_t.unsqueeze(0), h8)
                dec_t = self.dec8(torch.cat([h8[-1], macro_t], 1))
                dec_mean_t = self.dec8_mean(dec_t)
                dec_std_t = self.dec8_std(dec_t)

                loss += nll_gauss(dec_mean_t, dec_std_t, next_t)
                count += 1

                _, h8 = self.gru8(next_t.unsqueeze(0), h8)

        if step_size == 4:
            h4 = Variable(torch.zeros(self.params['n_layers'], data.size(1), self.params['rnn4_dim']))
            if self.params['cuda']:
                h4 = h4.cuda()
            for t in range(0, data.shape[0] - 2, 2):
                state_t = data[t].clone()
                next_t = data[t+1].clone()
                macro_t = data[t+2].clone()

                _, h4 = self.gru4(state_t.unsqueeze(0), h4)
                dec_t = self.dec4(torch.cat([h4[-1], macro_t], 1))
                dec_mean_t = self.dec4_mean(dec_t)
                dec_std_t = self.dec4_std(dec_t)

                loss += nll_gauss(dec_mean_t, dec_std_t, next_t)
                count += 1

                _, h4 = self.gru4(next_t.unsqueeze(0), h4)

        if step_size == 2:
            h2 = Variable(torch.zeros(self.params['n_layers'], data.size(1), self.params['rnn2_dim']))
            if self.params['cuda']:
                h2 = h2.cuda()
            for t in range(0, data.shape[0] - 2, 2):
                state_t = data[t].clone()
                next_t = data[t+1].clone()
                macro_t = data[t+2].clone()

                _, h2 = self.gru2(state_t.unsqueeze(0), h2)
                dec_t = self.dec2(torch.cat([h2[-1], macro_t], 1))
                dec_mean_t = self.dec2_mean(dec_t)
                dec_std_t = self.dec2_std(dec_t)

                loss += nll_gauss(dec_mean_t, dec_std_t, next_t)
                count += 1

                _, h2 = self.gru2(next_t.unsqueeze(0), h2)

        if step_size == 1:
            h1 = Variable(torch.zeros(self.params['n_layers'], data.size(1), self.params['rnn1_dim']))
            if self.params['cuda']:
                h1 = h1.cuda()
            for t in range(0, data.shape[0] - 2, 2):
                state_t = data[t].clone()
                next_t = data[t+1].clone()
                macro_t = data[t+2].clone()

                _, h1 = self.gru1(state_t.unsqueeze(0), h1)
                dec_t = self.dec1(torch.cat([h1[-1], macro_t], 1))
                dec_mean_t = self.dec1_mean(dec_t)
                dec_std_t = self.dec1_std(dec_t)

                loss += nll_gauss(dec_mean_t, dec_std_t, next_t)
                count += 1

                _, h1 = self.gru1(next_t.unsqueeze(0), h1)

        return loss / count / data.shape[1]

    def sample16(self, data, seq_len=0, macro_data=None):
        # data: seq_length * batch * 10
        h16 = Variable(torch.zeros(self.params['n_layers'], data.size(1), self.params['rnn16_dim']))
        if self.params['cuda']:
            h16 = h16.cuda()
        if seq_len == 0:
            seq_len = data.shape[0]
        ret = []
        state_t = data[0]
        ret.append(state_t)

        for t in range(seq_len - 1):
            _, h16 = self.gru16(state_t.unsqueeze(0), h16)
            dec_t = self.dec16(h16[-1])
            dec_mean_t = self.dec16_mean(dec_t)
            dec_std_t = self.dec16_std(dec_t)
            state_t = sample_gauss(dec_mean_t, dec_std_t)
            ret.append(state_t)

        return torch.stack(ret, 0)

    def sample8(self, data, seq_len=0, macro_data=None):
        # data: seq_length * batch * 10
        h8 = Variable(torch.zeros(self.params['n_layers'], data.size(1), self.params['rnn8_dim']))
        if self.params['cuda']:
            h8 = h8.cuda()
        if seq_len == 0:
            seq_len = data.shape[0]

        macro_seq_len = seq_len // 2 + 1
        if macro_data is None:
            macro_data = self.sample16(data[::2], macro_seq_len)
        ret = []
        state_t = data[0]
        ret.append(state_t)

        for t in range(macro_seq_len - 1):
            macro_t = macro_data[t+1]

            _, h8 = self.gru8(state_t.unsqueeze(0), h8)
            dec_t = self.dec8(torch.cat([h8[-1], macro_t], 1))
            dec_mean_t = self.dec8_mean(dec_t)
            dec_std_t = self.dec8_std(dec_t)
            state_t = sample_gauss(dec_mean_t, dec_std_t)

            ret.append(state_t)
            ret.append(macro_t)
            _, h8 = self.gru8(state_t.unsqueeze(0), h8)
            state_t = macro_t

        return torch.stack(ret, 0)[:seq_len]

    def sample4(self, data, seq_len=0, macro_data=None):
        # data: seq_length * batch * 10
        h4 = Variable(torch.zeros(self.params['n_layers'], data.size(1), self.params['rnn4_dim']))
        if self.params['cuda']:
            h4 = h4.cuda()
        if seq_len == 0:
            seq_len = data.shape[0]

        macro_seq_len = seq_len // 2 + 1
        if macro_data is None:
            macro_data = self.sample8(data[::2], macro_seq_len)
        ret = []
        state_t = data[0]
        ret.append(state_t)

        for t in range(macro_seq_len - 1):
            macro_t = macro_data[t+1]

            _, h4 = self.gru4(state_t.unsqueeze(0), h4)
            dec_t = self.dec4(torch.cat([h4[-1], macro_t], 1))
            dec_mean_t = self.dec4_mean(dec_t)
            dec_std_t = self.dec4_std(dec_t)
            state_t = sample_gauss(dec_mean_t, dec_std_t)

            ret.append(state_t)
            ret.append(macro_t)
            _, h4 = self.gru4(state_t.unsqueeze(0), h4)
            state_t = macro_t

        return torch.stack(ret, 0)[:seq_len]

    def sample2(self, data, seq_len=0, macro_data=None):
        # data: seq_length * batch * 10
        h2 = Variable(torch.zeros(self.params['n_layers'], data.size(1), self.params['rnn2_dim']))
        if self.params['cuda']:
            h2 = h2.cuda()
        if seq_len == 0:
            seq_len = data.shape[0]

        macro_seq_len = seq_len // 2 + 1
        if macro_data is None:
            macro_data = self.sample4(data[::2], macro_seq_len)
        ret = []
        state_t = data[0]
        ret.append(state_t)

        for t in range(macro_seq_len - 1):
            macro_t = macro_data[t+1]

            _, h2 = self.gru2(state_t.unsqueeze(0), h2)
            dec_t = self.dec2(torch.cat([h2[-1], macro_t], 1))
            dec_mean_t = self.dec2_mean(dec_t)
            dec_std_t = self.dec2_std(dec_t)
            state_t = sample_gauss(dec_mean_t, dec_std_t)

            ret.append(state_t)
            ret.append(macro_t)
            _, h2 = self.gru2(state_t.unsqueeze(0), h2)
            state_t = macro_t

        return torch.stack(ret, 0)[:seq_len]

    def sample1(self, data, seq_len=0, macro_data=None):
        # data: seq_length * batch * 10
        h1 = Variable(torch.zeros(self.params['n_layers'], data.size(1), self.params['rnn1_dim']))
        if self.params['cuda']:
            h1 = h1.cuda()
        if seq_len == 0:
            seq_len = data.shape[0]

        macro_seq_len = seq_len // 2 + 1
        if macro_data is None:
            macro_data = self.sample2(data[::2], macro_seq_len)
        ret = []
        state_t = data[0]
        ret.append(state_t)

        for t in range(macro_seq_len - 1):
            macro_t = macro_data[t+1]

            _, h1 = self.gru1(state_t.unsqueeze(0), h1)
            dec_t = self.dec1(torch.cat([h1[-1], macro_t], 1))
            dec_mean_t = self.dec1_mean(dec_t)
            dec_std_t = self.dec1_std(dec_t)
            state_t = sample_gauss(dec_mean_t, dec_std_t)

            ret.append(state_t)
            ret.append(macro_t)
            _, h1 = self.gru1(state_t.unsqueeze(0), h1)
            state_t = macro_t

        return torch.stack(ret, 0)[:seq_len]

    def sample(self, data, seq_len=0):
        return self.sample1(data, seq_len)