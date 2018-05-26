import argparse
import os
import sys
import pickle
import time
import numpy as np
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bball_data import BBallData
from model import *
from torch.autograd import Variable
from torch import nn

from helpers import *
import visdom

Tensor = torch.DoubleTensor
torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser()
parser.add_argument('--trial', type=int, default=100)
parser.add_argument('--seed', type=int, default=666)
parser.add_argument('--max_iter_num', type=int, default=100)
parser.add_argument('--model', type=str, default=PROG_RNN)
parser.add_argument('--y_dim', type=int, default=10)
parser.add_argument('--h_dim', type=int, default=200)
parser.add_argument('--rnn1_dim', type=int, default=300)
parser.add_argument('--rnn2_dim', type=int, default=250)
parser.add_argument('--rnn4_dim', type=int, default=200)
parser.add_argument('--rnn8_dim', type=int, default=150)
parser.add_argument('--rnn16_dim', type=int, default=100)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--subsample', type=int, default=1)

args = parser.parse_args()
use_gpu = True
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_gpu:
    torch.cuda.manual_seed_all(args.seed)

# model parameters
params = {
    'model' : args.model,
    'y_dim' : args.y_dim,
    'h_dim' : args.h_dim,
    'rnn1_dim' : args.rnn1_dim,
    'rnn2_dim' : args.rnn2_dim,
    'rnn4_dim' : args.rnn4_dim,
    'rnn8_dim' : args.rnn8_dim,
    'rnn16_dim' : args.rnn16_dim,
    'n_layers' : args.n_layers,
    'cuda' : use_gpu,
}

save_path = 'saved/%03d/' % args.trial
if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.makedirs(save_path+'model/')

# define and load model
policy_net = PROG_RNN(params).double()
if use_gpu:
    policy_net = policy_net.cuda()
policy_state_dict = torch.load(save_path+'model/policy_step1_training.pth')
policy_net.load_state_dict(policy_state_dict, strict=False)

# load test data
test_data = torch.Tensor(pickle.load(open('bball_data/data/Xte_role.p', 'rb'))).transpose(0, 1)[:, ::args.subsample, :]
print(test_data.shape)

# stats
vis = visdom.Visdom()
win_path_length = None
win_out_of_bound = None
win_step_change = None
win_ave_player_dis = None
win_diff_max_min = None
win_ave_angle = None
if os.path.exists('imgs'):
    shutil.rmtree('imgs')
if not os.path.exists('imgs'):
    os.makedirs('imgs')

# test on a fixed set of data
fixed_test_data = Variable(test_data[:5].squeeze().transpose(0, 1))
if use_gpu:
    fixed_test_data = fixed_test_data.cuda()
test_fixed_data(policy_net, fixed_test_data, 'fixed', 0, args.subsample, num_draw=1)

# test model
for i_iter in range(args.max_iter_num):
    print(i_iter)
    mod_stats, exp_stats = \
        test_sample(policy_net, test_data, use_gpu, i_iter, draw=True)

    update = 'append' if i_iter > 0 else None
    win_path_length = vis.line(X = np.array([i_iter]), \
        Y = np.column_stack((np.array([exp_stats['ave_length']]), np.array([mod_stats['ave_length']]))), \
        win = win_path_length, update = update, opts=dict(legend=['expert', 'model'], title="average path length"))
    win_out_of_bound = vis.line(X = np.array([i_iter]), \
        Y = np.column_stack((np.array([exp_stats['ave_out_of_bound']]), np.array([mod_stats['ave_out_of_bound']]))), \
        win = win_out_of_bound, update = update, opts=dict(legend=['expert', 'model'], title="average out of bound rate"))
    win_step_change = vis.line(X = np.array([i_iter]), \
        Y = np.column_stack((np.array([exp_stats['ave_change_step_size']]), np.array([mod_stats['ave_change_step_size']]))), \
        win = win_step_change, update = update, opts=dict(legend=['expert', 'model'], title="average step size change"))
    win_ave_player_dis = vis.line(X = np.array([i_iter]), \
        Y = np.column_stack((np.array([exp_stats['ave_player_distance']]), np.array([mod_stats['ave_player_distance']]))), \
        win = win_ave_player_dis, update = update, opts=dict(legend=['expert', 'model'], title="average player distance"))
    win_diff_max_min = vis.line(X = np.array([i_iter]), \
        Y = np.column_stack((np.array([exp_stats['diff_max_min']]), np.array([mod_stats['diff_max_min']]))), \
        win = win_diff_max_min, update = update, opts=dict(legend=['expert', 'model'], title="average max and min path diff"))
    win_ave_angle = vis.line(X = np.array([i_iter]), \
        Y = np.column_stack((np.array([exp_stats['ave_angle']]), np.array([mod_stats['ave_angle']]))), \
        win = win_ave_angle, update = update, opts=dict(legend=['expert', 'model'], title="average rotation angle"))
