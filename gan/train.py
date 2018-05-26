import argparse
import os
import math
import sys
import pickle
import time
import numpy as np
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torchvision import datasets, transforms
import torchvision.transforms as transforms

from torch import nn
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.utils
import torch.utils.data

import visdom


# ours
from model import Discriminator, num_trainable_params
from helpers import to_string, printlog, maybe_create, maybe_print, update_discrim, update_policy

from seq2seq import Seq2Seq

from pytorch.convlstm import ConvLSTM
from pytorch.reader import Smoke2dDataset, SmokeDataset


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--trial', type=int, required=True)
parser.add_argument('--model', type=str, required=True)

parser.add_argument('--x_dim', type=int, default=64, metavar='N')
parser.add_argument('--y_dim', type=int, default=64, metavar='N')
parser.add_argument('--d_dim', type=int, default=1, metavar='N')

parser.add_argument('--h_dim', type=int, required=True)
parser.add_argument('--rnn_dim', type=int, required=True)

parser.add_argument('--n_layers', type=int, required=False, default=2)
parser.add_argument('--seed', type=int, required=False, default=345)
parser.add_argument('--clip', type=int, required=True, help='gradient clipping')
parser.add_argument('--pre_start_lr', type=float, required=True, help='pretrain starting learning rate')
parser.add_argument('--pre_min_lr', type=float, required=True, help='pretrain minimum learning rate')
parser.add_argument('--batch_size', type=int, required=False, default=64)
parser.add_argument('--save_every', type=int, required=False, default=50, help='periodically save model')
parser.add_argument('--pretrain', type=int, required=False, default=50, help='num epochs to use supervised learning to pretrain')
parser.add_argument('--subsample', type=int, required=False, default=1, help='subsample sequeneces')
parser.add_argument('--cuda', action='store_true', default=True, help='use GPU')
parser.add_argument('--cont', action='store_true', default=False, help='continue training a model')
parser.add_argument('--pretrained_discrim', action='store_true', default=False, help='load pretrained discriminator')

parser.add_argument('--discrim_rnn_dim', type=int, required=True)
parser.add_argument('--discrim_layers', type=int, required=True, default=2)
parser.add_argument('--policy_learning_rate', type=float, default=1e-6, help='policy network learning rate for GAN training')
parser.add_argument('--discrim_learning_rate', type=float, default=1e-3, help='discriminator learning rate for GAN training')
parser.add_argument('--max_iter_num', type=int, default=60000, help='maximal number of main iterations (default: 60000)')
parser.add_argument('--log_interval', type=int, default=1, help='interval between training status logs (default: 1)')
parser.add_argument('--draw_interval', type=int, default=50, help='interval between drawing and more detailed information (default: 50)')
parser.add_argument('--pretrain_disc_iter', type=int, default=2000, help="pretrain discriminator iteration (default: 2000)")
parser.add_argument('--save_model_interval', type=int, default=50, help="interval between saving model (default: 50)")

parser.add_argument('--log_dir', type=str, default="/tmp/deep_fluid/log")
parser.add_argument('--data_dir', type=str, default="/tmp/deep_fluid/data")
parser.add_argument('--train_dir', type=str, default="/tmp/deep_fluid/data/train_data", metavar='N')
parser.add_argument('--test_dir', type=str, default="/tmp/deep_fluid/data/test_data", metavar='N')

parser.add_argument('--dataset', type=str, default="train", metavar='N')

parser.add_argument('--valid_size', type=float, default=0.5, metavar='N')

parser.add_argument('--input_len', type=int, default=3, metavar='N')
parser.add_argument('--output_len', type=int, default=2, metavar='N')
parser.add_argument('--train_sim_num', type=int, default=3, metavar='N')
parser.add_argument('--test_sim_num', type=int, default=1, metavar='N')
parser.add_argument('--sim_len', type=int, default=50, metavar='N')


args = parser.parse_args()


# pytorch settings
Tensor = torch.DoubleTensor
torch.set_default_tensor_type('torch.DoubleTensor')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

use_gpu = args.cuda

if not torch.cuda.is_available():
    args.cuda = False

# model parameters
params = {
    'model' : args.model,
    'y_dim' : args.y_dim,
    'h_dim' : args.h_dim,
    'rnn_dim' : args.rnn_dim,
    'n_layers' : args.n_layers,
    'discrim_rnn_dim' : args.discrim_rnn_dim,
    'discrim_num_layers' : args.discrim_layers,
    'cuda' : args.cuda,
    'device': device
}


# hyperparameters
pretrain_epochs = args.pretrain
clip = args.clip
start_lr = args.pre_start_lr
min_lr = args.pre_min_lr
batch_size = args.batch_size
save_every = args.save_every


# manual seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda: torch.cuda.manual_seed_all(args.seed)


# build model
policy_net = Seq2Seq(args, params).double().to(device)
discrim_net = Discriminator(params).double().to(device)

params['total_params'] = num_trainable_params(policy_net)
print(params)

# create save path and saving parameters
save_path = os.path.join(args.log_dir, 'saved/%03d/' % args.trial)
model_path = os.path.join(save_path, 'model')

maybe_create(save_path)
maybe_create(model_path)



# Data
normalize = transforms.Normalize(mean=0.00015, std=0.0088)

# train_dataset = Smoke2dDataset(args, train=True, transform=None)
# test_dataset  = Smoke2dDataset(args, train=False, transform=None)

train_dataset = SmokeDataset(args, train=True, transform=normalize)
test_dataset = SmokeDataset(args, train=False, transform=normalize)

num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(args.valid_size * num_train))

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=args.batch_size, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=args.batch_size, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(test_dataset,
    batch_size=1)



# figures and statistics
if os.path.exists('imgs'):
    shutil.rmtree('imgs')
if not os.path.exists('imgs'):
    os.makedirs('imgs')
vis = visdom.Visdom()
win_pre_policy = None
win_pre_path_length = None
win_pre_out_of_bound = None
win_pre_step_change = None


# continue a previous experiment
if args.cont and args.subsample < 16:
    maybe_print("loading model with step size {}...".format(args.subsample*2))
    state_dict = torch.load(save_path+'model/policy_step'+str(args.subsample*2)+'_training.pth')
    policy_net.load_state_dict(state_dict, strict=False)

    test_loss = run_epoch(False, args.subsample, policy_net, test_data, clip)
    printlog('Pretrain Test:\t' + str(test_loss))



############################################################################
##################       START ADVERSARIAL TRAINING       ##################
############################################################################

# load the best pretrained policy
# policy_state_dict = torch.load(save_path+'model/policy_step'+str(args.subsample)+'_state_dict_best_pretrain.pth')
# policy_net.load_state_dict(policy_state_dict)

# optimizer
optimizer_policy = torch.optim.Adam(
    filter(lambda p: p.requires_grad, policy_net.parameters()),
    lr=args.policy_learning_rate)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.discrim_learning_rate)

discrim_criterion = nn.BCELoss()
if use_gpu:
    discrim_criterion = discrim_criterion.cuda()

# stats
exp_p = []
win_path_length = None


# Save pretrained model
if args.pretrain_disc_iter > 250:
    torch.save(policy_net.state_dict(), save_path+'model/policy_step'+str(args.subsample)+'_pretrained.pth')
    torch.save(discrim_net.state_dict(), save_path+'model/discrim_step'+str(args.subsample)+'_pretrained.pth')


# GAN training
train_discrim = True


for batch_idx, (data, target) in enumerate(train_loader):

    maybe_print("Forward Pass")
    ts0 = time.time()

    data, target = data.double().to(device), target.double().to(device)

    print(data.shape, target.shape)
    print(data.type(), target.type())

    output = policy_net(data, target)

    ts1 = time.time()


    maybe_print("Updating Model")
    t0 = time.time()

    # update discriminator
    mod_p_epoch, exp_p_epoch = update_discrim(
        discrim_net, optimizer_discrim, discrim_criterion,
        exp_states, exp_actions,
        model_states, model_actions,
        i_iter, dis_times=3.0, use_gpu=use_gpu, train=train_discrim)

    exp_p.append(exp_p_epoch)

    # update policy network
    if i_iter > 3 and mod_p[-1] < 0.8:
        update_policy(policy_net, optimizer_policy, discrim_net, discrim_criterion, model_states_var, model_actions_var, i_iter, use_gpu)

    t1 = time.time()

    # log
    if i_iter % args.log_interval == 0:
        maybe_print('{}\tT_sample {:.4f}\tT_update {:.4f}\texp_p {:.3f}'.format(
            i_iter, ts1-ts0, t1-t0, exp_p[-1]))

    # vis

    # save
    if args.save_model_interval > 0 and (i_iter) % args.save_model_interval == 0:
        torch.save(policy_net.state_dict(), save_path+'model/policy_step'+str(args.subsample)+'_training.pth')
        torch.save(discrim_net.state_dict(), save_path+'model/discrim_step'+str(args.subsample)+'_training.pth')