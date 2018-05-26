from torch.autograd import Variable
from torch import nn
import torch
import numpy as np
import os
import struct
import pickle

use_gpu = torch.cuda.is_available()

def ones(*shape):
    return torch.ones(*shape).cuda() if use_gpu else torch.ones(*shape)

def zeros(*shape):
    return torch.zeros(*shape).cuda() if use_gpu else torch.zeros(*shape)

# transfer a 1-d vector into a string
def to_string(x):
    ret = ""
    for i in range(x.shape[0]):
        ret += "{:.3f} ".format(x[i])
    return ret


def printlog(line):
    print(line)
    with open(save_path+'log.txt', 'a') as file:
        file.write(line+'\n')

def maybe_create(d):
    if not os.path.exists(d):
        os.makedirs(d)

def maybe_print(s):
    print(s)


# train and pretrain discriminator
def update_discrim(discrim_net,
                   optimizer_discrim,
                   discrim_criterion,
                   exp_states,
                   exp_actions,
                   states,
                   actions,
                   i_iter,
                   dis_times,
                   use_gpu,
                   train=True):

    if use_gpu:
        exp_states, exp_actions, states, actions = exp_states.cuda(), exp_actions.cuda(), states.cuda(), actions.cuda()

    """update discriminator"""
    g_o_ave = 0.0
    e_o_ave = 0.0
    for _ in range(int(dis_times)):
        g_o = discrim_net(states, actions)
        e_o = discrim_net(exp_states, exp_actions)

        g_o_ave += g_o.cpu().data.mean()
        e_o_ave += e_o.cpu().data.mean()

        if train:
            optimizer_discrim.zero_grad()
            discrim_loss = discrim_criterion(g_o, zeros((g_o.shape[0], g_o.shape[1], 1))) +
                discrim_criterion(e_o, ones((e_o.shape[0], e_o.shape[1], 1)))
            discrim_loss.backward()
            optimizer_discrim.step()

    if dis_times > 0:
        return g_o_ave / dis_times, e_o_ave / dis_times


# train policy network
def update_policy(policy_net, optimizer_policy, discrim_net, discrim_criterion, \
                  states_var, actions_var, i_iter, use_gpu):
    optimizer_policy.zero_grad()
    g_o = discrim_net(states_var, actions_var)
    policy_loss = discrim_criterion(g_o, ones((g_o.shape[0], g_o.shape[1], 1)))
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 10)
    optimizer_policy.step()
