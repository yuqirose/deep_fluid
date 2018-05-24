
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from torch.nn import Parameter
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from train import train, test
from reader import Smoke2dDataset, SmokeDataset
import os
import numpy as np
from seq2seq import Seq2Seq
# from mrnseq import Seq2Seq
from cnn_ae import Conv2dAE, Conv2dLSTM


import visdom
viz = visdom.Visdom()

# Training settings
parser = argparse.ArgumentParser(description='adaptive vrnn')
parser.add_argument('--exp', type=int, default=64, metavar='N')
parser.add_argument('--sess', default="", metavar='N')

parser.add_argument('--train-dir', type=str, default="../tensorflow/train_data", metavar='N')
parser.add_argument('--test-dir', type=str, default="../tensorflow/test_data", metavar='N')
parser.add_argument('--save-dir', type=str, default="../saves", metavar='N')

parser.add_argument('--model', type=str, default="vrnn", metavar='N')

parser.add_argument('--dataset', type=str, default="train", metavar='N')

parser.add_argument('--valid-size', type=float, default=0.5, metavar='N')

parser.add_argument('--input-len', type=int, default=3, metavar='N')
parser.add_argument('--output-len', type=int, default=2, metavar='N')
parser.add_argument('--train-sim-num', type=int, default=3, metavar='N')
parser.add_argument('--test-sim-num', type=int, default=1, metavar='N')
parser.add_argument('--sim-len', type=int, default=50, metavar='N')

parser.add_argument('--x-dim', type=int, default=64, metavar='N')
parser.add_argument('--y-dim', type=int, default=64, metavar='N')
parser.add_argument('--d-dim', type=int, default=1, metavar='N')
parser.add_argument('--h-dim', type=int, default=64, metavar='N')

parser.add_argument('--batch-size', type=int, default=5, metavar='N')
parser.add_argument('--n-layers', type=int, default=1, metavar='N')
parser.add_argument('--n-epochs', type=int, default=10, metavar='N',
                                        help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                                        help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                                        help='SGD momentum (default: 0.5)')
parser.add_argument('--l2', type=float, default=0.005, metavar='LR')
parser.add_argument('--opt', default="sgd")

parser.add_argument('--use-focus', action='store_true', default=False,
                                        help='disables CUDA training')
parser.add_argument('--no-cuda', action='store_true', default=False,
                                        help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                                        help='random seed (default: 1)')
parser.add_argument('--print-freq', type=int, default=10, metavar='N',
                                        help='how many batches to wait before printing status')
parser.add_argument('--vis-scalar-freq', type=int, default=1, metavar='N',
                                        help='how many batches to wait before visualing results')
parser.add_argument('--save-freq', type=int, default=5, metavar='N',
                                        help='how many epochs to wait before saving training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N')
parser.add_argument('--valid-freq', type=int, default=10, metavar='N')
parser.add_argument('--prev-ckpt', default="", help='')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()




if __name__ == "__main__":
    if args.cuda and torch.cuda.is_available(): print("Using CUDA")

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

    model = Seq2Seq(args)


    if args.cuda:
        model = model.cuda()
        
    #plot
    # initialize visdom loss plot
    fig = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.ones((1, 2)).cpu(),
        opts=dict(
            xlabel='Epoch',
            ylabel='Loss',
            title='Training/Valid Loss - Epoch',
            legend=['Train Loss', 'Valid Loss']
        )
    )

    epoch_fig = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=100*torch.ones((1,)).cpu(),
        opts=dict(
            xlabel='Iteration',
            ylabel='Loss',
            title='Current Epoch Training Loss',
            legend=['Loss']
        )
    )
    train_losses = []
    valid_losses = []

    for epoch in range(1, args.n_epochs + 1):
        
        #training + validation
        train_loss = train(train_loader, epoch, model, args, epoch_fig)
        valid_loss = test(valid_loader, epoch, model,args)
        
        # save train valid loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        #plot
        viz.line(
            X=torch.ones((1)).cpu() * epoch,
            Y=torch.Tensor([train_loss,valid_loss]).unsqueeze(0).cpu(),
            win=fig,
            update='append'
        )
        #saving model
        if epoch % args.save_freq == 1:
            fn = 'vrnn_state_dict_'+str(epoch)+'.pth'
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save(model.state_dict(), args.save_dir+"/"+fn)
            print('Saved model to '+fn)

    # testing
    test_loss = test(test_loader, epoch, model, args, valid=False)
    # save losses
    torch.save([train_losses, valid_losses, test_loss], args.save_dir+"/losses.pth")
