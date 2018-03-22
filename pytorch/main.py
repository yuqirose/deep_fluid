
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn import Parameter
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from train import train, test
from reader import SmokeDataset
from seq2seq import Seq2Seq
import os
import numpy as np



# Training settings
parser = argparse.ArgumentParser(description='adaptive vrnn')
parser.add_argument('--exp', type=int, default=64, metavar='N')
parser.add_argument('--sess', default="", metavar='N')

parser.add_argument('--train-dir', type=str, default="../tensorflow/train_data", metavar='N')
parser.add_argument('--test-dir', type=str, default="../tensorflow/test_data", metavar='N')
parser.add_argument('--save-dir', type=str, default="../tensorflow/saves", metavar='N')

parser.add_argument('--model', type=str, default="vrnn", metavar='N')

parser.add_argument('--dataset', type=str, default="train", metavar='N')

parser.add_argument('--valid-size', type=float, default=0.5, metavar='N')

parser.add_argument('--input-len', type=int, default= 1, metavar='N')
parser.add_argument('--output-len', type=int, default=3, metavar='N')
parser.add_argument('--x-dim', type=int, default=64
    , metavar='N')
parser.add_argument('--y-dim', type=int, default=64
    , metavar='N')
parser.add_argument('--h-dim', type=int, default=100, metavar='N')

parser.add_argument('--batch-size', type=int, default=1, metavar='N')
parser.add_argument('--n-layers', type=int, default=1, metavar='N')
parser.add_argument('--n-epochs', type=int, default=50, metavar='N',
                                        help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                                        help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                                        help='SGD momentum (default: 0.5)')
parser.add_argument('--l2', type=float, default=0.1, metavar='LR')
parser.add_argument('--opt', default="sgd")


parser.add_argument('--no-cuda', action='store_true', default=False,
                                        help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                                        help='random seed (default: 1)')
parser.add_argument('--print-freq', type=int, default=10, metavar='N',
                                        help='how many batches to wait before printing status')
parser.add_argument('--vis-scalar-freq', type=int, default=10, metavar='N',
                                        help='how many batches to wait before visualing results')
parser.add_argument('--save-freq', type=int, default=100, metavar='N',
                                        help='how many batches to wait before saving training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N')
parser.add_argument('--valid-freq', type=int, default=500, metavar='N')
parser.add_argument('--prev-ckpt', default="", help='')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()




if __name__ == "__main__":
    if args.cuda and torch.cuda.is_available(): print("Using CUDA")

    train_dataset = SmokeDataset(args, args.train_dir, num_sim=2)
    test_dataset  = SmokeDataset(args, args.test_dir, num_sim=1)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(args.valid_size * num_train))


    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)


    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size,  sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=args.batch_size)

    model = Seq2Seq(args)
    if args.cuda:
        model = model.cuda()
        

    for epoch in range(1, args.n_epochs + 1):
        
        #training + validation
        train(train_loader, epoch, model, args)
        test(valid_loader, epoch, model,args)

        #saving model
        if epoch % args.save_freq == 1:
            fn = 'vrnn_state_dict_'+str(epoch)+'.pth'
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save(model.state_dict(), args.save_dir+fn)
            print('Saved model to '+fn)

    # testing
    test(test_loader, epoch, model, args,valid=False)
