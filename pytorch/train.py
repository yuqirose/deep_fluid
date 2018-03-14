import math
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
import numpy as np

# visdom visualization
import visdom
viz = visdom.Visdom()

startup_sec = 1
while not viz.check_connection() and startup_sec > 0:
    time.sleep(0.1)
    startup_sec -= 0.1
assert viz.check_connection(), 'No connection could be formed quickly'

PLOT_ON = False

"""adaptive resolution sequence prediction"""
def train(train_loader, epoch, model, args):

    train_loss = 0
    clip = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data = torch.transpose(data, 0, 1)
        target = torch.transpose(target, 0, 1)

        #forward + backward + optimize
        optimizer.zero_grad()
        output = model(data)

        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        print(output[0,0,:], target[0,0,:])
        #grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm(model.parameters(), clip)

        #printing
        if batch_idx % args.print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / args.batch_size ))

        train_loss += loss.data[0]


    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

    if PLOT_ON == True and epoch %5==0:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plot_scatter3d(ax,  data[:,0,:].data, 'k')
        plot_scatter3d(ax,  target[:,0,:].data, 'r')
        plot_scatter3d(ax, output[:,0,:].data, 'b')
        viz.matplot(plt)

def test(test_loader, epoch, model):
    """uses test data to evaluate 
    likelihood of the model"""
    
    loss = 0.0
    for i, (data, target) in enumerate(test_loader):                                            
        
        data, target = Variable(data), Variable(target)
        data = torch.transpose(data, 0, 1)
        target = torch.transpose(target, 0, 1)

        # inference
        output = model(data)

        loss = F.mse_loss(output, target)
        loss /= len(test_loader.dataset)

    print('====> Test set loss: Loss = {:.4f} '.format(loss.data[0]))
