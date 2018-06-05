import torch 
from torch.autograd import Variable
import torchvision.utils as vutils
import numpy as np 
import matplotlib.pyplot as plt


import visdom
viz = visdom.Visdom()

from reader import read_npz_file

def mean_pool(x, kernel_sz, cuda=False):
    batch_sz = x.size(0)
    channel = x.size(1)
    width = x.size(2)
    height = x.size(3)

    n_x = int(width/kernel_sz)
    n_y = int(height/kernel_sz)

    y = Variable(torch.zeros(batch_sz, channel, n_x, n_y))
    if cuda: y = y.cuda()
    for c in range(channel):
        for i in range(n_x):
            for j in range(n_y):
                y[:, c, i,j] = x[:, c, i*kernel_sz: (i+1)*kernel_sz, \
                            j*kernel_sz: (j+1)*kernel_sz ].contiguous().view(batch_sz,-1).mean(1)

    return y

def kronecker_product(t1, t2):
    """
    Computes the Kronecker product between two tensors.
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_height, t2_width, 1)
          .view(out_height, out_width)
    )
    return expanded_t1 * tiled_t2

def mean_unpool(x, kernel_sz):
    batch_sz = x.size(0)
    channel = x.size(1)
    width = x.size(2)
    height = x.size(3)

    n_x = int(width * kernel_sz)
    n_y = int(height * kernel_sz)

    ones = Variable(torch.ones(kernel_sz, kernel_sz).type(x.data.type()))
    res = Variable(torch.zeros(batch_sz, channel, n_x, n_y).type(x.data.type()))
    for b in range(batch_sz):
        for c in range(channel):
            res[b,c] = kronecker_product(x[b,c], ones)  
    return res  



def vis_videos(video):
    # B xC x T x H x W
    video = video.permute(0,2,1,3,4)
    n_batch = video.size(0)
    n_frames = video.size(1)

    for i in range(n_batch):
        vid = video[i].squeeze(0)
        viz.images(vid)
    




if __name__ == "__main__":
# test mean_pool function
#load test data

    data_dir = "../tensorflow/train_data"
    sim_idx = 1000
    step_idx = 0
    T = 4
    

    x =  read_npz_file(data_dir, sim_idx, step_idx, 'pressure')
    x2 = read_npz_file(data_dir, sim_idx, step_idx, 'pressure')

    x = torch.from_numpy(np.vstack([x,x2])).type(torch.FloatTensor)
    print(' x shape', x.shape)

    y = mean_pool(x.unsqueeze(1), 4)
    y2 = mean_unpool(y,4)


    print('y2 shape', y2.shape)


    viz.images(x[0].cpu(),
        opts=dict(
        caption='true', 
        jpgquality=20       
        )
    )

    viz.images(y2[0].cpu(),
        opts=dict(
        caption='mean', 
        jpgquality=20       
        )
    )


    # test visualization
    states = np.empty([0, 1, 64, 64])

    for t in range(T):
            arr_p = read_npz_file(data_dir, sim_idx, step_idx+t,'pressure')
            states = np.append(states, np.expand_dims(arr_p,0), 0)

    data = torch.from_numpy(states).type(torch.FloatTensor)
    vis_videos(data)




