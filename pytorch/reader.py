"""Summary
"""
import random, struct, sys, os
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append("../tensorflow/tools")
import uniio


class SmokeDataset(Dataset):
    def __init__(self, args, data_dir="../tensorflow/data/", num_sim=1, num_frame=3,
        train=True, valid=False, train_valid_split=0.1, transform=None):
        """
        Args:
            args (TYPE): Description
            data_dir (TYPE): Description
            num (int, optional): Description
            train (bool, optional): Description
            valid (bool, optional): Description
            train_valid_split (float, optional): Description
            transform (callable, optional): Optional transform to be applied
                on a sample.
        
        Deleted Parameters:
            root_dir (string): Directory with all the images.
        """
        self.args = args
        self.data_dir = data_dir
        self.N = num_sim
        self.T = num_frame
        self.train = train
        self.valid = valid
        self.train_valid_split = train_valid_split

    

    def __len__(self):
        """ num of frames        
        Returns:
            int: Total num of frames
        """
        return self.N

    def __getitem__(self, sim):
        """Summary
        
        Args:
            idx (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        states = np.empty([self.args.x_dim,self.args.y_dim,0])

        sim = sim + 1000 #start from 1000
        for t in range(self.T):
            filename = "%s/simSimple_%04d/density_%04d.uni" 
            uniPath = filename % (self.data_dir, sim, t)  # 100 files per sim
            header, content = uniio.readUni(uniPath)
            h = header['dimX']
            w  = header['dimY']
            arr = np.reshape(content, [w, h])
            arr = arr[::-1] # reverse order
            arr = np.reshape(arr, [w, h, 1])
            states = np.append( states, arr, 2 )

        print(states.shape)
        data = states[:self.args.input_len,]
        label = states[self.args.input_len:,]

        data = torch.from_numpy(data).type(torch.FloatTensor)
        label = torch.from_numpy(label).type(torch.FloatTensor)

        return data, label


class LorenzDataset(Dataset):
    """
    Attributes:
        args (TYPE): Description
        data_dir (TYPE): Description
        N (TYPE): Description
        res (TYPE): Description
        s0 (TYPE): Description
        train (TYPE): Description
        train_valid_split (TYPE): Description
        valid (TYPE): Description
    """
    def __init__(self, args, data_dir, res=1, train=True,
        valid=False, train_valid_split=0.1, transform=None):
        """
        Args:
            args (TYPE): Description
            data_dir (TYPE): Description
            res (int, optional): Description
            train (bool, optional): Description
            valid (bool, optional): Description
            train_valid_split (float, optional): Description
            transform (callable, optional): Optional transform to be applied
                on a sample.
        
        Deleted Parameters:
            root_dir (string): Directory with all the images.
        """
        self.args = args
        self.data_dir = data_dir
        self.res = res
        self.train = train
        self.valid = valid
        self.train_valid_split = train_valid_split

        # Generate seed
        self.N = int(1e2)
        self.s0 = np.random.rand(self.N, 3)
        self._dt = 0.01
        
    @property
    def dt(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        return self._dt


    def __len__(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        return self.N

    def __getitem__(self, idx):
        """Summary
        
        Args:
            idx (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        i = random.randrange(0, self.N)

        total_time = self.args.input_len + self.args.output_len

        # Generate data on-the-fly
        states = self.gen_lorenz_series(self.s0[i], total_time, 1)
        #states = self.load_data_from_file(i)

        data = states[:self.args.input_len,]
        label = states[self.args.input_len:,]

        data = torch.from_numpy(data).type(torch.FloatTensor)
        label = torch.from_numpy(label).type(torch.FloatTensor)

        return data, label

    def lorenz(self, x, y, z, s=10, r=28, b=2.667):
        """Summary
        
        Args:
            x (TYPE): Description
            y (TYPE): Description
            z (TYPE): Description
            s (int, optional): Description
            r (int, optional): Description
            b (float, optional): Description
        
        Returns:
            TYPE: Description
        """
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return np.array([x_dot, y_dot, z_dot])

    def gen_lorenz_series(self, s0, num_steps, num_freq):
        """Summary
        
        Args:
            s0 (TYPE): Description
            num_steps (TYPE): Description
            num_freq (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # dt = 0.01

        s = np.empty((num_steps,3))
        s[0] = s0
        ss = np.empty((num_steps//num_freq,3))
        j = 0
        for i in range(num_steps-1):
            # Derivatives of the X, Y, Z state,
            if i%num_freq ==0:
                ss[j] = s[i]
                j += 1
            sdot = self.lorenz(s[i,0], s[i,1], s[i,2])
            s[i + 1] = s[i] + sdot  * self._dt
        return ss

    def cal_lyapunav(self, x, y, z, s=10, r=28, b=2.667):
        """calculate the Lyapunav exponents,
        http://www.math.tamu.edu/~mpilant/math614/Matlab/Lyapunov/LorenzSpectrum.pdf
        can also consider Richardson extrapolation
        
        Args:
            s (TYPE): Description
        """

        J = np.matrix([[-s, s, 0], [-z+r, -1, -x], [y, x, -b]])
        I = np.eye((3))
        dt = np.ones(3)*self._dt
        exps= I + dt@J
        print(exps)
        return exps






