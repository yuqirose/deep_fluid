from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from reader import SmokeDataset


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='smoke', help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default= 96, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')

# Imported from main.py
parser.add_argument('--exp', type=int, default=64, metavar='N')
parser.add_argument('--sess', default="", metavar='N')

parser.add_argument('--train-dir', type=str, default="../tensorflow/train_data", metavar='N')
parser.add_argument('--test-dir', type=str, default="../tensorflow/test_data", metavar='N')
parser.add_argument('--save-dir', type=str, default="../saves", metavar='N')

parser.add_argument('--valid-size', type=float, default=0.5, metavar='N')

parser.add_argument('--input-len', type=int, default=4, metavar='N')
parser.add_argument('--output-len', type=int, default=2, metavar='N')
parser.add_argument('--train-sim-num', type=int, default=3, metavar='N')
parser.add_argument('--test-sim-num', type=int, default=1, metavar='N')
parser.add_argument('--sim-len', type=int, default=50, metavar='N')

parser.add_argument('--x-dim', type=int, default=64, metavar='N')
parser.add_argument('--y-dim', type=int, default=64, metavar='N')
parser.add_argument('--d-dim', type=int, default=1, metavar='N')
parser.add_argument('--h-dim', type=int, default=64, metavar='N')

parser.add_argument('--n-layers', type=int, default=1, metavar='N')
parser.add_argument('--n-epochs', type=int, default=10, metavar='N',
                                        help='number of epochs to train (default: 10)')
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
# print(args)

try:
    os.makedirs(args.save_dir)
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(args.imageSize),
                                   transforms.CenterCrop(args.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif args.dataset == 'lsun':
    dataset = dset.LSUN(root=args.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(args.imageSize),
                            transforms.CenterCrop(args.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif args.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=args.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(args.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif args.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, args.imageSize, args.imageSize),
                            transform=transforms.ToTensor())

elif args.dataset == 'smoke':
    normalize = transforms.Normalize(mean=0.00015, std=0.0088)
    dataset = SmokeDataset(args, train=True, transform=normalize)


assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                         shuffle=True, num_workers=int(args.workers))

device = torch.device("cuda:0" if args.cuda else "cpu")
ngpu = int(args.ngpu)
nz = int(args.nz)
ngf = int(args.ngf)
ndf = int(args.ndf)
nc = 1


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose3d(nz, ngf * 8, 4, (2,2,2), (0,0,0), bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x t x 4 x 4
            nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, (1,2,2), (2,1,1), bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x t x 8 x 8
            nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, (1,2,2), (2,1,1), bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x t x 16 x 16
            nn.ConvTranspose3d(ngf * 2, ngf, 4, (1,2,2), (2,1,1), bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x t x 32 x 32
            nn.ConvTranspose3d(ngf, nc, 4, (1,2,2), (0,1,1), bias=False),
            nn.Tanh()
            # state size. (nc) x t x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            # print('output ', output.shape)

        return output


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if args.netG != '':
    netG.load_state_dict(torch.load(args.netG))
# print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x t x 64 x 64
            nn.Conv3d(nc, ndf, 4, (1,2,2), (2,1,1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x t x 32 x 32
            nn.Conv3d(ndf, ndf * 2, 4, (1,2,2), (2,1,1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x t x 16 x 16
            nn.Conv3d(ndf * 2, ndf * 4, 4, (1,2,2), (2,1,1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x t x 8 x 8
            nn.Conv3d(ndf * 4, ndf * 8, 4, (2,2,2), (2,1,1), bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x t x 4 x 4
            nn.Conv3d(ndf * 8, 1, 4, (2,2,2), (0,0,0), bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if args.netD != '':
    netD.load_state_dict(torch.load(args.netD))
# print(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(args.batchSize, nz, 1, 1, 1)
real_label = 1
fake_label = 0

# setup argsimizer
argsimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
argsimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

for epoch in range(args.niter):
    for i, (data, target)  in enumerate(dataloader, 0):

        # permute B X D X C x H x W ==> B X C X D x H x W
        data = data.permute(0,2,1,3,4)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data.to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)
        # print('input ', real_cpu.shape)
        # print('label', label.shape)
        output = netD(real_cpu)
        # print('pred', output.shape)

        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        argsimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        argsimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, args.niter, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu[-1],
                    '%s/real_samples.png' % args.save_dir,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake[-1].detach(),
                    '%s/fake_samples_epoch_%03d.png' % (args.save_dir, epoch),
                    normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.save_dir, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.save_dir, epoch))
