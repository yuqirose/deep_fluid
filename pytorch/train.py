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

PLOT_ON = True

"""adaptive resolution sequence prediction"""
def train(train_loader, epoch, model, args, epoch_fig):

    train_loss = 0
    # intialize to zero
    viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1,)).cpu(),
        win= epoch_fig,
        update='update'
    )

    clip = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.l2)

    for batch_idx, (data, target) in enumerate(train_loader):
        # print("data shape ", data.shape,"target shape",target.shape)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        #forward + backward + optimize
        optimizer.zero_grad()
        # batch x channel x height x width
        if args.use_focus:
            output, output1, focal_area= model(data,target)

        else:
            output  = model(data, target)

        # output
        # print("output shape ", output.shape)
        # print("target shape ", target.shape)
        # print('target', target.data[0][:10])
        # print('output', output.data[0][:10])
        if args.use_focus:
            # mask_target = target.masked_scatter_(focal_area, target)
            loss = F.mse_loss(output, target)
        else:
            loss = F.mse_loss(output, target)


        loss.backward()
        optimizer.step()

        #grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm(model.parameters(), clip)
        train_loss += loss.data[0]

        #printing
        if batch_idx % args.print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / args.batch_size ))
            # plot the loss curve per epoch
            viz.line(
            X=torch.ones((1,)).cpu() * batch_idx,
            Y=torch.Tensor([loss.data[0]/args.batch_size]).cpu(),
            win= epoch_fig,
            update='append'
            )

    train_loss/= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss ))

    if PLOT_ON == True:
        # get model weights
        # for m in model.modules():
        #     if isinstance(m, nn.Conv2d):
        #         print(m.weight.data.shape)
        # conv1_weight = model.encoder[0].weight.data

        # plot the filter per epoch
        # viz.images(
        #     conv1_weight.view(-1,1, 4,4),
        #     opts=dict(
        #         jpgquality =1,
        #         caption="Conv2d Filter"
        #     )
        # )
        if target.data.dim()==5:
            target = torch.squeeze(target,1)
            output = torch.squeeze(output,1)
            if args.use_focus:
                output1 = torch.squeeze(output1,1)


        for t in range(args.output_len):
            target_img = target.data[0][t][0]#first dimension pressure
            output_img = output.data[0][t][0]

            viz.heatmap(target_img.cpu(), opts=dict(colormap='Greys', title='true'+"_"+str(t)))
            viz.heatmap(output_img.cpu(), opts=dict(colormap='Greys', title='pred'+"_"+str(t)))
        # viz.images(target_img.cpu(),
        #     opts=dict(
        #     caption='true',
        #     jpgquality=20
        #     )
        # )

        # viz.images(output_img.cpu(),
        #     opts=dict(
        #     caption='pred',
        #     jpgquality=20
        #     )
        # )

        if args.use_focus:
            output1_img = output1.data[0][0]
            mask_img = focal_area.data[0][0].type(torch.FloatTensor)

        # print('target shape', target_img.shape, 'pred shape',output1_img.shape, 'focal shape', output2_img.shape)
            viz.heatmap(output1_img.cpu(), opts=dict(xmin=-50, xmax=50, colormap='Greys', title='pred-base'))
            viz.heatmap(mask_img.cpu(), opts=dict(xmin=-1, xmax=1, colormap='Greys', title='mask'))

            # viz.images(output1_img.cpu(),
            #     opts=dict(
            #     caption='pred-base',
            #     jpgquality=20
            #     )
            # )
            # viz.images(mask_img.cpu(),
            #     opts=dict(
            #     caption='mask',
            #     jpgquality=20
            #     )
            # )




    return train_loss

def test(test_loader, epoch, model, args, valid=True):
    """uses test data to evaluate
    likelihood of the model"""

    test_loss = 0.0
    for i, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        # data = torch.transpose(data, 0, 1)
        # target = torch.transpose(target, 0, 1)

        # inference
        if args.use_focus:
            output, output1, focal_area = model(data, target)
        else :
            output = model(data, target)


        # output
        # print('target', target.data[0][:10])
        # print('output', output.data[0][:10])
        loss = F.mse_loss(output, target)

        test_loss += loss.data[0]
        if valid==False:
            # save test predictions
            sim_idx = i / args.sim_len
            sim_idx += 1000 #start from 1000
            step_idx = i  % args.sim_len
            fname ="s%04d_t%04d"
            fname = fname%(sim_idx,step_idx)
            save_fname = "/true_"+fname
            np.savez(args.save_dir+save_fname, data.data.cpu().numpy())
            save_fname = "/pred_low_"+fname
            np.savez(args.save_dir+save_fname, output.data.cpu().numpy())
            save_fname = "/pred_high_"+fname

            if args.use_focus:
                np.savez(args.save_dir+save_fname, output1.data.cpu().numpy())
                save_fname = "/mask_"+fname
                np.savez(args.save_dir+save_fname, focal_area.data.type(torch.FloatTensor).cpu().numpy())

            print('Saved prediction to '+args.save_dir+save_fname)

            if PLOT_ON == True and step_idx%50==1:
                if target.data.dim()==5:
                    target = torch.squeeze(target,1)
                    output = torch.squeeze(output,1)
                    if args.use_focus:
                        output1 = torch.squeeze(output1,1)


                target_img = target.data[0][0][0] #first dimension pressure
                output_img = output.data[0][0][0]
                viz.heatmap(target_img.cpu(), opts=dict(colormap='Greys', title='true'+fname))
                viz.heatmap(output_img.cpu(), opts=dict(colormap='Greys', title='pred'+fname))
                # viz.images(target_img.cpu(),
                #     opts=dict(
                #     caption='true'+fname,
                #     jpgquality=20
                #     )
                #  )

                # viz.images(output_img.cpu(),
                #     opts=dict(
                #     caption='pred'+fname,
                #     jpgquality=20
                #     )
                # )

                if args.use_focus:
                    mask_img = focal_area.data[0][0].type(torch.FloatTensor)
                    output1_img = output1.data[0][0]
                    viz.heatmap(output1_img.cpu(), opts=dict(colormap='Greys', title='pred-base'+fname))
                    viz.heatmap(mask_img.cpu(), opts=dict(colormap='Greys', title='mask'+fname))
                    # viz.images(output1_img.cpu(),
                    #     opts=dict(
                    #     caption='pred-base',
                    #     jpgquality=20
                    #     )
                    # )
                    # viz.images(mask_img.cpu(),
                    #    opts=dict(
                    #    caption='mask',
                    #    jpgquality=20
                    #    )
                    # )

    test_loss /= len(test_loader.dataset)

    if valid==True:
        print('====> Valid set loss: Loss = {:.4f} '.format(test_loss))
    else:
        print('====> Test set loss: Loss = {:.4f} '.format(test_loss))



    # video = output.permute(0,2,3,1).data.cpu().numpy()
    # viz.video(tensor=video) #LxHxWxC

    return test_loss


