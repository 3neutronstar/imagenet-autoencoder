#!/usr/bin/env python

import argparse

import matplotlib.pyplot as plt

import torch

from torchvision.transforms import transforms

import sys
sys.path.append("./")

import utils
import models.builer as builder
import dataloader
import torchvision 

import os
import random
#fix seed
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# fix seed in numpy
import numpy as np
np.random.seed(1)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('--arch', default='vgg16', type=str, 
                        help='backbone architechture')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--val_list', type=str)              
    parser.add_argument('--mixup', default=None, type=str)              
    
    args = parser.parse_args()

    args.parallel = 0
    if args.mixup:
        args.batch_size = 2
    else: args.batch_size = 1
    args.workers = 0

    return args

def main(args):
    print('=> torch version : {}'.format(torch.__version__))

    utils.init_seeds(1, cuda_deterministic=False)

    print('=> modeling the network ...')
    model = builder.BuildAutoEncoder(args)     
    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))

    print('=> loading pth from {} ...'.format(args.resume))
    utils.load_dict(args.resume, model)
    
    print('=> building the dataloader ...')
    train_loader = dataloader.val_loader(args)

    total_len= len(train_loader.dataset)

    plt.figure(figsize=(16, 9))

    model.eval()
    print('=> reconstructing ...')
    with torch.no_grad():
        results=[]
        for i, (input, target) in enumerate(train_loader):

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
            if args.mixup:
                output = model(input,mixup=True)
            else:
                output = model(input)
                input = transforms.ToPILImage()(input.squeeze().cpu())


            if args.mixup:
                output = output.squeeze().cpu()
                results.append(output)

            else: 
                output = transforms.ToPILImage()(output.squeeze().cpu())
                plt.subplot(8,16,2*i+1, xticks=[], yticks=[])
                plt.imshow(input)

                plt.subplot(8,16,2*i+2, xticks=[], yticks=[])
                plt.imshow(output)

            if i == 63:
                break
    if args.mixup:
        for i,r in enumerate(results):
            grid=torchvision.utils.make_grid(r, nrow=11, padding=1, normalize=True, range=None, scale_each=False, pad_value=0)
            # save
            torchvision.utils.save_image(grid, 'figs/reconstruction{}.jpg'.format(i))        
    else:plt.savefig('figs/reconstruction.jpg')


if __name__ == '__main__':

    args = get_args()

    main(args)


