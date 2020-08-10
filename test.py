import sys
import os

import warnings

from models.model_vgg import CSRNet as CSRNet_vgg
from models.model_student_vgg import CSRNet as CSRNet_student

from utils import save_checkpoint
from utils import cal_para, crop_img_patches

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import json

import numpy as np
import argparse
import json
import dataset
import time

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')
parser.add_argument('--dataset', '-d', default='Shanghai', type=str,
                    help='Shanghai/UCF')
parser.add_argument('--checkpoint', '-c', metavar='CHECKPOINT', default=None, type=str,
                    help='path to the checkpoint')
parser.add_argument('--version', '-v', default=None, type=str,
                    help='vgg/quarter_vgg')
parser.add_argument('--transform', '-t', default=True, type=str,
                    help='1x1 conv transform')
parser.add_argument('--batch', default=1, type=int,
                    help='batch size')
parser.add_argument('--gpu', metavar='GPU', default='0', type=str,
                    help='GPU id to use.')

args = parser.parse_args()


def main():
    global args, best_prec1

    args.batch_size = 1
    args.workers = 4
    args.seed = time.time()
    if args.transform == 'false':
        args.transform = False

    with open(args.test_json, 'r') as outfile:
        test_list = json.load(outfile)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)

    if args.version == 'vgg':
        print 'VGG'
        model = CSRNet_vgg(pretrained=False)
        print model
        cal_para(model)

    elif args.version == 'quarter_vgg':
        print 'quarter_VGG'
        model = CSRNet_student(ratio=4, transform=args.transform)
        print model
        cal_para(model)  # including 1x1conv transform layer that can be removed
    else:
        raise NotImplementedError()

    model = model.cuda()

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)

            if args.transform is False:
                # remove 1x1 conv para
                for k in checkpoint['state_dict'].keys():
                    if k[:9] == 'transform':
                        del checkpoint['state_dict'][k]

            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.checkpoint, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.checkpoint))

    if args.dataset == 'UCF':
        test_ucf(test_list, model)
    else:
        test(test_list, model)


def test(test_list, model):
    print('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(test_list,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            train=False),
        shuffle=False,
        batch_size=args.batch_size)

    model.eval()

    mae = 0
    mse = 0

    for i, (img, target) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        with torch.no_grad():
            output = model(img)

        mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())
        mse += (output.data.sum() - target.sum().type(torch.FloatTensor).cuda()).pow(2)

    N = len(test_loader)
    mae = mae / N
    mse = torch.sqrt(mse / N)
    print(' * MAE {mae:.3f} \t    * MSE {mse:.3f}'
          .format(mae=mae, mse=mse))


def test_ucf(test_list, model):
    print 'begin test'
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(test_list,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            train=False,
                            dataset='ucf_test',
                            ),
        shuffle=False,
        batch_size=1)

    model.eval()

    mae = 0
    mse = 0

    for i, (img, target) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)

        people = 0
        img_patches = crop_img_patches(img, size=512)
        for patch in img_patches:
            with torch.no_grad():
                sub_output = model(patch)
            people += sub_output.data.sum()

        error = people - target.sum().type(torch.FloatTensor).cuda()
        mae += abs(error)
        mse += error.pow(2)

    N = len(test_loader)
    mae = mae / N
    mse = torch.sqrt(mse / N)
    print(' * MAE {mae:.3f} \t    * MSE {mse:.3f}'
          .format(mae=mae, mse=mse))

    return mae, mse


if __name__ == '__main__':
    main()