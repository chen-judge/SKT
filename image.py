import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
import time


def load_data(img_path,train=True, dataset='shanghai'):
    """ Load data

    Use crop_ratio between 0.5 and 1.0 for random crop
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    if train:
        if dataset == 'shanghai':
            crop_ratio = random.uniform(0.5, 1.0)
            crop_size = (int(crop_ratio*img.size[0]), int(crop_ratio*img.size[1]))
            dx = int(random.random() * (img.size[0]-crop_size[0]))
            dy = int(random.random() * (img.size[1]-crop_size[1]))

            img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
            target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]

        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    target = reshape_target(target, 3)
    target = np.expand_dims(target, axis=0)

    img = img.copy()
    target = target.copy()
    return img, target


def load_ucf_ori_data(img_path):
    """ Load original UCF-QNRF data for testing

    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    return img, target


def reshape_target(target, down_sample=3):
    """ Down sample GT to 1/8

    """
    height = target.shape[0]
    width = target.shape[1]

    # ceil_mode=True for nn.MaxPool2d in model
    for i in range(down_sample):
        height = int((height+1)/2)
        width = int((width+1)/2)
        # height = int(height/2)
        # width = int(width/2)

    target = cv2.resize(target, (width, height), interpolation=cv2.INTER_CUBIC) * (2**(down_sample*2))
    return target
