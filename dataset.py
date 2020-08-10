import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
import torchvision.transforms.functional as F


class listDataset(Dataset):
    def __init__(self, root, shape=None, transform=None,  train=False, seen=0,
                 batch_size=1, num_workers=20, dataset='shanghai'):
        if train and dataset == 'shanghai':
            root = root*4
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = dataset

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]

        if self.dataset == 'ucf_test':
            # test in UCF
            img, target = load_ucf_ori_data(img_path)
        else:
            img, target = load_data(img_path, self.train, self.dataset)

        if self.transform is not None:
            img = self.transform(img)
        return img, target