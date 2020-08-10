import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import os
import time


def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            

def save_checkpoint(state, mae_is_best, mse_is_best, path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(path, filename))
    epoch = state['epoch']
    if mae_is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'epoch'+str(epoch)+'_best_mae.pth.tar'))
    if mse_is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'epoch'+str(epoch)+'_best_mse.pth.tar'))


def gram(x, y):
    n = x.shape[0]
    c1 = x.shape[1]
    c2 = y.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    x = x.view(n, c1, -1, 1)[0, :, :, 0]
    y = y.view(n, c2, -1, 1)[0, :, :, 0]
    y = y.transpose(0, 1)
    # print x.shape
    # print y.shape
    z = torch.mm(x, y) / (w*h)
    return z


def cal_dense_fsp(features):
    fsp = []
    for groups in features:
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                x = groups[i]
                y = groups[j]

                norm1 = nn.InstanceNorm2d(x.shape[1])
                norm2 = nn.InstanceNorm2d(y.shape[1])
                x = norm1(x)
                y = norm2(y)
                res = gram(x, y)
                fsp.append(res)
    return fsp


def cosine_similarity(stu_map, tea_map):
    similiar = 1-F.cosine_similarity(stu_map, tea_map, dim=1)
    loss = similiar.sum()
    return loss


def scale_process(features, scale=[3, 2, 1], ceil_mode=True):
    # process features for multi-scale dense fsp
    new_features = []
    for i in range(len(features)):
        if i >= len(scale):
            new_features.append(features[i])
            continue
        down_ratio = pow(2, scale[i])
        pool = nn.MaxPool2d(kernel_size=down_ratio, stride=down_ratio, ceil_mode=ceil_mode)
        new_features.append(pool(features[i]))
    return new_features


def cal_para(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        # print "stucture of layer: " + str(list(i.size()))
        for j in i.size():
            l *= j
        # print "para in this layer: " + str(l)
        k = k + l
    print("the amount of para: " + str(k))


def crop_img_patches(img, size=512):
    """ crop the test images to patches

    while testing UCF data, we load original images, then use crop_img_patches to crop the test images to patches,
    calculate the crowd count respectively and sum them together finally
    """
    w = img.shape[3]
    h = img.shape[2]
    x = int(w/size)+1
    y = int(h/size)+1
    crop_w = int(w/x)
    crop_h = int(h/y)
    patches = []
    for i in range(x):
        for j in range(y):
            start_x = crop_w*i
            if i == x-1:
                end_x = w
            else:
                end_x = crop_w*(i+1)

            start_y = crop_h*j
            if j == y - 1:
                end_y = h
            else:
                end_y = crop_h*(j+1)

            sub_img = img[:, :, start_y:end_y, start_x:end_x]
            patches.append(sub_img)
    return patches
