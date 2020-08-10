import h5py
import scipy.io as io
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import time
import argparse

parser = argparse.ArgumentParser(description='UCF generation')
parser.add_argument('--mode', '-m', type=str,
                    help='train/test set')
parser.add_argument('--start', '-s', type=int, default=None,
                    help='The starting image id')
parser.add_argument('--end', '-e', type=int, default=None,
                    help='The ending image id')
global args
args = parser.parse_args()


def gaussian_filter_density(gt, threshold=30):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)  # nonzero value represent people in labels
    if gt_count == 0:  # gt_count is the amount of people
        return density

    pts = np.array(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))  # human label position
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=2)  # min nearest neighbor

    print 'generate density...'
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = distances[i][1]
            sigma = min(sigma, threshold)  # nearest
        else:
            sigma = np.average(np.array(gt.shape))/2./2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density


root = '/media/firstPartition/cjq/UCF-QNRF-test'

train_path = os.path.join(root, 'Train')
test_path = os.path.join(root, 'Test')

if args.mode == 'train':
    path = train_path
else:
    path = test_path

paths = glob.glob(os.path.join(path, '*.jpg'))
paths.sort()
if args.start and args.end:
    processed_imgs = paths[args.start:args.end]  # It will take a long time and can be processed in parts
else:
    processed_imgs = paths
# print processed_imgs

for img_path in processed_imgs:
    start = time.time()
    img = plt.imread(img_path)
    (name, _) = os.path.splitext(img_path)
    mat = io.loadmat(name+'_ann.mat')
    gt = mat['annPoints']

    k = np.zeros((img.shape[0], img.shape[1]))
    print 'GT len & shape: ', len(gt), img.shape, '  img path: ', img_path
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    k = gaussian_filter_density(k)
    # save the Density Maps GT as h5 format
    with h5py.File(img_path.replace('.jpg', '.h5'), 'w') as hf:
            hf['density'] = k
    print 'time: ', time.time() - start

