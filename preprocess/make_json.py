"""
Make json files for dataset
"""
import json
import os


def get_val(root):
    """
    Validation set follows part_A_val.json in CSRNet
    https://github.com/leeyeehoo/CSRNet-pytorch
    """
    with open("preprocess/part_A_val.json") as f:
        val_list = json.load(f)
    new_val = []
    for item in val_list:
        new_item = item.replace('/home/leeyh/Downloads/Shanghai/', root)
        new_val.append(new_item)
    with open('A_val.json', 'w') as f:
        json.dump(new_val, f)


def get_train(root):
    path = os.path.join(root, 'part_A_final', 'train_data', 'images')
    filenames = os.listdir(path)
    pathname = [os.path.join(path, filename) for filename in filenames]
    with open('A_train.json', 'w') as f:
        json.dump(pathname, f)


def get_test(root):
    path = os.path.join(root, 'part_A_final', 'test_data', 'images')
    filenames = os.listdir(path)
    pathname = [os.path.join(path, filename) for filename in filenames]
    with open('A_test.json', 'w') as f:
        json.dump(pathname, f)


if __name__ == '__main__':
    root = '/media/firstPartition/cjq/ShanghaiTech/'  # Dataset path
    get_train(root)
    get_val(root)
    get_test(root)
    print 'Finish!'

