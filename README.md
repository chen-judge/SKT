# Efficient Crowd Counting via Structured Knowledge Transfer (ACM MM 2020)

Paper: [Arxiv](https://arxiv.org/abs/2003.10120)

For distilling CSRNet, we train a teacher model and follow some code with [CSRNet-pytorch](https://github.com/leeyeehoo/CSRNet-pytorch)

## Datasets
ShanghaiTech: [Google Drive](https://drive.google.com/open?id=16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI)

UCF-QNRF: [Link](https://www.crcv.ucf.edu/data/ucf-qnrf/)

## Prerequisites
We strongly recommend Anaconda as the environment.

Python: 2.7

PyTorch: 0.4.0

## Preprocessing

### Ground-truth generation

```
# ShanghaiTech
python preprocess/ShanghaiTech_GT_generation.py
```

```
# UCF-QNRF
python preprocess/UCF_GT_generation.py --mode train
python preprocess/UCF_GT_generation.py --mode test
```

### Make data path files

Edit this file to change the path to your original datasets.

```
python preprocess/make_json.py
```


## Training

Edit this file for distillation training

```
bash SKT_distill.sh
```

## Testing

Edit this file for testing models
```
bash test.sh
```

## Models
Our models are [here](https://drive.google.com/drive/folders/17oxen8sNHtumcFL8hu9Z0Owuc6dWD8zV?usp=sharing)
