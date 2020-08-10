# Efficient Crowd Counting via Structured Knowledge Transfer (ACM MM 2020 Oral)

Crowd counting is an application-oriented task and its inference efficiency is crucial for real-world applications. However, most previous works relied on heavy backbone networks and required prohibitive run-time consumption, which would seriously restrict their deployment scopes and cause poor scalability. To liberate these crowd counting models, we propose a novel Structured Knowledge Transfer (SKT) framework, which fully exploits the structured knowledge of a well-trained teacher network to generate a lightweight but still highly effective student network. 

Extensive evaluations on three benchmarks well demonstrate the effectiveness of our SKT for extensive crowd counting models. In particular, only using around $6\%$ of the parameters and computation cost of original models, our distilled VGG-based models obtain at least 6.5$\times$ speed-up on an Nvidia 1080 GPU and even achieve state-of-the-art performance.
  
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

1. Generation the ground-truth density maps for training

```
# ShanghaiTech
python preprocess/ShanghaiTech_GT_generation.py
```

```
# UCF-QNRF
python preprocess/UCF_GT_generation.py --mode train
python preprocess/UCF_GT_generation.py --mode test
```

2. Make data path files and edit this file to change the path to your original datasets.

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
