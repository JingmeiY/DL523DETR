# DL523DETR
EC523 Deep Learning Project - Object Detection

# Swin Transformer and DETR
This repo contains the reimplemented code for Swin Transformer, DETR as well as a proposed model that applies Swin Transformer to replace the original Resnet50 backbone in DETR to reproduce object detection results of [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf). It is based on the repos (https://github.com/facebookresearch/detr, https://github.com/microsoft/Swin-Transformer/blob/ma
in/models/swin_transformer.py).

# Structure of Swin T
The reimplemented code structure is the same as the framework exhibited on paper [Swin Transformer Hierarchical Vision Transformer using ShiftedWindows] figure 3, which first splits an input RGB image into non-overlapping patches and passes several Transformer blocks with modified self-attention
computation.

# Prerequisites
Linux or macOS 

Python 3.7+

PyTorch 1.11

CUDA 11.3

GCC 5+

# Dataset preparations
Download necessary datasets from https://cocodataset.org/#download and http://shuoyang1213.me/WIDERFACE/.

# Training and Test
To train DETR on a single node with 4 gpus for 10 epochs run:
```python  
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --coco_path ../coco/images --output_dir output 
```
Because training is extremely costing time, we use checkpoint.
```python
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --coco_path ../coco/images --output_dir output --start_epoch 3 --resume output/checkpoint.pth --epochs 10
```


# Results
We trained and tested the model on the WiDER FACE dataset which includes 32203 images and labels 393703 faces as well as the mini-COCO training dataset includes about 25Kimages, roughly 20 \% of the COCO 2017 training set.
```python
python test2.py --coco_path ../coco/images/test2017 --resume output/checkpoint_9.pth
```
5 epochs:
![show_epc3](https://user-images.githubusercontent.com/87682737/162815547-7dd5c4cb-4b54-4e53-ba44-014905d7e7aa.png)
10 epochs:
![show_epc9](https://user-images.githubusercontent.com/87682737/162815659-0928d48d-e1a9-437f-a61f-c6509af304ee.png)
