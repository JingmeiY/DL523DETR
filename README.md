# DL523DETR
EC523 Deep Learning Project - Object Detection

# Swin Transformer and DETR
This repo contains the reimplemented code for DETR and Swin Transformer to reproduce object detection results of [Object DEtection with TRansformers](https://arxiv.org/abs/2005.12872). It is based on the repos (https://github.com/facebookresearch/detr, https://github.com/microsoft/Swin-Transformer/blob/ma
in/models/swin_transformer.py). We reimplemented the code files that can capture the key features of these two frameworks, including swin_transformer.py, detr.py, transformer.py, etc. Other files like backbones.py are from the original github source https://github.com/facebookresearch/detr.

# Structure of DETR
The three main components in this scheme are a CNN backbone, an encoder-decoder Transformer, and a feed-forward network (FNN). It is necessary to have a loss that can uniquely assign the predicted boxes to ground-truth boxes and an architecture that can predict a set of objects and model their relation in a single pass. This design utilizes the backbone to learn a 2D representation of an input image. We used pre-trained resnet18, resnet34, resnet50, and resnet101 as our backbones for comparison. A positional encoding was appended to the learned representation before passing it into the encoder. Each encoder layer contains a multi-head self-attention module and a feed-forward network. The decoder follows the same structure as the one used in the standard transformer.  Compared to a classical transformer, the transformer incorporated in this architecture can work in parallel. However, the standard transformer uses an autoregressive model, which predicts the output sequence one element at a time. In this case, the computation is significantly improved. 


# Structure of Swin T
The reimplemented code structure is the same as the framework exhibited on paper [Swin Transformer Hierarchical Vision Transformer using ShiftedWindows](https://arxiv.org/abs/2103.14030). The shifted windowing scheme increases efficiency by limiting self-attention computation to non-overlapping local windows while allowing for cross-window connection. It starts from small-sized patches and gradually merges neighboring patches into deeper Transformer layers to construct a hierarchical representation. This design has linear computational complexity by computing self-attention locally within non-overlapping windows. The overall structure is shown in [Swin Transformer Hierarchical Vision Transformer using ShiftedWindows](https://arxiv.org/abs/2103.14030) Figure 3.

# Prerequisites
Linux or macOS 

Python 3.7+

PyTorch 1.11

CUDA 11.3

GCC 5+

# Dataset preparations
Download necessary datasets from https://cocodataset.org/#download. (WIDER FACE dataset http://shuoyang1213.me/WIDERFACE/ can also be used.)

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
In this project, we used the same hyperparameters as in [Object DEtection with TRansformers](https://arxiv.org/abs/2005.12872). DETR is trained using AdamW with weight decay handling, and gradient clipping is applied with a maximal gradient norm. The learning rate of the backbone is set to be 10-5 since it can stabilize the training process, while the learning rate for the transformer is 10-4.

```python
python test2.py --coco_path ../coco/images/test2017 --resume output/checkpoint_9.pth
```
Figure below shows the performance of DETR. These three images from top to bottom are output by the same model with three different epochs: 5, 15, and 20 using ResNet50 backbone.
![show_epc3](https://github.com/JingmeiY/DL523DETR/blob/main/results%20images/1.png)
We can also observed that the complexity of images influences the detection performance. Figure below gives an example.
![show_epc9](https://github.com/JingmeiY/DL523DETR/blob/main/results%20images/2.png)
![show_epc9](https://github.com/JingmeiY/DL523DETR/blob/main/results%20images/3.png)
