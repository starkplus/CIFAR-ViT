# 参考文献

## 核心论文

### Vision Transformer

1. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**
   - Authors: Alexey Dosovitskiy, et al.
   - Conference: ICLR 2021
   - Link: https://arxiv.org/abs/2010.11929
   - 贡献: 首次将 Transformer 成功应用于图像分类

2. **Training data-efficient image transformers & distillation through attention**
   - Authors: Hugo Touvron, et al.
   - Conference: ICML 2021
   - Link: https://arxiv.org/abs/2012.12877
   - 贡献: DeiT，提出数据高效的 ViT 训练方法

### 模型轻量化

3. **DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification**
   - Authors: Yongming Rao, et al.
   - Conference: NeurIPS 2021
   - Link: https://arxiv.org/abs/2106.02034
   - 贡献: 动态 token 剪枝，减少计算量

4. **Efficient Vision Transformers with Spatial Reduction Attention**
   - Authors: Wenhai Wang, et al.
   - Conference: ICCV 2021
   - Link: https://arxiv.org/abs/2106.13797
   - 贡献: 空间降维注意力机制

5. **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention**
   - Authors: Angelos Katharopoulos, et al.
   - Conference: ICML 2020
   - Link: https://arxiv.org/abs/2006.16236
   - 贡献: 线性复杂度的注意力机制

6. **You Only Need Less Attention at Each Stage in Vision Transformers**
   - Authors: Tianyu Zhang, et al.
   - Conference: CVPR 2024
   - Link: https://arxiv.org/abs/2406.00427
   - 贡献: 渐进式降低注意力复杂度

### 数据增强

7. **mixup: Beyond Empirical Risk Minimization**
   - Authors: Hongyi Zhang, et al.
   - Conference: ICLR 2018
   - Link: https://arxiv.org/abs/1710.09412
   - 贡献: MixUp 数据增强方法

8. **CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features**
   - Authors: Sangdoo Yun, et al.
   - Conference: ICCV 2019
   - Link: https://arxiv.org/abs/1905.04899
   - 贡献: CutMix 数据增强方法

9. **Improved Regularization of Convolutional Neural Networks with Cutout**
   - Authors: Terrance DeVries, Graham W. Taylor
   - Conference: arXiv 2017
   - Link: https://arxiv.org/abs/1708.04552
   - 贡献: Cutout 正则化方法

### 类别不平衡

10. **Focal Loss for Dense Object Detection**
    - Authors: Tsung-Yi Lin, et al.
    - Conference: ICCV 2017
    - Link: https://arxiv.org/abs/1708.02002
    - 贡献: Focal Loss 处理类别不平衡

11. **Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss**
    - Authors: Kaidi Cao, et al.
    - Conference: NeurIPS 2019
    - Link: https://arxiv.org/abs/1906.07413
    - 贡献: LDAM 损失函数

## 相关工作

### Transformer 架构

12. **Attention is All You Need**
    - Authors: Ashish Vaswani, et al.
    - Conference: NeurIPS 2017
    - Link: https://arxiv.org/abs/1706.03762
    - 贡献: 提出 Transformer 架构

13. **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**
    - Authors: Ze Liu, et al.
    - Conference: ICCV 2021
    - Link: https://arxiv.org/abs/2103.14030
    - 贡献: 层次化 Vision Transformer

### 优化技术

14. **Layer Normalization**
    - Authors: Jimmy Lei Ba, et al.
    - Conference: arXiv 2016
    - Link: https://arxiv.org/abs/1607.06450
    - 贡献: Layer Normalization 技术

15. **Decoupled Weight Decay Regularization**
    - Authors: Ilya Loshchilov, Frank Hutter
    - Conference: ICLR 2019
    - Link: https://arxiv.org/abs/1711.05101
    - 贡献: AdamW 优化器

16. **SGDR: Stochastic Gradient Descent with Warm Restarts**
    - Authors: Ilya Loshchilov, Frank Hutter
    - Conference: ICLR 2017
    - Link: https://arxiv.org/abs/1608.03983
    - 贡献: 余弦退火学习率调度

## 数据集

17. **Learning Multiple Layers of Features from Tiny Images**
    - Authors: Alex Krizhevsky
    - Year: 2009
    - Link: https://www.cs.toronto.edu/~kriz/cifar.html
    - 贡献: CIFAR-10 和 CIFAR-100 数据集

## 实现参考

### 开源项目

1. **timm (PyTorch Image Models)**
   - GitHub: https://github.com/rwightman/pytorch-image-models
   - 贡献: 大量预训练模型和训练代码

2. **vit-pytorch**
   - GitHub: https://github.com/lucidrains/vit-pytorch
   - 贡献: 简洁的 ViT 实现

3. **DynamicViT Official Implementation**
   - GitHub: https://github.com/raoyongming/DynamicViT
   - 贡献: Dynamic ViT 官方实现

### 教程和博客

4. **The Illustrated Transformer**
   - Link: https://jalammar.github.io/illustrated-transformer/
   - 贡献: Transformer 可视化教程

5. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Paper Explained)**
   - Link: https://www.youtube.com/watch?v=TrdevFK_am4
   - 贡献: ViT 论文解读视频

## 工具和库

### 深度学习框架

1. **PyTorch**
   - Website: https://pytorch.org/
   - Version: 2.0+
   - 用途: 主要深度学习框架

2. **torchvision**
   - GitHub: https://github.com/pytorch/vision
   - 用途: 计算机视觉工具库

### 可视化和分析

3. **TensorBoard**
   - Website: https://www.tensorflow.org/tensorboard
   - 用途: 训练可视化

4. **matplotlib**
   - Website: https://matplotlib.org/
   - 用途: 绘图库

5. **seaborn**
   - Website: https://seaborn.pydata.org/
   - 用途: 统计可视化

### 性能分析

6. **thop (PyTorch-OpCounter)**
   - GitHub: https://github.com/Lyken17/pytorch-OpCounter
   - 用途: 计算 FLOPs 和参数量

7. **torchsummary**
   - GitHub: https://github.com/sksq96/pytorch-summary
   - 用途: 模型摘要

## 课程资源

1. **CS231n: Convolutional Neural Networks for Visual Recognition**
   - Stanford University
   - Link: http://cs231n.stanford.edu/

2. **Deep Learning Specialization**
   - Coursera by Andrew Ng
   - Link: https://www.coursera.org/specializations/deep-learning

## 书籍

1. **Deep Learning**
   - Authors: Ian Goodfellow, Yoshua Bengio, Aaron Courville
   - Publisher: MIT Press, 2016
   - Link: https://www.deeplearningbook.org/

2. **Dive into Deep Learning**
   - Authors: Aston Zhang, et al.
   - Year: 2023
   - Link: https://d2l.ai/

---

**整理日期**: 2025-10-22  
**贡献者**: Haonan Wang