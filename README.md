# CIFAR-ViT: Vision Transformer 图像分类实验

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## 📋 项目概述

本项目实现了基于 PyTorch 的 Vision Transformer (ViT)，在 CIFAR-10 数据集上进行图像分类任务。项目包含标准 ViT 以及两种轻量化变体模型，并对类别不平衡问题提供了解决方案。

### 实验要求

根据《深度学习与计算机视觉》课程实验要求：

1. ✅ 实现标准 Vision Transformer 模型
2. ✅ 在 CIFAR-10 数据集上训练和评估
3. ✅ 处理类别不平衡数据集（CIFAR10_imbalanced）
4. ✅ 实现至少2种 ViT 轻量化方法降低模型复杂度
5. ✅ 进行模型复杂度分析（参数量、FLOPs）
6. ✅ 提供完整的实验报告和代码

## 🎯 主要特性

- **多种 ViT 模型**: 标准 ViT-Base、ViT-Small、Dynamic ViT、Lightweight ViT
- **类别不平衡处理**: 支持加权采样和损失函数类别权重
- **数据增强**: MixUp、CutMix、Cutout 等多种增强策略
- **模型轻量化**: 
  - Dynamic ViT: 动态 token 剪枝
  - Lightweight ViT: 高效线性注意力机制
- **完整工具链**: 训练、评估、可视化、复杂度分析
- **实验管理**: TensorBoard 日志、模型检查点、配置文件管理

## 📁 项目结构


 CIFAR-ViT/
 ├── data/ # 数据目录
 │ ├── cifar10/ # CIFAR-10 数据集
 │ └── download_data.py # 数据下载脚本
 ├── models/ # 模型定义
 │ ├── vit.py # 标准 ViT
 │ ├── vit_variants.py # ViT 变体模型
 │ ├── patch_embedding.py # Patch 嵌入层
 │ ├── attention.py # 注意力机制
 │ └── transformer_block.py # Transformer 块
 ├── utils/ # 工具函数
 │ ├── data_loader.py # 数据加载
 │ ├── augmentation.py # 数据增强
 │ ├── metrics.py # 评估指标
 │ └── visualization.py # 可视化工具
 ├── train/ # 训练相关
 │ ├── trainer.py # 训练器
 │ ├── train_balanced.py # 平衡数据训练
 │ └── train_imbalanced.py # 不平衡数据训练
 ├── evaluation/ # 评估相关
 │ ├── evaluate.py # 模型评估
 │ └── complexity_analysis.py # 复杂度分析
 ├── experiments/ # 实验配置和结果
 │ ├── configs/ # 配置文件
 │ └── results/ # 实验结果
 ├── scripts/ # 运行脚本
 │ ├── run_training.sh # 训练脚本
 │ ├── run_evaluation.sh # 评估脚本
 │ └── export_results.sh # 结果导出
 ├── docs/ # 文档
 ├── tests/ # 单元测试
 ├── main.py # 主程序入口
 ├── requirements.txt # 依赖包
 └── README.md # 项目说明
 text

## 🚀 快速开始

### 1. 环境配置


 克隆项目
 git clone https://github.com/yourusername/CIFAR-ViT.git
 cd CIFAR-ViT
 创建虚拟环境（推荐）
 conda create -n cifar-vit python=3.8
conda 创建-n cifar-vit python=3.8
 conda activate cifar-vit  conda 激活 cifar-vit
 安装依赖
 pip install -r requirements.txt
pip 安装 -r 要求.txt
 text

### 2. 数据准备


 下载 CIFAR-10 数据集并创建不平衡版本
 python data/download_data.py
python 数据/download_data.py
 text

### 3. 训练模型

#### 方式一：使用主程序


 训练 ViT-Base（平衡数据集）
 python main.py train --config experiments/configs/vit_base.yaml
python main.py train --config 实验/configs/vit_base.yaml
 训练 ViT-Base（不平衡数据集）
 python main.py train --config experiments/configs/vit_base.yaml --imbalanced
 训练 Dynamic ViT
 python main.py train --config experiments/configs/dynamic_vit.yaml
python main.py train --config 实验/configs/dynamic_vit.yaml
 训练 Lightweight ViT
 python main.py train --config experiments/configs/avit.yaml
python main.py train --config 实验/configs/avit.yaml
 text

#### 方式二：使用训练脚本


 在平衡数据集上训练
 python train/train_balanced.py --config experiments/configs/vit_base.yaml
python train/train_balanced.py --config 实验/configs/vit_base.yaml
 在不平衡数据集上训练
 python train/train_imbalanced.py --config experiments/configs/vit_base.yaml
python train/train_imbalanced.py --config 实验/configs/vit_base.yaml
 text

#### 方式三：批量训练（使用shell脚本）


 chmod +x scripts/run_training.sh
chmod +x 脚本/run_training.sh
 ./scripts/run_training.sh
 text

### 4. 评估模型


 评估训练好的模型
 python main.py evaluate   python main.py 评估
 --config experiments/configs/vit_base.yaml 
--配置实验/configs/vit_base.yaml
 --checkpoint experiments/results/checkpoints/vit_base/best_model.pth
--检查点实验/结果/检查点/vit_base/best_model.pth
 或使用评估脚本
 chmod +x scripts/run_evaluation.sh
chmod +x 脚本/run_evaluation.sh
 ./scripts/run_evaluation.sh
 text

### 5. 复杂度分析


 分析所有模型
 python main.py analyze --config experiments/configs/vit_base.yaml
python main.py 分析 --config 实验/configs/vit_base.yaml
 分析指定模型
 python main.py analyze --config experiments/configs/vit_base.yaml 
python main.py 分析 --config 实验/configs/vit_base.yaml
 --models vit_base dynamic_vit
 text

## 📊 模型对比

| 模型 | 参数量 | FLOPs | 准确率 | 推理时间 |
|------|--------|-------|--------|----------|
| ViT-Base | 8.5M | 1.2G | ~85% | 15ms |
| ViT-Small | 6.0M | 0.8G | ~83% | 12ms |
| Dynamic ViT | 5.2M | 0.6G | ~84% | 10ms |
| Lightweight ViT | 4.8M | 0.5G | ~82% | 8ms |

*注：实际性能需要在你的硬件上测试*

## 🔧 配置说明

配置文件位于 `experiments/configs/` 目录下，主要参数包括：


 模型配置
 model_type: 'vit_base' # 模型类型
 img_size: 32 # 图像大小
 embed_dim: 512 # 嵌入维度
 depth: 6 # Transformer 层数
 num_heads: 8 # 注意力头数
 训练配置
 epochs: 100 # 训练轮数
 batch_size: 128 # 批次大小
 learning_rate: 0.0003 # 学习率
 optimizer: 'adamw' # 优化器
 数据增强
 use_mixup: true # 是否使用 MixUp
 use_cutmix: true # 是否使用 CutMix
 不平衡数据处理
 use_class_weights: true # 使用类别权重
 use_weighted_sampler: false # 使用加权采样
 text

## 📈 实验结果

### 训练曲线

训练过程中的损失和准确率曲线会自动保存到 `experiments/results/checkpoints/` 目录。

### 评估指标

- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1 分数 (F1-Score)
- 混淆矩阵 (Confusion Matrix)

### 可视化

所有可视化结果（训练曲线、混淆矩阵、注意力图等）会保存到 `experiments/results/figures/` 目录。

## 🎓 实验报告

详细的实验报告请参考 `docs/experiment_report.md`，包含：

1. 实验背景和目的
2. 模型架构说明
3. 实验设置和配置
4. 实验结果和分析
5. 结论和改进方向

## 💡 使用技巧

### 处理类别不平衡

项目提供两种方法处理类别不平衡：

1. **加权采样** (Weighted Sampling)

 use_weighted_sampler: true
使用加权采样器：true
 text

2. **类别权重损失** (Class-weighted Loss)

 use_class_weights: true  use_class_weights：true
 text

### 减少过拟合

- 使用数据增强：MixUp、CutMix、Cutout
- 调整 dropout 率
- 使用标签平滑 (Label Smoothing)
- 减小模型大小

### 提升训练速度

- 增加 batch size（如果显存允许）
- 使用混合精度训练（需要 PyTorch AMP）
- 使用多 GPU 训练
- 减少数据加载线程数

## 🐛 常见问题

### 1. CUDA out of memory

减小 batch size 或使用更小的模型：

 batch_size: 64 # 降低 batch size
 model_type: 'vit_small' # 使用更小的模型
 text

### 2. 数据加载慢

调整数据加载器参数：

 num_workers: 8 # 增加工作线程
 pin_memory: true # 启用 pin memory
 text

### 3. 训练不收敛

- 降低学习率
- 使用学习率预热 (warmup)
- 检查数据增强是否过强
- 尝试不同的优化器

## 📚 参考文献

1. Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
2. Rao, Y., et al. (2021). "DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification." NeurIPS 2021.
3. Wang, S., et al. (2021). "Linformer: Self-Attention with Linear Complexity." arXiv:2006.04768.

## 👥 团队分工

- **Haonan Wang**: 模型实现、训练框架搭建、数据处理、评估指标实现、可视化、实验分析、文档编写


## 📝 License

本项目采用 MIT License - 详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题，请联系：nhao4968@gmail.com

---

<div align="center">
Made with ❤️ for Deep Learning and Computer Vision Course
</div>
