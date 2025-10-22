#!/bin/bash

# 训练脚本 - 在平衡数据集上训练不同模型

echo "=========================================="
echo "CIFAR-ViT Training Script"
echo "=========================================="

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 创建必要的目录
mkdir -p experiments/results/logs
mkdir -p experiments/results/checkpoints
mkdir -p experiments/results/figures

# 检查数据是否下载
if [ ! -d "data/cifar10" ]; then
    echo "Downloading CIFAR-10 dataset..."
    python data/download_data.py
fi

# 训练 ViT-Base
echo ""
echo "Training ViT-Base on balanced dataset..."
python main.py train --config experiments/configs/vit_base.yaml

# 训练 Dynamic ViT
echo ""
echo "Training Dynamic ViT on balanced dataset..."
python main.py train --config experiments/configs/dynamic_vit.yaml

# 训练 Lightweight ViT
echo ""
echo "Training Lightweight ViT on balanced dataset..."
python main.py train --config experiments/configs/avit.yaml

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="