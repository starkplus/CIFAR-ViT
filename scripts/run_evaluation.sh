#!/bin/bash

# 评估脚本 - 评估训练好的模型

echo "=========================================="
echo "CIFAR-ViT Evaluation Script"
echo "=========================================="

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 评估 ViT-Base
echo ""
echo "Evaluating ViT-Base..."
if [ -f "experiments/results/checkpoints/vit_base/best_model.pth" ]; then
    python main.py evaluate \
        --config experiments/configs/vit_base.yaml \
        --checkpoint experiments/results/checkpoints/vit_base/best_model.pth
else
    echo "ViT-Base checkpoint not found!"
fi

# 评估 Dynamic ViT
echo ""
echo "Evaluating Dynamic ViT..."
if [ -f "experiments/results/checkpoints/dynamic_vit/best_model.pth" ]; then
    python main.py evaluate \
        --config experiments/configs/dynamic_vit.yaml \
        --checkpoint experiments/results/checkpoints/dynamic_vit/best_model.pth
else
    echo "Dynamic ViT checkpoint not found!"
fi

# 评估 Lightweight ViT
echo ""
echo "Evaluating Lightweight ViT..."
if [ -f "experiments/results/checkpoints/lightweight_vit/best_model.pth" ]; then
    python main.py evaluate \
        --config experiments/configs/avit.yaml \
        --checkpoint experiments/results/checkpoints/lightweight_vit/best_model.pth
else
    echo "Lightweight ViT checkpoint not found!"
fi

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="