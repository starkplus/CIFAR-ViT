#!/bin/bash

# 结果导出脚本 - 收集所有实验结果

echo "=========================================="
echo "CIFAR-ViT Results Export Script"
echo "=========================================="

# 创建结果汇总目录
EXPORT_DIR="experiments/results/summary_$(date +%Y%m%d_%H%M%S)"
mkdir -p $EXPORT_DIR

echo "Exporting results to: $EXPORT_DIR"

# 复制训练曲线
echo "Copying training curves..."
find experiments/results/checkpoints -name "training_curves*.png" -exec cp {} $EXPORT_DIR \;

# 复制混淆矩阵
echo "Copying confusion matrices..."
find experiments/results -name "confusion_matrix*.png" -exec cp {} $EXPORT_DIR \;

# 复制评估结果
echo "Copying evaluation results..."
find experiments/results -name "evaluation_results*.txt" -exec cp {} $EXPORT_DIR \;

# 复制复杂度分析
echo "Copying complexity analysis..."
find experiments/results -name "complexity_analysis*.txt" -exec cp {} $EXPORT_DIR \;

# 运行复杂度分析
echo ""
echo "Running complexity analysis..."
python main.py analyze --config experiments/configs/vit_base.yaml

# 复制到导出目录
cp experiments/results/complexity_analysis.txt $EXPORT_DIR/ 2>/dev/null

# 生成汇总报告
echo ""
echo "Generating summary report..."
cat > $EXPORT_DIR/README.md << EOF
# CIFAR-ViT Experiment Results

## 实验日期
$(date +"%Y-%m-%d %H:%M:%S")

## 文件说明

### 训练曲线
- \`training_curves*.png\`: 各模型的训练和验证损失/准确率曲线

### 评估结果
- \`evaluation_results*.txt\`: 各模型在测试集上的详细评估指标
- \`confusion_matrix*.png\`: 混淆矩阵可视化

### 复杂度分析
- \`complexity_analysis.txt\`: 各模型的参数量、FLOPs、推理速度对比

## 模型列表
1. ViT-Base: 标准Vision Transformer
2. Dynamic ViT: 动态token剪枝版本
3. Lightweight ViT: 使用高效注意力机制的轻量版本

## 实验配置
- 数据集: CIFAR-10
- 图像大小: 32x32
- Batch Size: 128
- 训练Epochs: 100
- 优化器: AdamW

EOF

echo ""
echo "=========================================="
echo "Results exported to: $EXPORT_DIR"
echo "=========================================="