# CIFAR-ViT 使用指南

## 快速开始流程

### 第一步：环境准备

```


# 1. 安装依赖

pip install -r requirements.txt

# 2. 验证安装

python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"

```

### 第二步：数据准备

```


# 下载 CIFAR-10 并创建不平衡版本

python data/download_data.py

```

这会自动：
- 下载 CIFAR-10 数据集到 `data/cifar10/`
- 创建不平衡索引文件 `data/imbalanced_indices.npy`

### 第三步：训练模型

#### 使用主程序

```


# 训练标准 ViT-Base

python main.py train --config experiments/configs/vit_base.yaml

# 训练 Dynamic ViT

python main.py train --config experiments/configs/dynamic_vit.yaml

# 训练 Lightweight ViT

python main.py train --config experiments/configs/avit.yaml

# 训练 imbalanced ViT

python main.py train --config experiments/configs/vit_base_imbalanced.yaml --imbalanced

# 训练 imbalanced_wight ViT

python main.py train --config experiments/configs/vit_base_imbalanced_weighted.yaml --imbalanced

# 训练 imbalanced_sampler ViT

python main.py train --config experiments/configs/vit_base_imbalanced_sampler.yaml --imbalanced

# 训练 imbalanced_weighted_sampler ViT

python main.py train --config experiments/configs/vit_base_imbalanced_weighted_sampler.yaml --imbalanced
```




### 第四步：监控训练

```


# 启动 TensorBoard

tensorboard --logdir experiments/results/logs

# 在浏览器中打开

# http://localhost:6006

```

### 第五步：评估模型

```


# 评估训练好的模型

python main.py evaluate \
--config experiments/configs/vit_base.yaml \
--checkpoint experiments/results/checkpoints/vit_base/best_model.pth

```

### 第六步：复杂度分析

```


# 分析所有模型

python main.py analyze --config experiments/configs/vit_base.yaml

# 分析特定模型

python main.py analyze \
--config experiments/configs/vit_base.yaml \
--models vit_base dynamic_vit

```

## 配置文件说明

### 修改训练参数

编辑 `experiments/configs/vit_base.yaml`:

```


# 调整学习率

learning_rate: 0.0001  \# 降低学习率

# 调整 batch size

batch_size: 64  \# 如果显存不足

# 调整训练轮数

epochs: 50  \# 快速实验

# 启用数据增强

use_mixup: true
use_cutmix: true

```

### 处理类别不平衡

```


# 方法 1: 使用加权采样

use_weighted_sampler: true

# 方法 2: 使用类别权重

use_class_weights: true

# 方法 3: 两者结合

use_weighted_sampler: true
use_class_weights: true

```

## 常用命令

### 测试模型

```


# 运行单元测试

python -m pytest tests/

# 测试特定模块

python tests/test_models.py
python tests/test_data_loader.py

```

### 批量实验

```


# 使用 shell 脚本批量训练

chmod +x scripts/run_training.sh
./scripts/run_training.sh

# 批量评估

chmod +x scripts/run_evaluation.sh
./scripts/run_evaluation.sh

# 导出结果

chmod +x scripts/export_results.sh
./scripts/export_results.sh

```

## 故障排除

### 问题 1: CUDA out of memory

**解决方案**：
```


# 降低 batch size

batch_size: 32  \# 或更小

# 使用更小的模型

model_type: 'vit_small'

```

### 问题 2: 数据加载慢

**解决方案**：
```


# 增加工作线程

num_workers: 8

# 启用 pin memory

pin_memory: true

```

### 问题 3: 训练不收敛

**解决方案**：
```


# 降低学习率

learning_rate: 0.0001

# 增加 warmup

warmup_epochs: 10

# 减少数据增强

use_mixup: false
use_cutmix: false

```

## 输出文件说明

训练后会生成以下文件：

```

experiments/results/
├── checkpoints/
│   └── vit_base/
│       ├── best_model.pth          \# 最佳模型
│       ├── checkpoint_epoch_10.pth  \# 定期检查点
│       ├── training_history.pkl     \# 训练历史
│       └── training_curves.png      \# 训练曲线
├── logs/
│   └── vit_base/                    \# TensorBoard 日志
└── figures/
├── confusion_matrix.png         \# 混淆矩阵
└── attention_maps.png           \# 注意力可视化

```

## 高级用法

### 自定义模型

修改 `models/vit.py` 或创建新文件：

```

from models import VisionTransformer

# 自定义配置

custom_model = VisionTransformer(
img_size=32,
patch_size=4,
embed_dim=256,  \# 自定义嵌入维度
depth=8,        \# 自定义深度
num_heads=4,    \# 自定义注意力头数
num_classes=10
)

```

### 使用预训练模型

```

import torch
from models import create_vit_base

# 加载预训练权重

model = create_vit_base(num_classes=10)
checkpoint = torch.load('path/to/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])

```

### 迁移到其他数据集

```


# 修改配置

model = VisionTransformer(
img_size=224,      \# ImageNet size
patch_size=16,     \# Larger patches
num_classes=1000   \# ImageNet classes
)

```

## 提交实验报告

1. **收集结果**：
```

./scripts/export_results.sh

```

2. **编写报告**：使用 `docs/experiment_report.md` 作为模板

3. **打包提交**：
```


# 创建提交包

zip -r 实验三_姓名_学号.zip \
experiments/results/ \
docs/experiment_report.md \
README.md

```

## 性能优化建议

1. **使用混合精度训练**（需要修改代码启用 AMP）
2. **使用多 GPU 训练**（需要 DistributedDataParallel）
3. **启用梯度累积**（小显存情况下）
4. **使用编译优化**（PyTorch 2.0+ 的 torch.compile）

---

**最后更新**: 2025-10-22  
**维护者**: Haonan Wang Huazhong University of Science and Tecnology