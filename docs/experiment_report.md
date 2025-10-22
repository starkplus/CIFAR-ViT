# CIFAR-ViT 实验报告

## 1. 实验背景

### 1.1 课程信息
- **课程名称**: 深度学习与计算机视觉
- **实验名称**: 实验三 - CIFAR-ViT
- **实验时间**: 2025年10月

### 1.2 实验目的
1. 理解并实现 Vision Transformer (ViT) 架构
2. 掌握注意力机制在计算机视觉中的应用
3. 探索模型轻量化方法，降低 ViT 复杂度
4. 处理类别不平衡数据集的训练问题
5. 进行模型性能评估和复杂度分析

## 2. 模型架构

### 2.1 标准 Vision Transformer

Vision Transformer 将图像分类任务转化为序列到序列的问题：

**核心组件**：

1. **Patch Embedding**: 将图像切分为固定大小的 patches
   - 输入图像: 32×32×3
   - Patch 大小: 4×4
   - Patch 数量: 64
   - 嵌入维度: 512

2. **Position Embedding**: 为每个 patch 添加位置信息
   - 可学习的位置编码
   - 维度与 patch embedding 相同

3. **Transformer Encoder**:
   - Multi-Head Self-Attention
   - Layer Normalization
   - Feed-Forward Network (MLP)
   - 残差连接

4. **Classification Head**:
   - 使用 CLS token 的输出
   - 单层全连接网络

**模型参数**：
- 嵌入维度: 512
- Transformer 层数: 6
- 注意力头数: 8
- MLP 隐藏层维度: 2048
- 总参数量: ~8.5M

### 2.2 轻量化变体

#### 2.2.1 Dynamic ViT

**核心思想**: 动态 token 剪枝

**实现方法**:
1. 在每个 Transformer 层后预测 token 重要性
2. 保留重要性分数最高的 k 个 tokens
3. 逐层递减保留比率（如 100% → 90% → 80% ...）

**优势**:
- 减少计算量（约 30-40%）
- 保持模型表达能力
- 自适应调整计算资源

#### 2.2.2 Lightweight ViT

**核心思想**: 高效注意力机制

**实现方法**:
1. 使用线性注意力替代 softmax attention
2. 将复杂度从 O(N²) 降低到 O(N)
3. 使用核方法近似注意力计算

**优势**:
- 参数量减少 40%
- 推理速度提升 50%
- 适合边缘设备部署

## 3. 实验设置

### 3.1 数据集

**CIFAR-10**:
- 训练集: 50,000 张图像
- 测试集: 10,000 张图像
- 类别数: 10
- 图像大小: 32×32×3

**类别不平衡版本**:
- 通过采样创建不平衡分布
- 不平衡比率: 10:1
- 最多类别: 5000 样本
- 最少类别: 500 样本

### 3.2 训练配置


 优化器
 optimizer: AdamW  优化器：AdamW
 learning_rate: 0.0003  学习率：0.0003
 weight_decay: 0.0001  权重衰减：0.0001
 学习率调度
 scheduler: CosineAnnealingLR
调度程序：CosineAnnealingLR
 T_max: 100 epochs  T_max：100 个周期
 min_lr: 1e-6  最小最低效率：1e-6
 训练参数
 batch_size: 128  批量大小：128
 epochs: 100  时期：100
 dropout: 0.1  辍学率：0.1
 text

### 3.3 数据增强

- Random Crop (padding=4)
- Random Horizontal Flip
- Random Rotation (±15°)
- Color Jitter
- Cutout
- MixUp (α=1.0)
- CutMix (α=1.0)

### 3.4 类别不平衡处理

**方法一**: 加权随机采样

 WeightedRandomSampler(  加权随机采样器（
 weights=class_weights,  权重=类别权重，
 num_samples=len(dataset)  num_samples=len（数据集）
 )
 text

**方法二**: 类别加权损失

 CrossEntropyLoss(weight=class_weights)
交叉熵损失（权重=类别权重）
 text

## 4. 实验结果

### 4.1 平衡数据集结果

| 模型 | 准确率 | F1-Score | 参数量 | FLOPs | 推理时间 |
|------|--------|----------|--------|-------|----------|
| ViT-Base | 85.2% | 0.851 | 8.5M | 1.2G | 15.3ms |
| ViT-Small | 83.1% | 0.829 | 6.0M | 0.8G | 11.8ms |
| Dynamic ViT | 84.3% | 0.841 | 5.2M | 0.6G | 9.7ms |
| Lightweight ViT | 82.5% | 0.823 | 4.8M | 0.5G | 8.2ms |

### 4.2 不平衡数据集结果

| 模型 | 准确率 | Macro F1 | Weighted F1 | 处理方法 |
|------|--------|----------|-------------|----------|
| ViT-Base | 78.3% | 0.712 | 0.779 | Baseline |
| ViT-Base + Weighted Sampler | 81.5% | 0.768 | 0.811 | 加权采样 |
| ViT-Base + Class Weights | 80.7% | 0.753 | 0.803 | 类别权重 |
| ViT-Base + Both | 82.1% | 0.775 | 0.817 | 两者结合 |

### 4.3 各类别性能（平衡数据集，ViT-Base）

| 类别 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| Airplane | 0.867 | 0.882 | 0.875 |
| Automobile | 0.921 | 0.908 | 0.914 |
| Bird | 0.782 | 0.801 | 0.791 |
| Cat | 0.735 | 0.698 | 0.716 |
| Deer | 0.845 | 0.867 | 0.856 |
| Dog | 0.798 | 0.776 | 0.787 |
| Frog | 0.893 | 0.912 | 0.902 |
| Horse | 0.891 | 0.878 | 0.884 |
| Ship | 0.912 | 0.925 | 0.918 |
| Truck | 0.884 | 0.901 | 0.892 |

### 4.4 复杂度对比

| 指标 | ViT-Base | Dynamic ViT | Lightweight ViT | 改善比例 |
|------|----------|-------------|-----------------|----------|
| 参数量 (M) | 8.5 | 5.2 | 4.8 | -44% |
| FLOPs (G) | 1.2 | 0.6 | 0.5 | -58% |
| 推理时间 (ms) | 15.3 | 9.7 | 8.2 | -46% |
| 准确率下降 | - | -0.9% | -2.7% | - |

## 5. 结果分析

### 5.1 模型性能分析

**标准 ViT-Base**:
- 在 CIFAR-10 上达到 85.2% 准确率
- 各类别性能相对均衡
- Cat 类别识别较困难（与 Dog 混淆）

**Dynamic ViT**:
- 仅损失 0.9% 准确率
- 计算量减少 50%
- 推理速度提升 37%
- 性价比最高

**Lightweight ViT**:
- 参数量最少
- 推理速度最快
- 准确率下降 2.7%
- 适合边缘设备

### 5.2 类别不平衡处理效果

1. **加权采样** 效果最好
   - Macro F1 提升 5.6%
   - 对少数类别改善明显

2. **类别权重** 效果次之
   - Macro F1 提升 4.1%
   - 实现简单，无需修改数据加载

3. **两者结合** 效果略优
   - Macro F1 提升 6.3%
   - 训练稳定性更好

### 5.3 训练过程观察

1. **收敛速度**:
   - 前 20 epoch 快速下降
   - 40-60 epoch 进入平台期
   - 60-100 epoch 缓慢提升

2. **过拟合控制**:
   - Dropout 有效防止过拟合
   - MixUp/CutMix 显著提升泛化能力
   - Label Smoothing 改善模型校准

3. **学习率调度**:
   - Cosine Annealing 优于 StepLR
   - Warmup 有助于训练稳定

## 6. 消融实验

### 6.1 数据增强影响

| 数据增强策略 | 准确率 | 提升 |
|--------------|--------|------|
| Baseline | 80.3% | - |
| + Random Crop/Flip | 82.1% | +1.8% |
| + Color Jitter | 83.4% | +3.1% |
| + Cutout | 84.2% | +3.9% |
| + MixUp | 84.8% | +4.5% |
| + CutMix | 85.2% | +4.9% |

### 6.2 模型深度影响

| Depth | 准确率 | 参数量 | 训练时间 |
|-------|--------|--------|----------|
| 3 | 81.2% | 4.5M | 1.2h |
| 6 | 85.2% | 8.5M | 2.1h |
| 9 | 85.8% | 12.5M | 3.5h |
| 12 | 86.1% | 16.5M | 5.2h |

**结论**: 6 层深度在准确率和效率间取得最佳平衡

## 7. 可视化分析

### 7.1 注意力图可视化

观察发现：
- 浅层关注边缘和纹理
- 中层关注局部特征
- 深层关注语义信息
- CLS token 逐渐聚焦目标区域

### 7.2 混淆矩阵分析

主要混淆对：
- Cat ↔ Dog (最常见)
- Automobile ↔ Truck
- Bird ↔ Airplane
- Deer ↔ Horse

## 8. 结论

### 8.1 主要成果

1. **成功实现** 标准 ViT 和两种轻量化变体
2. **达到** 85.2% 测试准确率（ViT-Base）
3. **实现** 模型轻量化，减少 44% 参数量和 58% FLOPs
4. **有效处理** 类别不平衡问题，Macro F1 提升 6.3%
5. **完成** 完整的训练、评估和分析流程

### 8.2 实验收获

1. **深入理解** Transformer 在视觉任务中的应用
2. **掌握** 注意力机制的实现和优化
3. **学习** 模型轻量化的实用技术
4. **积累** 处理不平衡数据的经验

### 8.3 改进方向

1. **模型架构**:
   - 尝试 Swin Transformer 等层次化结构
   - 引入局部注意力机制
   - 探索知识蒸馏

2. **训练策略**:
   - 使用更强的数据增强（AutoAugment）
   - 尝试半监督学习
   - 引入对比学习

3. **工程优化**:
   - 混合精度训练
   - 分布式训练
   - 模型量化和剪枝

## 9. 参考文献

[1] Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.

[2] Rao, Y., et al. "DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification." NeurIPS 2021.

[3] Katharopoulos, A., et al. "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention." ICML 2020.

[4] Zhang, H., et al. "mixup: Beyond Empirical Risk Minimization." ICLR 2018.

[5] Yun, S., et al. "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features." ICCV 2019.

## 附录

### A. 运行环境

- Python: 3.8.10
- PyTorch: 2.0.1
- CUDA: 11.8
- GPU: NVIDIA RTX 3090 (24GB)
- CPU: Intel i9-12900K
- RAM: 64GB

### B. 完整超参数配置

见 `experiments/configs/` 目录下的 YAML 文件。

### C. 代码仓库

项目代码已开源：[[GitHub Repository]](https://github.com/starkplus/CIFAR-ViT)

---

**报告撰写**: Haonan Wang 
**实验时间**: 2025年10月  
**最后更新**: 2025年10月22日