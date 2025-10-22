# CIFAR-ViT 架构说明

## 模型架构详解

### 1. Vision Transformer 整体架构


 输入图像 (32×32×3)
 ↓
 Patch Embedding  补丁嵌入
 ↓
 添加 CLS Token
 ↓
 Position Embedding  位置嵌入
 ↓
 Dropout  辍学
 ↓
 Transformer Encoder × N  变压器编码器 × N
 ├── Multi-Head Self-Attention
├── 多头自注意力
 ├── LayerNorm
 ├── Feed-Forward Network  ├── 前馈网络
 └── Residual Connection  └── 残差连接
 ↓
 LayerNorm  层范数
 ↓
 提取 CLS Token
 ↓
 Classification Head  分类主管
 ↓
 输出 Logits (10 类)
 text

### 2. 关键组件

#### 2.1 Patch Embedding

将图像转换为序列：


 输入: (B, 3, 32, 32)
 输出: (B, 64, 512)
 Conv2d(3, 512, kernel_size=4, stride=4)
Conv2d（3，512，kernel_size=4，步幅=4）
 Flatten  展平
 Transpose  转置
 text

#### 2.2 Multi-Head Self-Attention


 输入: (B, N, D)
 输出: (B, N, D)
 Q = Linear(D, D)(x) # Query
Q = Linear(D, D)(x) # 查询
 K = Linear(D, D)(x) # Key
K = Linear(D, D)(x) # 键
 V = Linear(D, D)(x) # Value
V = 线性（D，D）（x）#值
 多头拆分
 Q = Q.reshape(B, N, num_heads, head_dim)
Q = Q.重塑（B，N，num_heads，head_dim）
 K = K.reshape(B, N, num_heads, head_dim)
K = K.重塑（B，N，num_heads，head_dim）
 V = V.reshape(B, N, num_heads, head_dim)
V = V.重塑（B，N，num_heads，head_dim）
 注意力计算
 Attention = Softmax(Q @ K.T / sqrt(head_dim))
注意力机制 = Softmax(Q @ KT / sqrt(head_dim))
 Output = Attention @ V
输出 = 注意力 @ V
 多头合并
 Output = Output.reshape(B, N, D)
输出 = 输出.重塑（B，N，D）
 Output = Linear(D, D)(Output)
输出 = 线性（D，D）（输出）
 text

#### 2.3 Feed-Forward Network


 输入: (B, N, D)
 输出: (B, N, D)
 x = Linear(D, 4D)(x)  x = 线性（D，4D）（x）
 x = GELU(x)
 x = Dropout(x)
 x = Linear(4D, D)(x)  x = 线性（4D，D）（x）
 x = Dropout(x)
 text

### 3. 轻量化变体

#### 3.1 Dynamic ViT

动态剪枝流程：


 Transformer Block  变压器块
 ↓
 Token Importance Prediction
Token 重要性预测
 ↓
 Top-K Selection  Top-K 选择
 ↓
 继续下一层（仅处理保留的 tokens）
 text

关键代码：

 importance = predictor(tokens) # (B, N)
重要性 = 预测器（标记）#（B，N）
 top_k_indices = torch.topk(importance, k)image.jpg
top_k_indices = torch.topk(重要性，k) 图片.jpg
 tokens = tokens.gather(1, top_k_indices)
令牌 = 令牌. 收集（1，top_k_indices）
 text

#### 3.2 Lightweight ViT

线性注意力：


 标准注意力: O(N²D)
 Attention = Softmax(Q @ K.T) @ V
注意力 = Softmax(Q @ KT) @ V
 线性注意力: O(ND²)
 Q = ReLU(Q)
 K = ReLU(K)
 Context = K.T @ V # (D, D)
上下文 = KT @ V # (D, D)
 Output = Q @ Context # (N, D)
输出 = Q @ 上下文 # (N, D)
 text

## 数据流

### 训练阶段


 CIFAR-10 训练集
 ↓
 数据增强 (Random Crop, Flip, etc.)
数据增强(Random Crop, Flip, etc.)
 ↓
 DataLoader (batch_size=128)
数据加载器（batch_size=128）
 ↓
 Vision Transformer  视觉转换器
 ↓
 Cross Entropy Loss  交叉熵损失
 ↓
 AdamW Optimizer  AdamW 优化器
 ↓
 反向传播
 ↓
 更新参数
 text

### 推理阶段


 测试图像
 ↓
 标准化
 ↓
 Vision Transformer  视觉转换器
 ↓
 Softmax
 ↓
 预测类别
 text

## 复杂度分析

### 计算复杂度

**标准注意力**:
- Self-Attention: O(N²D)
- Feed-Forward: O(ND²)
- 总计: O(N²D + ND²)

**线性注意力**:
- Linear Attention: O(ND²)
- Feed-Forward: O(ND²)
- 总计: O(ND²)

其中：
- N: 序列长度 (64)
- D: 嵌入维度 (512)

### 参数量

| 组件 | 参数量 |
|------|--------|
| Patch Embedding | 3×4×4×512 = 24,576 |
| Position Embedding | 65×512 = 33,280 |
| Transformer Block (×6) | ~8M |
| Classification Head | 512×10 = 5,120 |
| **总计** | **~8.5M** |

## 实现细节

### 初始化策略


 Position Embedding  位置嵌入
 nn.init.trunc_normal_(pos_embed, std=0.02)
 Linear Layer  线性层
 nn.init.trunc_normal_(weight, std=0.02)
nn.init.trunc_normal_(权重，std=0.02)
 nn.init.constant_(bias, 0)
nn.init.constant_（偏差，0）
 LayerNorm  层范数
 nn.init.constant_(weight, 1.0)
nn.init.constant_（权重，1.0）
 nn.init.constant_(bias, 0)
nn.init.constant_（偏差，0）
 text

### 正则化技术

1. **Dropout**: 0.1
2. **Weight Decay**: 0.0001
3. **Label Smoothing**: 0.1
4. **Gradient Clipping**: 1.0

### 优化技巧

1. **学习率预热** (Warmup)
2. **余弦退火** (Cosine Annealing)
3. **混合精度训练** (AMP)
4. **梯度累积** (Gradient Accumulation)

## 扩展性

### 模型缩放


 Tiny  微小的
 embed_dim=192, depth=12, num_heads=3
embed_dim=192，深度=12，num_heads=3
 Small  小的
 embed_dim=384, depth=12, num_heads=6
embed_dim=384，深度=12，num_heads=6
 Base  根据
 embed_dim=512, depth=6, num_heads=8
embed_dim=512，深度=6，num_heads=8
 Large  大的
 embed_dim=768, depth=12, num_heads=12
embed_dim=768，深度=12，num_heads=12
 text

### 适配其他数据集


 ImageNet (224×224)
 patch_size=16, num_patches=196
补丁大小=16，补丁数量=196
 CIFAR-100
 num_classes=100  类数=100
 TinyImageNet (64×64)  TinyImageNet（64×64）
 img_size=64, patch_size=8
img_size=64，patch_size=8
 text

---

**更新日期**: 2025-10-22