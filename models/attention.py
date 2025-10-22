"""
注意力机制实现
包含标准多头自注意力和优化版本
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制

    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        dropout: dropout比率
        bias: 是否使用偏置
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"

        # Q, K, V 投影
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)

        # 输出投影
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            return_attention: 是否返回注意力权重

        Returns:
            out: (batch_size, seq_len, embed_dim)
            attn_weights (optional): (batch_size, num_heads, seq_len, seq_len)
        """
        B, N, C = x.shape

        # 生成 Q, K, V
        # (B, N, 3*C) -> (B, N, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力分数
        # (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N)
        # -> (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # 应用注意力权重
        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim)
        # -> (B, num_heads, N, head_dim)
        x = attn @ v

        # 合并多头
        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim)
        x = x.transpose(1, 2).reshape(B, N, C)

        # 输出投影
        x = self.proj(x)
        x = self.proj_dropout(x)

        if return_attention:
            return x, attn
        return x


class EfficientAttention(nn.Module):
    """
    高效注意力机制（线性复杂度）
    使用核方法近似softmax attention
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        线性注意力的前向传播
        复杂度: O(N * D^2) 而不是 O(N^2 * D)
        """
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 使用 ReLU 作为核函数近似
        q = F.relu(q)
        k = F.relu(k)

        # 线性注意力: O(N * D^2)
        # (B, H, N, D) @ (B, H, D, N) -> (B, H, D, D)
        k_cumsum = k.sum(dim=2, keepdim=True)
        D_inv = 1.0 / (q @ k_cumsum.transpose(-2, -1) + 1e-8)

        # (B, H, N, D) @ (B, H, D, N) @ (B, H, N, D)
        context = k.transpose(-2, -1) @ v
        x = (q @ context) * D_inv

        # 合并多头
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)

        return x


if __name__ == '__main__':
    # 测试代码
    batch_size = 2
    seq_len = 64
    embed_dim = 512
    num_heads = 8

    x = torch.randn(batch_size, seq_len, embed_dim)

    # 测试标准多头注意力
    print("Testing MultiHeadAttention...")
    mha = MultiHeadAttention(embed_dim, num_heads)
    out, attn = mha(x, return_attention=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Attention shape: {attn.shape}")

    # 测试高效注意力
    print("\nTesting EfficientAttention...")
    eff_attn = EfficientAttention(embed_dim, num_heads)
    out2 = eff_attn(x)
    print(f"Output shape: {out2.shape}")