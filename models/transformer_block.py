"""
Transformer编码器块实现
包含多头自注意力和前馈网络
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    多层感知机（前馈网络）
    
    Args:
        in_features: 输入特征维度
        hidden_features: 隐藏层维度
        out_features: 输出特征维度
        dropout: dropout比率
    """
    
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer编码器块
    
    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        mlp_ratio: MLP隐藏层维度相对于embed_dim的比率
        dropout: dropout比率
        attn_dropout: 注意力dropout比率
    """
    
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0, 
                 dropout=0.0, attn_dropout=0.0):
        super().__init__()
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Multi-Head Self-Attention
        from .attention import MultiHeadAttention
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout
        )
        
        # Layer Normalization
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP (Feed-Forward Network)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            dropout=dropout
        )
        
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            return_attention: 是否返回注意力权重
        
        Returns:
            x: (batch_size, seq_len, embed_dim)
            attn (optional): 注意力权重
        """
        # Pre-norm + Self-Attention + Residual
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_out
        else:
            x = x + self.attn(self.norm1(x))
        
        # Pre-norm + MLP + Residual
        x = x + self.mlp(self.norm2(x))
        
        if return_attention:
            return x, attn_weights
        return x


class EfficientTransformerBlock(nn.Module):
    """
    高效Transformer块（使用线性注意力）
    """
    
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        
        from .attention import EfficientAttention
        self.attn = EfficientAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            dropout=dropout
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


if __name__ == '__main__':
    # 测试代码
    batch_size = 2
    seq_len = 64
    embed_dim = 512
    num_heads = 8
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # 测试标准Transformer块
    print("Testing TransformerBlock...")
    block = TransformerBlock(embed_dim, num_heads)
    out, attn = block(x, return_attention=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Attention shape: {attn.shape}")
    
    # 测试高效Transformer块
    print("\nTesting EfficientTransformerBlock...")
    eff_block = EfficientTransformerBlock(embed_dim, num_heads)
    out2 = eff_block(x)
    print(f"Output shape: {out2.shape}")