"""
标准Vision Transformer (ViT)模型实现
"""
import torch
import torch.nn as nn
from .patch_embedding import PatchEmbedding
from .transformer_block import TransformerBlock


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT)
    
    Args:
        img_size: 输入图像大小
        patch_size: patch大小
        in_channels: 输入通道数
        num_classes: 分类类别数
        embed_dim: 嵌入维度
        depth: Transformer块的数量
        num_heads: 注意力头数
        mlp_ratio: MLP隐藏层维度比率
        dropout: dropout比率
        attn_dropout: 注意力dropout比率
    """
    
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=512,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer Encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout
            )
            for _ in range(depth)
        ])
        
        # Layer Norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification Head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        # 初始化 position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 初始化线性层
        self.apply(self._init_linear_weights)
        
    def _init_linear_weights(self, m):
        """初始化线性层权重"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch_size, in_channels, img_size, img_size)
            return_attention: 是否返回注意力权重
        
        Returns:
            logits: (batch_size, num_classes)
            attention_maps (optional): list of attention weights
        """
        B = x.shape[0]
        
        # Patch Embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # 添加 class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
        
        # 添加 position embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Transformer Encoder
        attention_maps = []
        for block in self.blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                attention_maps.append(attn)
            else:
                x = block(x)
        
        # Layer Norm
        x = self.norm(x)
        
        # 提取 class token 的输出
        cls_token_output = x[:, 0]  # (B, embed_dim)
        
        # Classification Head
        logits = self.head(cls_token_output)  # (B, num_classes)
        
        if return_attention:
            return logits, attention_maps
        return logits
    
    def get_attention_maps(self, x):
        """获取所有层的注意力图"""
        _, attention_maps = self.forward(x, return_attention=True)
        return attention_maps


def create_vit_tiny(num_classes=10, img_size=32):
    """创建 ViT-Tiny 模型"""
    return VisionTransformer(
        img_size=img_size,
        patch_size=4,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.1
    )


def create_vit_small(num_classes=10, img_size=32):
    """创建 ViT-Small 模型"""
    return VisionTransformer(
        img_size=img_size,
        patch_size=4,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.1
    )


def create_vit_base(num_classes=10, img_size=32):
    """创建 ViT-Base 模型"""
    return VisionTransformer(
        img_size=img_size,
        patch_size=4,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=512,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.1
    )


if __name__ == '__main__':
    # 测试代码
    batch_size = 2
    img_size = 32
    num_classes = 10
    
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    # 测试 ViT-Base
    print("Testing ViT-Base...")
    model = create_vit_base(num_classes=num_classes, img_size=img_size)
    
    logits, attn_maps = model(x, return_attention=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Number of attention maps: {len(attn_maps)}")
    print(f"Attention map shape: {attn_maps[0].shape}")
    
    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")