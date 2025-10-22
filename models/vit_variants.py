"""
ViT变体模型实现（轻量化版本）
包含 Dynamic ViT 和其他高效变体
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .patch_embedding import PatchEmbedding
from .transformer_block import TransformerBlock, EfficientTransformerBlock


class DynamicViT(nn.Module):
    """
    Dynamic Vision Transformer
    通过动态token剪枝减少计算量
    
    Args:
        img_size: 输入图像大小
        patch_size: patch大小
        in_channels: 输入通道数
        num_classes: 分类类别数
        embed_dim: 嵌入维度
        depth: Transformer块数量
        num_heads: 注意力头数
        mlp_ratio: MLP隐藏层维度比率
        dropout: dropout比率
        keep_rate: 每层保留的token比率列表
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
        keep_rate=None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        
        # 默认保留率：逐层递减
        if keep_rate is None:
            keep_rate = [1.0 - (0.1 * i) for i in range(depth)]
            keep_rate = [max(0.5, rate) for rate in keep_rate]
        self.keep_rate = keep_rate
        
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
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Token importance predictors
        self.predictors = nn.ModuleList([
            nn.Linear(embed_dim, 1)
            for _ in range(depth)
        ])
        
        # Layer Norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification Head
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_linear_weights)
        
    def _init_linear_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def prune_tokens(self, x, importance_scores, keep_rate):
        """
        根据重要性分数剪枝tokens
        
        Args:
            x: (B, N, D) tokens
            importance_scores: (B, N) 重要性分数
            keep_rate: 保留比率
        
        Returns:
            x: (B, N', D) 剪枝后的tokens
            indices: (B, N') 保留的token索引
        """
        B, N, D = x.shape
        
        # 保留的token数量
        num_keep = max(1, int(N * keep_rate))
        
        # 获取top-k重要的tokens
        _, indices = torch.topk(importance_scores, num_keep, dim=1)
        indices = indices.sort(dim=1)[0]  # 排序以保持顺序
        
        # 选择tokens
        x = torch.gather(
            x,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, D)
        )
        
        return x, indices
    
    def forward(self, x, inference_mode=False):
        """
        Args:
            x: (batch_size, in_channels, img_size, img_size)
            inference_mode: 推理模式下进行token剪枝
        
        Returns:
            logits: (batch_size, num_classes)
        """
        B = x.shape[0]
        
        # Patch Embedding
        x = self.patch_embed(x)
        
        # 添加 class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加 position embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # 保存原始位置编码用于token剪枝后恢复
        pos_embed = self.pos_embed
        
        # Transformer Encoder with dynamic pruning
        for i, (block, predictor) in enumerate(zip(self.blocks, self.predictors)):
            # Transformer block
            x = block(x)
            
            # Token pruning (除了class token)
            if inference_mode and i < len(self.blocks) - 1:
                cls_token = x[:, 0:1]  # 保留class token
                patch_tokens = x[:, 1:]  # patch tokens
                
                # 预测token重要性
                importance = predictor(patch_tokens).squeeze(-1)  # (B, N)
                
                # 剪枝
                patch_tokens, indices = self.prune_tokens(
                    patch_tokens,
                    importance,
                    self.keep_rate[i]
                )
                
                # 合并class token和剪枝后的tokens
                x = torch.cat([cls_token, patch_tokens], dim=1)
        
        # Layer Norm
        x = self.norm(x)
        
        # Classification
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        return logits


class LightweightViT(nn.Module):
    """
    轻量级ViT：使用高效注意力机制
    """
    
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=3.0,
        dropout=0.1
    ):
        super().__init__()
        
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
        
        # Efficient Transformer blocks
        self.blocks = nn.ModuleList([
            EfficientTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        B = x.shape[0]
        
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        return logits


if __name__ == '__main__':
    # 测试代码
    batch_size = 2
    img_size = 32
    num_classes = 10
    
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    # 测试 Dynamic ViT
    print("Testing Dynamic ViT...")
    dynamic_model = DynamicViT(
        img_size=img_size,
        num_classes=num_classes,
        embed_dim=384,
        depth=6,
        num_heads=6
    )
    
    logits = dynamic_model(x, inference_mode=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    num_params = sum(p.numel() for p in dynamic_model.parameters())
    print(f"Dynamic ViT parameters: {num_params:,}")
    
    # 测试 Lightweight ViT
    print("\nTesting Lightweight ViT...")
    light_model = LightweightViT(
        img_size=img_size,
        num_classes=num_classes,
        embed_dim=384,
        depth=6
    )
    
    logits2 = light_model(x)
    print(f"Output shape: {logits2.shape}")
    
    num_params2 = sum(p.numel() for p in light_model.parameters())
    print(f"Lightweight ViT parameters: {num_params2:,}")