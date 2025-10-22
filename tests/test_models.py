"""
模型测试
"""
import unittest
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    VisionTransformer,
    create_vit_tiny,
    create_vit_small,
    create_vit_base,
    DynamicViT,
    LightweightViT,
    PatchEmbedding,
    MultiHeadAttention,
    TransformerBlock
)


class TestPatchEmbedding(unittest.TestCase):
    """测试 Patch Embedding"""
    
    def test_output_shape(self):
        """测试输出形状"""
        batch_size = 4
        img_size = 32
        patch_size = 4
        in_channels = 3
        embed_dim = 512
        
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        output = patch_embed(x)
        expected_patches = (img_size // patch_size) ** 2
        
        self.assertEqual(output.shape, (batch_size, expected_patches, embed_dim))
    
    def test_num_patches(self):
        """测试 patch 数量"""
        patch_embed = PatchEmbedding(32, 4, 3, 512)
        self.assertEqual(patch_embed.num_patches, 64)


class TestMultiHeadAttention(unittest.TestCase):
    """测试多头注意力"""
    
    def test_output_shape(self):
        """测试输出形状"""
        batch_size = 2
        seq_len = 64
        embed_dim = 512
        num_heads = 8
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        attn = MultiHeadAttention(embed_dim, num_heads)
        
        output = attn(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_attention_weights(self):
        """测试注意力权重"""
        batch_size = 2
        seq_len = 64
        embed_dim = 512
        num_heads = 8
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        attn = MultiHeadAttention(embed_dim, num_heads)
        
        output, attn_weights = attn(x, return_attention=True)
        self.assertEqual(attn_weights.shape, (batch_size, num_heads, seq_len, seq_len))


class TestTransformerBlock(unittest.TestCase):
    """测试 Transformer 块"""
    
    def test_output_shape(self):
        """测试输出形状"""
        batch_size = 2
        seq_len = 64
        embed_dim = 512
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        block = TransformerBlock(embed_dim, num_heads=8)
        
        output = block(x)
        self.assertEqual(output.shape, x.shape)


class TestVisionTransformer(unittest.TestCase):
    """测试 Vision Transformer"""
    
    def test_forward_pass(self):
        """测试前向传播"""
        batch_size = 2
        img_size = 32
        num_classes = 10
        
        x = torch.randn(batch_size, 3, img_size, img_size)
        model = create_vit_base(num_classes, img_size)
        
        output = model(x)
        self.assertEqual(output.shape, (batch_size, num_classes))
    
    def test_attention_maps(self):
        """测试注意力图"""
        batch_size = 2
        img_size = 32
        num_classes = 10
        
        x = torch.randn(batch_size, 3, img_size, img_size)
        model = create_vit_base(num_classes, img_size)
        
        logits, attn_maps = model(x, return_attention=True)
        
        self.assertEqual(logits.shape, (batch_size, num_classes))
        self.assertEqual(len(attn_maps), model.depth)
    
    def test_different_sizes(self):
        """测试不同模型大小"""
        batch_size = 2
        img_size = 32
        num_classes = 10
        
        x = torch.randn(batch_size, 3, img_size, img_size)
        
        # Tiny
        model_tiny = create_vit_tiny(num_classes, img_size)
        output_tiny = model_tiny(x)
        self.assertEqual(output_tiny.shape, (batch_size, num_classes))
        
        # Small
        model_small = create_vit_small(num_classes, img_size)
        output_small = model_small(x)
        self.assertEqual(output_small.shape, (batch_size, num_classes))


class TestDynamicViT(unittest.TestCase):
    """测试 Dynamic ViT"""
    
    def test_forward_pass(self):
        """测试前向传播"""
        batch_size = 2
        img_size = 32
        num_classes = 10
        
        x = torch.randn(batch_size, 3, img_size, img_size)
        model = DynamicViT(
            img_size=img_size,
            num_classes=num_classes,
            embed_dim=384,
            depth=6,
            num_heads=6
        )
        
        # 训练模式
        output_train = model(x, inference_mode=False)
        self.assertEqual(output_train.shape, (batch_size, num_classes))
        
        # 推理模式（带剪枝）
        output_infer = model(x, inference_mode=True)
        self.assertEqual(output_infer.shape, (batch_size, num_classes))


class TestLightweightViT(unittest.TestCase):
    """测试 Lightweight ViT"""
    
    def test_forward_pass(self):
        """测试前向传播"""
        batch_size = 2
        img_size = 32
        num_classes = 10
        
        x = torch.randn(batch_size, 3, img_size, img_size)
        model = LightweightViT(
            img_size=img_size,
            num_classes=num_classes,
            embed_dim=384,
            depth=6
        )
        
        output = model(x)
        self.assertEqual(output.shape, (batch_size, num_classes))


class TestModelParameters(unittest.TestCase):
    """测试模型参数量"""
    
    def test_parameter_count(self):
        """测试参数量计算"""
        model = create_vit_base(num_classes=10, img_size=32)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 确保有参数
        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)
        
        # 打印参数量（用于验证）
        print(f"\nViT-Base parameters: {total_params:,} ({total_params/1e6:.2f}M)")


if __name__ == '__main__':
    unittest.main()