"""
图像分块嵌入层实现
将输入图像切分为固定大小的patches并进行线性投影
"""
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    将图像切分为patches并进行嵌入

    Args:
        img_size: 输入图像大小
        patch_size: patch大小
        in_channels: 输入通道数
        embed_dim: 嵌入维度
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # 计算patch数量
        self.num_patches = (img_size // patch_size) ** 2

        # 使用卷积层实现patch embedding
        # kernel_size和stride都设置为patch_size
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, img_size, img_size)

        Returns:
            (batch_size, num_patches, embed_dim)
        """
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"

        # 投影: (B, C, H, W) -> (B, embed_dim, H', W')
        # 其中 H' = W' = img_size / patch_size
        x = self.projection(x)

        # 展平并转置: (B, embed_dim, H', W') -> (B, num_patches, embed_dim)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)

        return x


if __name__ == '__main__':
    # 测试代码
    batch_size = 4
    img_size = 32
    patch_size = 4
    in_channels = 3
    embed_dim = 512

    # 创建模拟输入
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    print(f"Input shape: {x.shape}")

    # 创建patch embedding层
    patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

    # 前向传播
    output = patch_embed(x)
    print(f"Output shape: {output.shape}")
    print(f"Number of patches: {patch_embed.num_patches}")
    print(f"Expected: ({batch_size}, {patch_embed.num_patches}, {embed_dim})")