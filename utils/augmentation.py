"""
数据增强策略
包含各种数据增强方法
"""
import torch
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image, ImageFilter


class GaussianBlur:
    """高斯模糊"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class Cutout:
    """
    Cutout数据增强
    随机遮挡图像的一部分
    """

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img: Tensor of shape (C, H, W)

        Returns:
            Tensor with cutout applied
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class MixUp:
    """
    MixUp数据增强
    混合两个样本及其标签
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch_x, batch_y):
        """
        Args:
            batch_x: (B, C, H, W)
            batch_y: (B,) or (B, num_classes)

        Returns:
            mixed_x, mixed_y, lam
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size).to(batch_x.device)

        mixed_x = lam * batch_x + (1 - lam) * batch_x[index, :]

        y_a, y_b = batch_y, batch_y[index]

        return mixed_x, y_a, y_b, lam


class CutMix:
    """
    CutMix数据增强
    裁剪并粘贴图像块
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch_x, batch_y):
        """
        Args:
            batch_x: (B, C, H, W)
            batch_y: (B,)

        Returns:
            mixed_x, mixed_y, lam
        """
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size).to(batch_x.device)

        # 生成随机box
        _, _, H, W = batch_x.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # 随机中心点
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # 边界框
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # 应用cutmix
        batch_x[:, :, bby1:bby2, bbx1:bbx2] = batch_x[index, :, bby1:bby2, bbx1:bbx2]

        # 调整lambda以匹配实际像素比率
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        y_a, y_b = batch_y, batch_y[index]

        return batch_x, y_a, y_b, lam


def get_strong_augmentation(img_size=32):
    """
    获取强数据增强策略
    """
    return transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(sigma=[0.1, 2.0]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
        Cutout(n_holes=1, length=16)
    ])


def get_weak_augmentation(img_size=32):
    """
    获取弱数据增强策略
    """
    return transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])


if __name__ == '__main__':
    # 测试代码
    print("Testing augmentation methods...")

    # 测试Cutout
    cutout = Cutout(n_holes=1, length=16)
    img = torch.randn(3, 32, 32)
    img_cutout = cutout(img)
    print(f"Cutout: {img.shape} -> {img_cutout.shape}")

    # 测试MixUp
    mixup = MixUp(alpha=1.0)
    batch_x = torch.randn(8, 3, 32, 32)
    batch_y = torch.randint(0, 10, (8,))
    mixed_x, y_a, y_b, lam = mixup(batch_x, batch_y)
    print(f"MixUp: {batch_x.shape} -> {mixed_x.shape}, lambda={lam:.3f}")

    # 测试CutMix
    cutmix = CutMix(alpha=1.0)
    mixed_x, y_a, y_b, lam = cutmix(batch_x, batch_y)
    print(f"CutMix: {batch_x.shape} -> {mixed_x.shape}, lambda={lam:.3f}")