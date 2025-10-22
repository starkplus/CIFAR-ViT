"""
数据加载器测试
"""
import unittest
import torch
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import (
    get_cifar10_transforms,
    get_balanced_dataloader,
    get_imbalanced_dataloader
)
from utils.augmentation import Cutout, MixUp, CutMix


class TestTransforms(unittest.TestCase):
    """测试数据变换"""
    
    def test_train_transforms(self):
        """测试训练集变换"""
        transform = get_cifar10_transforms(is_train=True, img_size=32)
        self.assertIsNotNone(transform)
    
    def test_test_transforms(self):
        """测试测试集变换"""
        transform = get_cifar10_transforms(is_train=False, img_size=32)
        self.assertIsNotNone(transform)


class TestBalancedDataLoader(unittest.TestCase):
    """测试平衡数据加载器"""
    
    @unittest.skipIf(not os.path.exists('./data/cifar10'), "CIFAR-10 not downloaded")
    def test_dataloader_creation(self):
        """测试数据加载器创建"""
        train_loader, val_loader, test_loader = get_balanced_dataloader(
            batch_size=64,
            num_workers=0  # 使用0避免多进程问题
        )
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
    
    @unittest.skipIf(not os.path.exists('./data/cifar10'), "CIFAR-10 not downloaded")
    def test_batch_shapes(self):
        """测试batch形状"""
        train_loader, _, _ = get_balanced_dataloader(
            batch_size=32,
            num_workers=0
        )
        
        images, labels = next(iter(train_loader))
        
        self.assertEqual(images.shape[0], 32)  # batch_size
        self.assertEqual(images.shape[1], 3)   # channels
        self.assertEqual(images.shape[2], 32)  # height
        self.assertEqual(images.shape[3], 32)  # width
        self.assertEqual(labels.shape[0], 32)


class TestImbalancedDataLoader(unittest.TestCase):
    """测试不平衡数据加载器"""
    
    @unittest.skipIf(
        not os.path.exists('./data/imbalanced_indices.npy'),
        "Imbalanced indices not created"
    )
    def test_dataloader_creation(self):
        """测试数据加载器创建"""
        train_loader, val_loader, test_loader = get_imbalanced_dataloader(
            batch_size=64,
            num_workers=0,
            use_weighted_sampler=False
        )
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)


class TestAugmentation(unittest.TestCase):
    """测试数据增强"""
    
    def test_cutout(self):
        """测试Cutout"""
        cutout = Cutout(n_holes=1, length=16)
        x = torch.randn(3, 32, 32)
        
        output = cutout(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_mixup(self):
        """测试MixUp"""
        mixup = MixUp(alpha=1.0)
        
        batch_x = torch.randn(8, 3, 32, 32)
        batch_y = torch.randint(0, 10, (8,))
        
        mixed_x, y_a, y_b, lam = mixup(batch_x, batch_y)
        
        self.assertEqual(mixed_x.shape, batch_x.shape)
        self.assertGreaterEqual(lam, 0)
        self.assertLessEqual(lam, 1)
    
    def test_cutmix(self):
        """测试CutMix"""
        cutmix = CutMix(alpha=1.0)
        
        batch_x = torch.randn(8, 3, 32, 32)
        batch_y = torch.randint(0, 10, (8,))
        
        mixed_x, y_a, y_b, lam = cutmix(batch_x, batch_y)
        
        self.assertEqual(mixed_x.shape, batch_x.shape)
        self.assertGreaterEqual(lam, 0)
        self.assertLessEqual(lam, 1)


if __name__ == '__main__':
    unittest.main()