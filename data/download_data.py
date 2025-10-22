"""
数据下载脚本
下载 CIFAR-10 数据集并创建不平衡版本
"""
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np


def download_cifar10(data_dir='./data/cifar10'):
    """下载 CIFAR-10 数据集"""
    print("Downloading CIFAR-10 dataset...")

    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)

    # 下载训练集
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=None
    )

    # 下载测试集
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=None
    )

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Classes: {train_dataset.classes}")

    return train_dataset, test_dataset


def create_imbalanced_dataset(dataset, imbalance_ratio=0.1, save_dir='./data'):
    """
    创建类别不平衡的数据集

    Args:
        dataset: 原始数据集
        imbalance_ratio: 不平衡比率（最少类别样本数 / 最多类别样本数）
        save_dir: 保存目录

    Returns:
        不平衡数据集的索引列表
    """
    print(f"\nCreating imbalanced dataset with ratio {imbalance_ratio}...")

    # 获取所有标签
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))

    # 计算每个类别的样本数量
    class_indices = [np.where(targets == i)[0] for i in range(num_classes)]
    class_counts = [len(indices) for indices in class_indices]

    print(f"Original class distribution: {class_counts}")

    # 创建不平衡分布
    max_samples = max(class_counts)
    min_samples = int(max_samples * imbalance_ratio)

    # 为每个类别创建不同的采样数量
    imbalanced_indices = []
    new_class_counts = []

    for i in range(num_classes):
        # 线性递减的样本数量
        num_samples = int(max_samples - (max_samples - min_samples) * i / (num_classes - 1))

        # 随机采样
        selected_indices = np.random.choice(
            class_indices[i],
            size=min(num_samples, len(class_indices[i])),
            replace=False
        )

        imbalanced_indices.extend(selected_indices.tolist())
        new_class_counts.append(len(selected_indices))

    print(f"Imbalanced class distribution: {new_class_counts}")
    print(f"Total samples: {len(imbalanced_indices)}")

    # 保存索引
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'imbalanced_indices.npy')
    np.save(save_path, np.array(imbalanced_indices))
    print(f"Imbalanced indices saved to {save_path}")

    return imbalanced_indices


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 下载 CIFAR-10
    train_dataset, test_dataset = download_cifar10()

    # 创建不平衡训练集
    imbalanced_indices = create_imbalanced_dataset(
        train_dataset,
        imbalance_ratio=0.1,
        save_dir='./data'
    )

    print("\nData preparation completed!")
    print("=" * 50)
    print("You can now use:")
    print("  - Balanced dataset: Full CIFAR-10 training set")
    print("  - Imbalanced dataset: Using indices from 'data/imbalanced_indices.npy'")


if __name__ == '__main__':
    main()
