"""
数据加载器
支持CIFAR-10平衡和不平衡数据集加载
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms


def get_cifar10_transforms(is_train=True, img_size=32):
    """
    获取CIFAR-10数据增强transforms

    Args:
        is_train: 是否为训练集
        img_size: 图像大小

    Returns:
        transforms.Compose对象
    """
    if is_train:
        transform = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            ),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            ),
        ])

    return transform


def get_balanced_dataloader(
    data_dir='./data/cifar10',
    batch_size=128,
    img_size=32,
    num_workers=4,
    pin_memory=True
):
    """
    获取平衡的CIFAR-10数据加载器

    Args:
        data_dir: 数据目录
        batch_size: batch大小
        img_size: 图像大小
        num_workers: 数据加载线程数
        pin_memory: 是否使用pin_memory

    Returns:
        train_loader, val_loader, test_loader
    """
    print("Loading balanced CIFAR-10 dataset...")

    # 获取transforms
    train_transform = get_cifar10_transforms(is_train=True, img_size=img_size)
    test_transform = get_cifar10_transforms(is_train=False, img_size=img_size)

    # 加载训练集
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    # 划分训练集和验证集
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 为验证集更新transform
    val_dataset.dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=test_transform
    )

    # 加载测试集
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader


def get_imbalanced_dataloader(
    data_dir='./data/cifar10',
    indices_path='./data/imbalanced_indices.npy',
    batch_size=128,
    img_size=32,
    num_workers=4,
    pin_memory=True,
    use_weighted_sampler=False
):
    """
    获取不平衡的CIFAR-10数据加载器

    Args:
        data_dir: 数据目录
        indices_path: 不平衡索引文件路径
        batch_size: batch大小
        img_size: 图像大小
        num_workers: 数据加载线程数
        pin_memory: 是否使用pin_memory
        use_weighted_sampler: 是否使用加权采样器平衡类别

    Returns:
        train_loader, val_loader, test_loader
    """
    print("Loading imbalanced CIFAR-10 dataset...")

    # 获取transforms
    train_transform = get_cifar10_transforms(is_train=True, img_size=img_size)
    test_transform = get_cifar10_transforms(is_train=False, img_size=img_size)

    # 加载完整训练集
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    # 加载不平衡索引
    if not os.path.exists(indices_path):
        raise FileNotFoundError(
            f"Imbalanced indices not found at {indices_path}. "
            "Please run data/download_data.py first."
        )

    imbalanced_indices = np.load(indices_path)
    print(f"Loaded {len(imbalanced_indices)} imbalanced samples")

    # 创建不平衡数据集
    train_dataset = Subset(full_train_dataset, imbalanced_indices)

    # 统计类别分布
    targets = np.array([full_train_dataset.targets[i] for i in imbalanced_indices])
    class_counts = np.bincount(targets)
    print(f"Class distribution: {class_counts}")

    # 创建验证集（使用平衡的验证集）
    val_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=test_transform
    )

    # 划分验证集
    all_indices = set(range(len(full_train_dataset)))
    train_indices_set = set(imbalanced_indices)
    val_indices = list(all_indices - train_indices_set)
    val_indices = val_indices[:5000]  # 使用5000个样本作为验证集
    val_dataset = Subset(val_dataset, val_indices)

    # 加载测试集
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    # 创建训练集DataLoader
    if use_weighted_sampler:
        # 使用加权采样器平衡类别
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = class_weights[targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        print("Using WeightedRandomSampler for class balancing")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"Train set: {len(train_dataset)} samples (imbalanced)")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 测试代码
    print("Testing balanced dataloader...")
    train_loader, val_loader, test_loader = get_balanced_dataloader(
        batch_size=64
    )

    # 测试一个batch
    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")

    print("\nTesting imbalanced dataloader...")
    try:
        train_loader_imb, val_loader_imb, test_loader_imb = get_imbalanced_dataloader(
            batch_size=64,
            use_weighted_sampler=True
        )

        images, labels = next(iter(train_loader_imb))
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")