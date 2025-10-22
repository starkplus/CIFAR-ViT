"""
不平衡数据集训练脚本
"""
import os
import sys
import argparse
import yaml
import torch
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_vit_base, create_vit_small, DynamicViT, LightweightViT
from utils import get_imbalanced_dataloader
from train.trainer import Trainer


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(train_loader):
    """计算类别权重以处理不平衡"""
    # 统计每个类别的样本数
    class_counts = torch.zeros(10)
    
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
    
    # 计算权重：总样本数 / (类别数 * 各类别样本数)
    total_samples = class_counts.sum()
    class_weights = total_samples / (10 * class_counts)
    
    # 归一化
    class_weights = class_weights / class_weights.sum() * 10
    
    return class_weights


def main(args):
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    set_seed(config.get('seed', 42))
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print("\n" + "="*60)
    print("Loading CIFAR-10 Imbalanced Dataset")
    print("="*60)
    
    use_weighted_sampler = config.get('use_weighted_sampler', False)
    
    train_loader, val_loader, test_loader = get_imbalanced_dataloader(
        data_dir=config.get('data_dir', './data/cifar10'),
        indices_path=config.get('indices_path', './data/imbalanced_indices.npy'),
        batch_size=config.get('batch_size', 128),
        img_size=config.get('img_size', 32),
        num_workers=config.get('num_workers', 4),
        use_weighted_sampler=use_weighted_sampler
    )
    
    # 计算类别权重
    if config.get('use_class_weights', False) and not use_weighted_sampler:
        print("\nComputing class weights...")
        class_weights = compute_class_weights(train_loader)
        print(f"Class weights: {class_weights.numpy()}")
        config['class_weights'] = class_weights.tolist()
    
    # 创建模型
    print("\n" + "="*60)
    print("Creating Model")
    print("="*60)
    
    model_type = config.get('model_type', 'vit_base')
    
    if model_type == 'vit_base':
        model = create_vit_base(
            num_classes=10,
            img_size=config.get('img_size', 32)
        )
    elif model_type == 'vit_small':
        model = create_vit_small(
            num_classes=10,
            img_size=config.get('img_size', 32)
        )
    elif model_type == 'dynamic_vit':
        model = DynamicViT(
            img_size=config.get('img_size', 32),
            num_classes=10,
            embed_dim=config.get('embed_dim', 384),
            depth=config.get('depth', 6),
            num_heads=config.get('num_heads', 6)
        )
    elif model_type == 'lightweight_vit':
        model = LightweightViT(
            img_size=config.get('img_size', 32),
            num_classes=10,
            embed_dim=config.get('embed_dim', 384),
            depth=config.get('depth', 6)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 计算模型参数量
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_type}")
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 创建训练器
    print("\n" + "="*60)
    print("Initializing Trainer")
    print("="*60)
    print(f"Using weighted sampler: {use_weighted_sampler}")
    print(f"Using class weights in loss: {config.get('use_class_weights', False)}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # 训练
    history = trainer.train()
    
    # 保存训练历史
    import pickle
    history_path = os.path.join(
        config.get('save_dir', './experiments/results/checkpoints'),
        'training_history_imbalanced.pkl'
    )
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"\nTraining history saved to {history_path}")
    
    # 可视化训练曲线
    from utils import plot_training_curves
    fig_path = os.path.join(
        config.get('save_dir', './experiments/results/checkpoints'),
        'training_curves_imbalanced.png'
    )
    plot_training_curves(history, save_path=fig_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ViT on Imbalanced CIFAR-10')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    
    args = parser.parse_args()
    main(args)