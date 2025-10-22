"""
主程序入口
提供统一的命令行接口进行训练、评估和分析
"""
import os
import argparse
import yaml
import torch
import numpy as np

from models import create_vit_base, create_vit_small, DynamicViT, LightweightViT
from utils import get_balanced_dataloader, get_imbalanced_dataloader
from train.trainer import Trainer


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    """训练模式"""
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
    print(f"Loading {'Imbalanced' if args.imbalanced else 'Balanced'} CIFAR-10 Dataset")
    print("="*60)

    if args.imbalanced:
        train_loader, val_loader, test_loader = get_imbalanced_dataloader(
            data_dir=config.get('data_dir', './data/cifar10'),
            indices_path=config.get('indices_path', './data/imbalanced_indices.npy'),
            batch_size=config.get('batch_size', 128),
            img_size=config.get('img_size', 32),
            num_workers=config.get('num_workers', 4),
            use_weighted_sampler=config.get('use_weighted_sampler', False)
        )
    else:
        train_loader, val_loader, test_loader = get_balanced_dataloader(
            data_dir=config.get('data_dir', './data/cifar10'),
            batch_size=config.get('batch_size', 128),
            img_size=config.get('img_size', 32),
            num_workers=config.get('num_workers', 4)
        )

    # 创建模型
    print("\n" + "="*60)
    print("Creating Model")
    print("="*60)

    model_type = config.get('model_type', 'vit_base')

    if model_type == 'vit_base':
        model = create_vit_base(num_classes=10, img_size=config.get('img_size', 32))
    elif model_type == 'vit_small':
        model = create_vit_small(num_classes=10, img_size=config.get('img_size', 32))
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

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_type}")
    print(f"Total parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # 创建训练器并训练
    trainer = Trainer(model, train_loader, val_loader, config, device)
    history = trainer.train()

    # 保存训练历史
    import pickle
    history_path = os.path.join(config.get('save_dir', './experiments/results/checkpoints'), 'training_history.pkl')
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)

    print(f"\n✓ Training completed! History saved to {history_path}")


def evaluate(args):
    """评估模式"""
    from evaluation.evaluate import load_model, evaluate_model
    from utils import calculate_metrics, print_metrics, plot_confusion_matrix

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载测试数据
    _, _, test_loader = get_balanced_dataloader(
        data_dir=config.get('data_dir', './data/cifar10'),
        batch_size=config.get('batch_size', 128),
        img_size=config.get('img_size', 32),
        num_workers=config.get('num_workers', 4)
    )

    # 加载模型
    model = load_model(args.checkpoint, config.get('model_type', 'vit_base'), config, device)

    # 评估
    all_preds, all_targets = evaluate_model(model, test_loader, device)

    # 计算并打印指标
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    metrics = calculate_metrics(all_preds, all_targets, num_classes=10)
    print_metrics(metrics, class_names)

    # 保存结果
    save_dir = config.get('save_dir', './experiments/results')
    os.makedirs(save_dir, exist_ok=True)
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(all_targets, all_preds, class_names, save_path=cm_path)

    print(f"\n✓ Evaluation completed! Results saved to {save_dir}")


def analyze(args):
    """复杂度分析模式"""
    from evaluation.complexity_analysis import analyze_model

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 分析模型
    models_to_analyze = args.models if args.models else ['vit_base', 'vit_small', 'dynamic_vit', 'lightweight_vit']

    all_results = []
    for model_type in models_to_analyze:
        try:
            results = analyze_model(model_type, config, device)
            all_results.append(results)
        except Exception as e:
            print(f"Error analyzing {model_type}: {e}")

    print("\n✓ Analysis completed!")


def main():
    parser = argparse.ArgumentParser(description='CIFAR-ViT: Vision Transformer for CIFAR-10')
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # 训练模式
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--config', type=str, required=True, help='Path to config file')
    train_parser.add_argument('--imbalanced', action='store_true', help='Use imbalanced dataset')

    # 评估模式
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--config', type=str, required=True, help='Path to config file')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')

    # 分析模式
    analyze_parser = subparsers.add_parser('analyze', help='Analyze model complexity')
    analyze_parser.add_argument('--config', type=str, required=True, help='Path to config file')
    analyze_parser.add_argument('--models', nargs='+', help='Models to analyze')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'analyze':
        analyze(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()