"""
模型评估脚本
"""
import os
import sys
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_vit_base, create_vit_small, DynamicViT, LightweightViT
from utils import (
    get_balanced_dataloader,
    calculate_metrics,
    print_metrics,
    plot_confusion_matrix
)


def load_model(model_path, model_type, config, device):
    """加载训练好的模型"""
    # 创建模型
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
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    if 'best_acc' in checkpoint:
        print(f"Best validation accuracy: {checkpoint['best_acc']:.2f}%")
    
    return model


def evaluate_model(model, data_loader, device):
    """评估模型"""
    all_preds = []
    all_targets = []
    
    model.eval()
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc='Evaluating'):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    return all_preds, all_targets


def main(args):
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载测试数据
    print("\n" + "="*60)
    print("Loading Test Dataset")
    print("="*60)
    
    _, _, test_loader = get_balanced_dataloader(
        data_dir=config.get('data_dir', './data/cifar10'),
        batch_size=config.get('batch_size', 128),
        img_size=config.get('img_size', 32),
        num_workers=config.get('num_workers', 4)
    )
    
    # 加载模型
    print("\n" + "="*60)
    print("Loading Model")
    print("="*60)
    
    model = load_model(
        model_path=args.model_path,
        model_type=config.get('model_type', 'vit_base'),
        config=config,
        device=device
    )
    
    # 评估
    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60)
    
    all_preds, all_targets = evaluate_model(model, test_loader, device)
    
    # 计算指标
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    metrics = calculate_metrics(all_preds, all_targets, num_classes=10)
    
    # 打印指标
    print_metrics(metrics, class_names)
    
    # 保存结果
    save_dir = config.get('save_dir', './experiments/results')
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存混淆矩阵
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(all_targets, all_preds, class_names, save_path=cm_path)
    
    # 保存指标到文件
    results_path = os.path.join(save_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {config.get('model_type', 'vit_base')}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Macro F1: {metrics['f1_macro']:.4f}\n")
        f.write(f"Weighted F1: {metrics['f1_weighted']:.4f}\n")
        f.write("\nPer-Class Metrics:\n")
        f.write("-"*60 + "\n")
        for i, name in enumerate(class_names):
            f.write(f"{name:<15} "
                   f"P: {metrics['precision_per_class'][i]:.4f} "
                   f"R: {metrics['recall_per_class'][i]:.4f} "
                   f"F1: {metrics['f1_per_class'][i]:.4f}\n")
    
    print(f"\nResults saved to {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ViT on CIFAR-10')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    args = parser.parse_args()
    main(args)