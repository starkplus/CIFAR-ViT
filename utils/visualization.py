"""
可视化工具
用于绘制训练曲线、注意力图、混淆矩阵等
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import os


def plot_training_curves(history, save_path=None):
    """
    绘制训练曲线（损失和准确率）
    
    Args:
        history: 包含train_loss, train_acc, val_loss, val_acc的字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        save_path: 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_class_distribution(labels, class_names, title='Class Distribution', save_path=None):
    """
    绘制类别分布
    
    Args:
        labels: 标签数组
        class_names: 类别名称列表
        title: 图表标题
        save_path: 保存路径
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(unique)), counts, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution saved to {save_path}")
    
    plt.show()


def visualize_attention_map(image, attention_weights, patch_size=4, save_path=None):
    """
    可视化注意力图
    
    Args:
        image: 原始图像 (C, H, W) 或 (H, W, C)
        attention_weights: 注意力权重 (num_heads, num_patches+1, num_patches+1)
        patch_size: patch大小
        save_path: 保存路径
    """
    # 确保图像是numpy数组
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    
    # 如果是(C, H, W)格式，转换为(H, W, C)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    # 反归一化
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    
    # 获取CLS token对所有patch的注意力
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.cpu().numpy()
    
    # 平均所有注意力头
    attention = attention_weights.mean(axis=0)  # (num_patches+1, num_patches+1)
    
    # 提取CLS token对patch的注意力
    cls_attention = attention[0, 1:]  # 跳过CLS token自己
    
    # 重塑为2D
    num_patches = int(np.sqrt(len(cls_attention)))
    attention_map = cls_attention.reshape(num_patches, num_patches)
    
    # 上采样到原始图像大小
    from scipy.ndimage import zoom
    H, W = image.shape[:2]
    attention_map_resized = zoom(attention_map, (H / num_patches, W / num_patches), order=1)
    
    # 绘制
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # 注意力图
    im = axes[1].imshow(attention_map_resized, cmap='jet', alpha=0.8)
    axes[1].set_title('Attention Map', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 叠加图
    axes[2].imshow(image)
    axes[2].imshow(attention_map_resized, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention map saved to {save_path}")
    
    plt.show()


def plot_model_comparison(results_dict, metric='accuracy', save_path=None):
    """
    比较不同模型的性能
    
    Args:
        results_dict: {model_name: {metric: value}} 格式的字典
        metric: 要比较的指标
        save_path: 保存路径
    """
    models = list(results_dict.keys())
    values = [results_dict[model][metric] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(models)), values, color='lightcoral', edgecolor='darkred', alpha=0.7)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.title(f'Model Comparison - {metric.capitalize()}', fontsize=14, fontweight='bold')
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to {save_path}")
    
    plt.show()


def plot_learning_rate_schedule(lr_history, save_path=None):
    """
    绘制学习率变化曲线
    
    Args:
        lr_history: 学习率历史列表
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lr_history, linewidth=2, color='green')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning rate schedule saved to {save_path}")
    
    plt.show()


if __name__ == '__main__':
    # 测试代码
    print("Testing visualization functions...")
    
    # 测试训练曲线
    history = {
        'train_loss': [2.3, 1.8, 1.5, 1.2, 1.0, 0.9, 0.8],
        'val_loss': [2.2, 1.9, 1.6, 1.4, 1.2, 1.1, 1.0],
        'train_acc': [20, 35, 45, 55, 65, 70, 75],
        'val_acc': [18, 32, 42, 52, 60, 65, 70]
    }
    plot_training_curves(history)
    
    # 测试类别分布
    labels = np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.4, 0.3, 0.15, 0.1, 0.05])
    class_names = ['Class A', 'Class B', 'Class C', 'Class D', 'Class E']
    plot_class_distribution(labels, class_names)