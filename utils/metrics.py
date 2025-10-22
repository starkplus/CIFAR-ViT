"""
评估指标计算
包含准确率、精确率、召回率、F1分数等
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class AverageMeter:
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    计算top-k准确率

    Args:
        output: (batch_size, num_classes) 模型输出logits
        target: (batch_size,) 真实标签
        topk: tuple, 计算哪些top-k

    Returns:
        list of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())

        return res


def calculate_metrics(all_preds, all_targets, num_classes=10):
    """
    计算各种分类指标

    Args:
        all_preds: numpy array of predictions
        all_targets: numpy array of ground truth labels
        num_classes: 类别数量

    Returns:
        dict containing various metrics
    """
    # 总体准确率
    acc = accuracy_score(all_targets, all_preds)

    # 精确率、召回率、F1分数（macro和weighted）
    precision_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    precision_weighted = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    # 每个类别的指标
    precision_per_class = precision_score(all_targets, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_targets, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_targets, all_preds, average=None, zero_division=0)

    # 混淆矩阵
    conf_matrix = confusion_matrix(all_targets, all_preds)

    metrics = {
        'accuracy': acc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': conf_matrix
    }

    return metrics


def print_metrics(metrics, class_names=None):
    """
    打印评估指标

    Args:
        metrics: calculate_metrics返回的字典
        class_names: 类别名称列表
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)

    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")

    print(f"\nMacro Average:")
    print(f"  Precision: {metrics['precision_macro']:.4f}")
    print(f"  Recall:    {metrics['recall_macro']:.4f}")
    print(f"  F1-Score:  {metrics['f1_macro']:.4f}")

    print(f"\nWeighted Average:")
    print(f"  Precision: {metrics['precision_weighted']:.4f}")
    print(f"  Recall:    {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score:  {metrics['f1_weighted']:.4f}")

    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)

    num_classes = len(metrics['precision_per_class'])
    for i in range(num_classes):
        class_name = class_names[i] if class_names else f"Class {i}"
        print(f"{class_name:<15} "
              f"{metrics['precision_per_class'][i]:<12.4f} "
              f"{metrics['recall_per_class'][i]:<12.4f} "
              f"{metrics['f1_per_class'][i]:<12.4f}")

    print("="*60)


def get_classification_report(all_preds, all_targets, class_names=None):
    """
    获取sklearn的分类报告
    """
    return classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        digits=4
    )


if __name__ == '__main__':
    # 测试代码
    print("Testing metrics...")

    # 模拟数据
    num_samples = 100
    num_classes = 10

    all_preds = np.random.randint(0, num_classes, num_samples)
    all_targets = np.random.randint(0, num_classes, num_samples)

    # 计算指标
    metrics = calculate_metrics(all_preds, all_targets, num_classes)

    # 打印指标
    class_names = [f"Class_{i}" for i in range(num_classes)]
    print_metrics(metrics, class_names)

    # 分类报告
    print("\nClassification Report:")
    print(get_classification_report(all_preds, all_targets, class_names))