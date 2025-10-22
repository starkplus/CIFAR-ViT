"""
训练器主类
包含完整的训练和验证逻辑
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from utils.metrics import AverageMeter, accuracy
from utils.augmentation import MixUp, CutMix


class Trainer:
    """
    模型训练器
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 配置字典
        device: 设备
    """
    
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # 将模型移到设备
        self.model = self.model.to(device)
        
        # 损失函数
        self.criterion = self._get_criterion()
        
        # 优化器
        self.optimizer = self._get_optimizer()
        
        # 学习率调度器
        self.scheduler = self._get_scheduler()
        
        # 数据增强
        self.use_mixup = config.get('use_mixup', False)
        self.use_cutmix = config.get('use_cutmix', False)
        if self.use_mixup:
            self.mixup = MixUp(alpha=config.get('mixup_alpha', 1.0))
        if self.use_cutmix:
            self.cutmix = CutMix(alpha=config.get('cutmix_alpha', 1.0))
        
        # TensorBoard
        self.use_tensorboard = config.get('use_tensorboard', True)
        if self.use_tensorboard:
            log_dir = config.get('log_dir', './experiments/results/logs')
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # 最佳模型
        self.best_acc = 0.0
        self.best_epoch = 0
        
    def _get_criterion(self):
        """获取损失函数"""
        loss_type = self.config.get('loss', 'cross_entropy')
        
        if loss_type == 'cross_entropy':
            # 支持类别权重处理不平衡
            class_weights = self.config.get('class_weights', None)
            if class_weights is not None:
                class_weights = torch.tensor(class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif loss_type == 'label_smoothing':
            label_smoothing = self.config.get('label_smoothing', 0.1)
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return criterion
    
    def _get_optimizer(self):
        """获取优化器"""
        optimizer_type = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.0001)
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        return optimizer
    
    def _get_scheduler(self):
        """获取学习率调度器"""
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            T_max = self.config.get('epochs', 100)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            step_size = self.config.get('step_size', 30)
            gamma = self.config.get('gamma', 0.1)
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_type == 'multistep':
            milestones = self.config.get('milestones', [60, 80])
            gamma = self.config.get('gamma', 0.1)
            scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=gamma
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            batch_size = images.size(0)
            
            # 应用数据增强
            if self.use_mixup and np.random.rand() < 0.5:
                images, targets_a, targets_b, lam = self.mixup(images, targets)
                outputs = self.model(images)
                loss = lam * self.criterion(outputs, targets_a) + \
                       (1 - lam) * self.criterion(outputs, targets_b)
            elif self.use_cutmix and np.random.rand() < 0.5:
                images, targets_a, targets_b, lam = self.cutmix(images, targets)
                outputs = self.model(images)
                loss = lam * self.criterion(outputs, targets_a) + \
                       (1 - lam) * self.criterion(outputs, targets_b)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.config.get('clip_grad', False):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('clip_value', 1.0)
                )
            
            self.optimizer.step()
            
            # 计算准确率
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            
            # 更新统计
            losses.update(loss.item(), batch_size)
            top1.update(acc1, batch_size)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{top1.avg:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        return losses.avg, top1.avg
    
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                targets = targets.to(self.device)
                batch_size = images.size(0)
                
                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # 计算准确率
                acc1 = accuracy(outputs, targets, topk=(1,))[0]
                
                # 更新统计
                losses.update(loss.item(), batch_size)
                top1.update(acc1, batch_size)
        
        return losses.avg, top1.avg
    
    def train(self):
        """完整训练流程"""
        num_epochs = self.config.get('epochs', 100)
        save_dir = self.config.get('save_dir', './experiments/results/checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate(epoch)
            
            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # 保存历史
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # TensorBoard记录
            if self.use_tensorboard:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            # 打印信息
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_epoch = epoch
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': self.best_acc,
                    'config': self.config
                }, best_model_path)
                print(f"✓ Best model saved! (Acc: {val_acc:.2f}%)")
            
            # 定期保存检查点
            if epoch % self.config.get('save_freq', 10) == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config
                }, checkpoint_path)
        
        # 训练结束
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_acc:.2f}% (Epoch {self.best_epoch})")
        print(f"{'='*60}\n")
        
        if self.use_tensorboard:
            self.writer.close()
        
        return self.history