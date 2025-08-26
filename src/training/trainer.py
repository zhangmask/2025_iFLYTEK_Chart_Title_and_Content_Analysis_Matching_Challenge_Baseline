import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
from tqdm import tqdm
from loguru import logger

from config import TRAINING_CONFIG, PATHS
from src.features.image_features import ImageFeatureExtractor
from src.features.text_features import TextFeatureExtractor
from src.features.similarity import SimilarityCalculator, ContrastiveLoss
from src.training.evaluator import ModelEvaluator


class CrossModalTrainer:
    """跨模态模型训练器"""
    
    def __init__(self, 
                 image_extractor: ImageFeatureExtractor,
                 text_extractor: TextFeatureExtractor,
                 similarity_calculator: SimilarityCalculator,
                 config: dict = None):
        """
        初始化训练器
        
        Args:
            image_extractor: 图像特征提取器
            text_extractor: 文本特征提取器
            similarity_calculator: 相似度计算器
            config: 训练配置
        """
        self.image_extractor = image_extractor
        self.text_extractor = text_extractor
        self.similarity_calculator = similarity_calculator
        
        # 训练配置
        self.config = config or TRAINING_CONFIG
        self.device = torch.device(self.config['device'])
        
        # 损失函数
        self.contrastive_loss = ContrastiveLoss(temperature=self.config.get('temperature', 0.07))
        
        # 优化器
        self.optimizer = None
        self.scheduler = None
        
        # 训练状态
        self.current_epoch = 0
        self.best_score = 0.0
        self.train_losses = []
        self.val_scores = []
        
        # 评估器
        self.evaluator = ModelEvaluator(image_extractor, text_extractor, similarity_calculator)
        
        logger.info("跨模态训练器初始化完成")
    
    def setup_optimizer(self, learning_rate: float = None, weight_decay: float = None):
        """设置优化器和学习率调度器"""
        lr = learning_rate or self.config.get('learning_rate', 1e-4)
        wd = weight_decay or self.config.get('weight_decay', 1e-5)
        
        # 收集所有需要训练的参数
        params = []
        
        # 图像特征提取器参数
        if self.config.get('train_image_extractor', True):
            params.extend(self.image_extractor.parameters())
        
        # 文本特征提取器参数
        if self.config.get('train_text_extractor', True):
            if hasattr(self.text_extractor, 'projection') and self.text_extractor.projection is not None:
                params.extend(self.text_extractor.projection.parameters())
            
            # 训练主模型
            if hasattr(self.text_extractor, 'model'):
                params.extend(self.text_extractor.model.parameters())
        
        # 相似度网络参数
        if (hasattr(self.similarity_calculator, 'similarity_net') and 
            self.similarity_calculator.similarity_method == 'learned'):
            params.extend(self.similarity_calculator.similarity_net.parameters())
        
        # 创建优化器
        optimizer_type = self.config.get('optimizer', 'adamw')
        if optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(params, lr=lr, weight_decay=wd)
        elif optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=wd)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        # 学习率调度器
        scheduler_type = self.config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.get('epochs', 100)
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        
        logger.info(f"优化器设置完成: {optimizer_type}, 学习率: {lr}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.image_extractor.train()
        self.text_extractor.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # 获取批次数据
                images = batch['image'].to(self.device)
                texts = batch['captions']
                
                # 前向传播
                self.optimizer.zero_grad()
                
                # 提取特征
                image_features = self.image_extractor(images)
                text_features = self.text_extractor(texts)
                
                # 计算损失
                loss = self.contrastive_loss(image_features, text_features)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                if self.config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.get_trainable_parameters(), 
                        self.config['grad_clip']
                    )
                
                self.optimizer.step()
                
                # 更新统计
                total_loss += loss.item()
                num_batches += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss / num_batches:.4f}"
                })
                
            except Exception as e:
                logger.error(f"训练批次 {batch_idx} 失败: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证模型"""
        self.image_extractor.eval()
        self.text_extractor.eval()
        
        all_image_features = []
        all_text_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                try:
                    images = batch['image'].to(self.device)
                    texts = batch['captions']
                    labels = batch.get('label', None)
                    
                    # 提取特征
                    image_features = self.image_extractor(images)
                    text_features = self.text_extractor(texts)
                    
                    all_image_features.append(image_features.cpu().numpy())
                    all_text_features.append(text_features.cpu().numpy())
                    
                    if labels is not None:
                        all_labels.extend(labels.tolist())
                    
                except Exception as e:
                    logger.error(f"验证批次失败: {e}")
                    continue
        
        # 合并所有特征
        if all_image_features:
            image_features = np.vstack(all_image_features)
            text_features = np.vstack(all_text_features)
            
            # 计算评估指标
            if all_labels:
                metrics = self.similarity_calculator.compute_retrieval_metrics(
                    image_features, text_features, all_labels
                )
            else:
                # 如果没有标签，计算基本的相似度统计
                similarity_matrix = self.similarity_calculator.compute_similarity_matrix(
                    image_features, text_features
                )
                metrics = {
                    'mean_similarity': float(np.mean(similarity_matrix)),
                    'max_similarity': float(np.max(similarity_matrix)),
                    'min_similarity': float(np.min(similarity_matrix))
                }
        else:
            metrics = {'error': 'No valid validation data'}
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, 
              epochs: int = None, save_dir: str = None) -> Dict[str, List[float]]:
        """完整训练流程"""
        epochs = epochs or self.config.get('epochs', 100)
        save_dir = Path(save_dir or PATHS['models'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置优化器
        if self.optimizer is None:
            self.setup_optimizer()
        
        logger.info(f"开始训练，共 {epochs} 个epoch")
        
        training_history = {
            'train_loss': [],
            'val_metrics': []
        }
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            training_history['train_loss'].append(train_loss)
            
            # 验证
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                training_history['val_metrics'].append(val_metrics)
            
            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics and 'top1_accuracy' in val_metrics:
                        self.scheduler.step(val_metrics['top1_accuracy'])
                else:
                    self.scheduler.step()
            
            # 记录日志
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch + 1}/{epochs} - "
                       f"Loss: {train_loss:.4f} - "
                       f"Time: {epoch_time:.2f}s")
            
            if val_metrics:
                logger.info(f"Validation metrics: {val_metrics}")
            
            # 保存最佳模型
            current_score = val_metrics.get('top1_accuracy', train_loss)
            if val_metrics and 'top1_accuracy' in val_metrics:
                is_best = current_score > self.best_score
            else:
                is_best = current_score < self.best_score  # 对于loss，越小越好
            
            if is_best:
                self.best_score = current_score
                self.save_checkpoint(save_dir / 'best_model.pt', is_best=True)
            
            # 定期保存检查点
            if (epoch + 1) % self.config.get('save_freq', 10) == 0:
                self.save_checkpoint(save_dir / f'checkpoint_epoch_{epoch + 1}.pt')
        
        logger.info(f"训练完成，最佳分数: {self.best_score:.4f}")
        return training_history
    
    def save_checkpoint(self, save_path: Path, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'best_score': self.best_score,
            'image_extractor_state_dict': self.image_extractor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_scores': self.val_scores
        }
        
        # 保存文本特征提取器（如果有投影层）
        if hasattr(self.text_extractor, 'projection') and self.text_extractor.projection is not None:
            checkpoint['text_projection_state_dict'] = self.text_extractor.projection.state_dict()
        
        # 保存相似度网络（如果是学习的）
        if (hasattr(self.similarity_calculator, 'similarity_net') and 
            self.similarity_calculator.similarity_method == 'learned'):
            checkpoint['similarity_net_state_dict'] = self.similarity_calculator.similarity_net.state_dict()
        
        torch.save(checkpoint, save_path)
        
        if is_best:
            logger.info(f"最佳模型已保存到: {save_path}")
        else:
            logger.info(f"检查点已保存到: {save_path}")
    
    def load_checkpoint(self, load_path: Path):
        """加载检查点"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # 加载模型状态
        self.image_extractor.load_state_dict(checkpoint['image_extractor_state_dict'])
        
        if 'text_projection_state_dict' in checkpoint:
            if hasattr(self.text_extractor, 'projection') and self.text_extractor.projection is not None:
                self.text_extractor.projection.load_state_dict(checkpoint['text_projection_state_dict'])
        
        if 'similarity_net_state_dict' in checkpoint:
            if hasattr(self.similarity_calculator, 'similarity_net'):
                self.similarity_calculator.similarity_net.load_state_dict(checkpoint['similarity_net_state_dict'])
        
        # 加载训练状态
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_score = checkpoint.get('best_score', 0.0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_scores = checkpoint.get('val_scores', [])
        
        # 加载优化器和调度器状态
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"检查点已从 {load_path} 加载")
    
    def get_trainable_parameters(self):
        """获取所有可训练参数"""
        params = []
        
        if self.config.get('train_image_extractor', True):
            params.extend(self.image_extractor.parameters())
        
        if self.config.get('train_text_extractor', True):
            if hasattr(self.text_extractor, 'projection') and self.text_extractor.projection is not None:
                params.extend(self.text_extractor.projection.parameters())
            
            if hasattr(self.text_extractor, 'model'):
                params.extend(self.text_extractor.model.parameters())
        
        if (hasattr(self.similarity_calculator, 'similarity_net') and 
            self.similarity_calculator.similarity_method == 'learned'):
            params.extend(self.similarity_calculator.similarity_net.parameters())
        
        return params
    
    def get_training_info(self) -> dict:
        """获取训练信息"""
        total_params = sum(p.numel() for p in self.get_trainable_parameters())
        
        return {
            'current_epoch': self.current_epoch,
            'best_score': self.best_score,
            'total_trainable_params': total_params,
            'device': str(self.device),
            'config': self.config
        }