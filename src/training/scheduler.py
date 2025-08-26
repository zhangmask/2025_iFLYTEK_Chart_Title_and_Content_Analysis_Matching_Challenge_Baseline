import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, 
    ReduceLROnPlateau, CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts
)
from typing import Dict, Any, Optional, Union
import math
from loguru import logger


class LearningRateScheduler:
    """学习率调度器包装类"""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 scheduler_type: str = 'cosine',
                 scheduler_params: Optional[Dict[str, Any]] = None):
        """
        初始化学习率调度器
        
        Args:
            optimizer: PyTorch优化器
            scheduler_type: 调度器类型
            scheduler_params: 调度器参数
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params or {}
        
        self.scheduler = self._create_scheduler()
        self.step_count = 0
        
        logger.info(f"创建学习率调度器: {scheduler_type}")
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """创建具体的调度器"""
        if self.scheduler_type == 'step':
            return StepLR(
                self.optimizer,
                step_size=self.scheduler_params.get('step_size', 30),
                gamma=self.scheduler_params.get('gamma', 0.1)
            )
        
        elif self.scheduler_type == 'multistep':
            return MultiStepLR(
                self.optimizer,
                milestones=self.scheduler_params.get('milestones', [30, 60, 90]),
                gamma=self.scheduler_params.get('gamma', 0.1)
            )
        
        elif self.scheduler_type == 'exponential':
            return ExponentialLR(
                self.optimizer,
                gamma=self.scheduler_params.get('gamma', 0.95)
            )
        
        elif self.scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.scheduler_params.get('T_max', 100),
                eta_min=self.scheduler_params.get('eta_min', 0)
            )
        
        elif self.scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode=self.scheduler_params.get('mode', 'min'),
                factor=self.scheduler_params.get('factor', 0.5),
                patience=self.scheduler_params.get('patience', 10),
                threshold=self.scheduler_params.get('threshold', 1e-4),
                min_lr=self.scheduler_params.get('min_lr', 0)
            )
        
        elif self.scheduler_type == 'cyclic':
            return CyclicLR(
                self.optimizer,
                base_lr=self.scheduler_params.get('base_lr', 1e-5),
                max_lr=self.scheduler_params.get('max_lr', 1e-3),
                step_size_up=self.scheduler_params.get('step_size_up', 2000),
                mode=self.scheduler_params.get('mode', 'triangular')
            )
        
        elif self.scheduler_type == 'onecycle':
            return OneCycleLR(
                self.optimizer,
                max_lr=self.scheduler_params.get('max_lr', 1e-3),
                total_steps=self.scheduler_params.get('total_steps', 1000),
                pct_start=self.scheduler_params.get('pct_start', 0.3),
                anneal_strategy=self.scheduler_params.get('anneal_strategy', 'cos')
            )
        
        elif self.scheduler_type == 'cosine_restart':
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.scheduler_params.get('T_0', 10),
                T_mult=self.scheduler_params.get('T_mult', 1),
                eta_min=self.scheduler_params.get('eta_min', 0)
            )
        
        elif self.scheduler_type == 'warmup_cosine':
            return WarmupCosineScheduler(
                self.optimizer,
                warmup_epochs=self.scheduler_params.get('warmup_epochs', 5),
                total_epochs=self.scheduler_params.get('total_epochs', 100),
                min_lr=self.scheduler_params.get('min_lr', 0)
            )
        
        elif self.scheduler_type == 'linear_warmup':
            return LinearWarmupScheduler(
                self.optimizer,
                warmup_epochs=self.scheduler_params.get('warmup_epochs', 5),
                total_epochs=self.scheduler_params.get('total_epochs', 100)
            )
        
        else:
            raise ValueError(f"不支持的调度器类型: {self.scheduler_type}")
    
    def step(self, metric: Optional[float] = None):
        """执行一步调度"""
        if self.scheduler_type == 'plateau':
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()
        
        self.step_count += 1
    
    def get_last_lr(self) -> float:
        """获取最后的学习率"""
        return self.scheduler.get_last_lr()[0]
    
    def get_current_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        return {
            'scheduler': self.scheduler.state_dict(),
            'step_count': self.step_count,
            'scheduler_type': self.scheduler_type,
            'scheduler_params': self.scheduler_params
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态字典"""
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.step_count = state_dict['step_count']


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """带预热的余弦退火调度器"""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 warmup_epochs: int = 5,
                 total_epochs: int = 100,
                 min_lr: float = 0,
                 last_epoch: int = -1):
        """
        初始化
        
        Args:
            optimizer: 优化器
            warmup_epochs: 预热轮数
            total_epochs: 总轮数
            min_lr: 最小学习率
            last_epoch: 最后一轮
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 预热阶段：线性增长
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # 余弦退火阶段
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class LinearWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """线性预热调度器"""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 warmup_epochs: int = 5,
                 total_epochs: int = 100,
                 last_epoch: int = -1):
        """
        初始化
        
        Args:
            optimizer: 优化器
            warmup_epochs: 预热轮数
            total_epochs: 总轮数
            last_epoch: 最后一轮
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 预热阶段：线性增长
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # 线性衰减阶段
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [
                base_lr * (1 - progress)
                for base_lr in self.base_lrs
            ]


class AdaptiveLRScheduler:
    """自适应学习率调度器"""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 patience: int = 10,
                 factor: float = 0.5,
                 min_lr: float = 1e-7,
                 threshold: float = 1e-4):
        """
        初始化自适应调度器
        
        Args:
            optimizer: 优化器
            patience: 耐心值
            factor: 衰减因子
            min_lr: 最小学习率
            threshold: 改善阈值
        """
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.threshold = threshold
        
        self.best_metric = None
        self.wait_count = 0
        self.step_count = 0
        
        logger.info(f"创建自适应学习率调度器，耐心值: {patience}")
    
    def step(self, metric: float):
        """执行一步调度"""
        self.step_count += 1
        
        if self.best_metric is None:
            self.best_metric = metric
        elif metric < self.best_metric - self.threshold:
            # 指标有改善
            self.best_metric = metric
            self.wait_count = 0
        else:
            # 指标没有改善
            self.wait_count += 1
            
            if self.wait_count >= self.patience:
                # 降低学习率
                self._reduce_lr()
                self.wait_count = 0
    
    def _reduce_lr(self):
        """降低学习率"""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            
            if new_lr < old_lr:
                logger.info(f"学习率从 {old_lr:.2e} 降低到 {new_lr:.2e}")
    
    def get_current_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        return {
            'best_metric': self.best_metric,
            'wait_count': self.wait_count,
            'step_count': self.step_count,
            'patience': self.patience,
            'factor': self.factor,
            'min_lr': self.min_lr,
            'threshold': self.threshold
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态字典"""
        self.best_metric = state_dict['best_metric']
        self.wait_count = state_dict['wait_count']
        self.step_count = state_dict['step_count']
        self.patience = state_dict['patience']
        self.factor = state_dict['factor']
        self.min_lr = state_dict['min_lr']
        self.threshold = state_dict['threshold']


def create_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_config: Dict[str, Any]) -> Union[LearningRateScheduler, AdaptiveLRScheduler]:
    """创建学习率调度器的工厂函数"""
    scheduler_type = scheduler_config.get('type', 'cosine')
    
    if scheduler_type == 'adaptive':
        return AdaptiveLRScheduler(
            optimizer,
            patience=scheduler_config.get('patience', 10),
            factor=scheduler_config.get('factor', 0.5),
            min_lr=scheduler_config.get('min_lr', 1e-7),
            threshold=scheduler_config.get('threshold', 1e-4)
        )
    else:
        return LearningRateScheduler(
            optimizer,
            scheduler_type=scheduler_type,
            scheduler_params=scheduler_config.get('params', {})
        )