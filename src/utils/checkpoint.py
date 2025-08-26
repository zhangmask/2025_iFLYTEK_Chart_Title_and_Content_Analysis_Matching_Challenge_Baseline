import torch
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import json
from loguru import logger


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, 
                 checkpoint_dir: str = "checkpoints",
                 max_checkpoints: int = 5,
                 save_best: bool = True,
                 monitor_metric: str = "val_accuracy",
                 mode: str = "max"):
        """
        初始化检查点管理器
        
        Args:
            checkpoint_dir: 检查点保存目录
            max_checkpoints: 最大保存检查点数量
            save_best: 是否保存最佳模型
            monitor_metric: 监控指标
            mode: 监控模式 ('max' 或 'min')
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.checkpoint_history = []
        
        # 加载历史记录
        self._load_history()
        
        logger.info(f"检查点管理器初始化完成，目录: {checkpoint_dir}")
    
    def save_checkpoint(self,
                       model_state: Dict[str, Any],
                       optimizer_state: Optional[Dict[str, Any]] = None,
                       scheduler_state: Optional[Dict[str, Any]] = None,
                       epoch: int = 0,
                       metrics: Optional[Dict[str, float]] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       filename: Optional[str] = None) -> str:
        """
        保存检查点
        
        Args:
            model_state: 模型状态字典
            optimizer_state: 优化器状态字典
            scheduler_state: 学习率调度器状态字典
            epoch: 训练轮次
            metrics: 评估指标
            metadata: 元数据
            filename: 文件名
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_epoch_{epoch}_{timestamp}.pth"
        
        filepath = self.checkpoint_dir / filename
        
        # 准备保存数据
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {},
            'metadata': metadata or {}
        }
        
        if optimizer_state is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer_state
        
        if scheduler_state is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler_state
        
        try:
            # 保存检查点
            torch.save(checkpoint_data, filepath)
            
            # 更新历史记录
            checkpoint_info = {
                'filepath': str(filepath),
                'epoch': epoch,
                'timestamp': checkpoint_data['timestamp'],
                'metrics': metrics or {},
                'filename': filename
            }
            
            self.checkpoint_history.append(checkpoint_info)
            
            # 检查是否为最佳模型
            is_best = False
            if self.save_best and metrics and self.monitor_metric in metrics:
                current_metric = metrics[self.monitor_metric]
                
                if self.mode == 'max':
                    is_best = current_metric > self.best_metric
                else:
                    is_best = current_metric < self.best_metric
                
                if is_best:
                    self.best_metric = current_metric
                    best_path = self.checkpoint_dir / "best_model.pth"
                    shutil.copy2(filepath, best_path)
                    logger.info(f"保存最佳模型: {best_path} (指标: {current_metric:.4f})")
            
            # 清理旧检查点
            self._cleanup_checkpoints()
            
            # 保存历史记录
            self._save_history()
            
            logger.info(f"检查点保存成功: {filepath} (轮次: {epoch})")
            if is_best:
                logger.info(f"新的最佳模型! {self.monitor_metric}: {current_metric:.4f}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
            raise
    
    def load_checkpoint(self, 
                       filepath: Optional[str] = None,
                       load_best: bool = False,
                       load_latest: bool = False) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            filepath: 检查点文件路径
            load_best: 是否加载最佳模型
            load_latest: 是否加载最新模型
            
        Returns:
            检查点数据
        """
        if load_best:
            filepath = self.checkpoint_dir / "best_model.pth"
        elif load_latest:
            filepath = self.get_latest_checkpoint()
        
        if filepath is None:
            raise ValueError("未指定检查点文件路径")
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"检查点文件不存在: {filepath}")
        
        try:
            checkpoint_data = torch.load(filepath, map_location='cpu')
            logger.info(f"检查点加载成功: {filepath}")
            
            # 记录加载信息
            epoch = checkpoint_data.get('epoch', 0)
            metrics = checkpoint_data.get('metrics', {})
            
            if metrics:
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                logger.info(f"加载模型信息 - 轮次: {epoch}, 指标: {metrics_str}")
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            raise
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        获取最新检查点路径
        
        Returns:
            最新检查点路径
        """
        if not self.checkpoint_history:
            return None
        
        # 按时间戳排序，获取最新的
        latest = max(self.checkpoint_history, key=lambda x: x['timestamp'])
        return latest['filepath']
    
    def get_best_checkpoint(self) -> Optional[str]:
        """
        获取最佳检查点路径
        
        Returns:
            最佳检查点路径
        """
        best_path = self.checkpoint_dir / "best_model.pth"
        return str(best_path) if best_path.exists() else None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        列出所有检查点
        
        Returns:
            检查点信息列表
        """
        return sorted(self.checkpoint_history, key=lambda x: x['epoch'])
    
    def delete_checkpoint(self, filepath: str) -> bool:
        """
        删除指定检查点
        
        Args:
            filepath: 检查点文件路径
            
        Returns:
            是否删除成功
        """
        filepath = Path(filepath)
        
        try:
            if filepath.exists():
                filepath.unlink()
                
                # 从历史记录中移除
                self.checkpoint_history = [
                    item for item in self.checkpoint_history 
                    if item['filepath'] != str(filepath)
                ]
                
                self._save_history()
                logger.info(f"检查点删除成功: {filepath}")
                return True
            else:
                logger.warning(f"检查点文件不存在: {filepath}")
                return False
                
        except Exception as e:
            logger.error(f"删除检查点失败: {e}")
            return False
    
    def cleanup_all_checkpoints(self, keep_best: bool = True):
        """
        清理所有检查点
        
        Args:
            keep_best: 是否保留最佳模型
        """
        try:
            for checkpoint_info in self.checkpoint_history.copy():
                filepath = Path(checkpoint_info['filepath'])
                if filepath.exists():
                    filepath.unlink()
            
            self.checkpoint_history.clear()
            
            if not keep_best:
                best_path = self.checkpoint_dir / "best_model.pth"
                if best_path.exists():
                    best_path.unlink()
                    logger.info("最佳模型已删除")
            
            self._save_history()
            logger.info("所有检查点清理完成")
            
        except Exception as e:
            logger.error(f"清理检查点失败: {e}")
    
    def _cleanup_checkpoints(self):
        """
        清理旧检查点，保持最大数量限制
        """
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        # 按时间戳排序，删除最旧的
        sorted_checkpoints = sorted(
            self.checkpoint_history, 
            key=lambda x: x['timestamp']
        )
        
        to_remove = sorted_checkpoints[:-self.max_checkpoints]
        
        for checkpoint_info in to_remove:
            filepath = Path(checkpoint_info['filepath'])
            try:
                if filepath.exists():
                    filepath.unlink()
                    logger.debug(f"删除旧检查点: {filepath}")
                
                self.checkpoint_history.remove(checkpoint_info)
                
            except Exception as e:
                logger.warning(f"删除旧检查点失败: {e}")
    
    def _save_history(self):
        """
        保存历史记录
        """
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'checkpoint_history': self.checkpoint_history,
                    'best_metric': self.best_metric,
                    'monitor_metric': self.monitor_metric,
                    'mode': self.mode
                }, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"保存历史记录失败: {e}")
    
    def _load_history(self):
        """
        加载历史记录
        """
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        
        if not history_file.exists():
            return
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.checkpoint_history = data.get('checkpoint_history', [])
            self.best_metric = data.get('best_metric', 
                                      float('-inf') if self.mode == 'max' else float('inf'))
            
            # 验证文件是否存在
            valid_history = []
            for checkpoint_info in self.checkpoint_history:
                if Path(checkpoint_info['filepath']).exists():
                    valid_history.append(checkpoint_info)
                else:
                    logger.debug(f"检查点文件不存在，从历史记录中移除: {checkpoint_info['filepath']}")
            
            self.checkpoint_history = valid_history
            
            logger.debug(f"历史记录加载完成，共 {len(self.checkpoint_history)} 个检查点")
            
        except Exception as e:
            logger.warning(f"加载历史记录失败: {e}")
            self.checkpoint_history = []
    
    def get_checkpoint_info(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        获取检查点信息
        
        Args:
            filepath: 检查点文件路径
            
        Returns:
            检查点信息
        """
        for checkpoint_info in self.checkpoint_history:
            if checkpoint_info['filepath'] == filepath:
                return checkpoint_info
        return None
    
    def resume_training(self, 
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       load_best: bool = False) -> Tuple[int, Dict[str, float]]:
        """
        恢复训练
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            load_best: 是否加载最佳模型
            
        Returns:
            (起始轮次, 指标字典)
        """
        checkpoint_data = self.load_checkpoint(load_best=load_best, load_latest=not load_best)
        
        # 加载模型状态
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # 加载调度器状态
        if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        
        start_epoch = checkpoint_data.get('epoch', 0) + 1
        metrics = checkpoint_data.get('metrics', {})
        
        logger.info(f"训练恢复成功，从第 {start_epoch} 轮开始")
        
        return start_epoch, metrics
    
    def export_checkpoint(self, 
                         filepath: str, 
                         export_path: str,
                         include_optimizer: bool = False,
                         include_scheduler: bool = False):
        """
        导出检查点（仅包含模型权重）
        
        Args:
            filepath: 源检查点路径
            export_path: 导出路径
            include_optimizer: 是否包含优化器状态
            include_scheduler: 是否包含调度器状态
        """
        checkpoint_data = self.load_checkpoint(filepath)
        
        export_data = {
            'model_state_dict': checkpoint_data['model_state_dict'],
            'epoch': checkpoint_data.get('epoch', 0),
            'metrics': checkpoint_data.get('metrics', {}),
            'metadata': checkpoint_data.get('metadata', {}),
            'timestamp': checkpoint_data.get('timestamp')
        }
        
        if include_optimizer and 'optimizer_state_dict' in checkpoint_data:
            export_data['optimizer_state_dict'] = checkpoint_data['optimizer_state_dict']
        
        if include_scheduler and 'scheduler_state_dict' in checkpoint_data:
            export_data['scheduler_state_dict'] = checkpoint_data['scheduler_state_dict']
        
        try:
            torch.save(export_data, export_path)
            logger.info(f"检查点导出成功: {export_path}")
            
        except Exception as e:
            logger.error(f"导出检查点失败: {e}")
            raise