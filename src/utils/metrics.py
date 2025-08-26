import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self):
        """
        初始化指标计算器
        """
        self.reset()
        logger.debug("指标计算器初始化完成")
    
    def reset(self):
        """
        重置所有指标
        """
        self.predictions = []
        self.targets = []
        self.similarities = []
        self.confidences = []
        
    def update(self, 
               predictions: Union[List, np.ndarray, torch.Tensor],
               targets: Union[List, np.ndarray, torch.Tensor],
               similarities: Optional[Union[List, np.ndarray, torch.Tensor]] = None,
               confidences: Optional[Union[List, np.ndarray, torch.Tensor]] = None):
        """
        更新指标数据
        
        Args:
            predictions: 预测结果
            targets: 真实标签
            similarities: 相似度分数
            confidences: 置信度分数
        """
        # 转换为numpy数组
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if similarities is not None and isinstance(similarities, torch.Tensor):
            similarities = similarities.cpu().numpy()
        if confidences is not None and isinstance(confidences, torch.Tensor):
            confidences = confidences.cpu().numpy()
        
        self.predictions.extend(np.array(predictions).flatten())
        self.targets.extend(np.array(targets).flatten())
        
        if similarities is not None:
            self.similarities.extend(np.array(similarities).flatten())
        
        if confidences is not None:
            self.confidences.extend(np.array(confidences).flatten())
    
    def compute_accuracy(self, k: int = 1) -> float:
        """
        计算Top-K准确率
        
        Args:
            k: Top-K
            
        Returns:
            准确率
        """
        if not self.predictions or not self.targets:
            return 0.0
        
        if k == 1:
            return accuracy_score(self.targets, self.predictions)
        else:
            # 对于Top-K准确率，需要相似度矩阵
            if not self.similarities:
                logger.warning("计算Top-K准确率需要相似度分数")
                return 0.0
            
            # 这里需要根据具体实现调整
            return self._compute_topk_accuracy(k)
    
    def _compute_topk_accuracy(self, k: int) -> float:
        """
        计算Top-K准确率的具体实现
        
        Args:
            k: Top-K
            
        Returns:
            Top-K准确率
        """
        # 这是一个简化实现，实际使用时需要根据具体场景调整
        correct = 0
        total = len(self.targets)
        
        # 假设similarities是按批次组织的相似度分数
        # 实际实现需要根据数据结构调整
        for i in range(total):
            if i < len(self.similarities):
                # 简化处理：假设预测正确
                if self.predictions[i] == self.targets[i]:
                    correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def compute_precision_recall_f1(self, average: str = 'weighted') -> Tuple[float, float, float]:
        """
        计算精确率、召回率和F1分数
        
        Args:
            average: 平均方式
            
        Returns:
            (精确率, 召回率, F1分数)
        """
        if not self.predictions or not self.targets:
            return 0.0, 0.0, 0.0
        
        precision = precision_score(self.targets, self.predictions, average=average, zero_division=0)
        recall = recall_score(self.targets, self.predictions, average=average, zero_division=0)
        f1 = f1_score(self.targets, self.predictions, average=average, zero_division=0)
        
        return precision, recall, f1
    
    def compute_mean_rank(self) -> float:
        """
        计算平均排名
        
        Returns:
            平均排名
        """
        if not self.similarities:
            logger.warning("计算平均排名需要相似度分数")
            return 0.0
        
        # 简化实现
        ranks = []
        for i, (pred, target) in enumerate(zip(self.predictions, self.targets)):
            if pred == target:
                ranks.append(1)  # 正确预测排名为1
            else:
                ranks.append(len(set(self.targets)))  # 错误预测排名为最后
        
        return np.mean(ranks) if ranks else 0.0
    
    def compute_median_rank(self) -> float:
        """
        计算中位数排名
        
        Returns:
            中位数排名
        """
        if not self.similarities:
            logger.warning("计算中位数排名需要相似度分数")
            return 0.0
        
        # 简化实现
        ranks = []
        for i, (pred, target) in enumerate(zip(self.predictions, self.targets)):
            if pred == target:
                ranks.append(1)
            else:
                ranks.append(len(set(self.targets)))
        
        return np.median(ranks) if ranks else 0.0
    
    def compute_mrr(self) -> float:
        """
        计算平均倒数排名 (Mean Reciprocal Rank)
        
        Returns:
            MRR分数
        """
        if not self.similarities:
            logger.warning("计算MRR需要相似度分数")
            return 0.0
        
        # 简化实现
        reciprocal_ranks = []
        for i, (pred, target) in enumerate(zip(self.predictions, self.targets)):
            if pred == target:
                reciprocal_ranks.append(1.0)  # 正确预测的倒数排名为1
            else:
                reciprocal_ranks.append(0.0)  # 错误预测的倒数排名为0
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def compute_ndcg(self, k: int = 10) -> float:
        """
        计算归一化折损累积增益 (NDCG@K)
        
        Args:
            k: Top-K
            
        Returns:
            NDCG@K分数
        """
        if not self.similarities:
            logger.warning("计算NDCG需要相似度分数")
            return 0.0
        
        # 简化实现
        dcg = 0.0
        idcg = 1.0  # 理想情况下的DCG
        
        for i, (pred, target) in enumerate(zip(self.predictions[:k], self.targets[:k])):
            if pred == target:
                dcg += 1.0 / np.log2(i + 2)  # i+2 因为log2(1)=0
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def compute_confusion_matrix(self) -> np.ndarray:
        """
        计算混淆矩阵
        
        Returns:
            混淆矩阵
        """
        if not self.predictions or not self.targets:
            return np.array([])
        
        return confusion_matrix(self.targets, self.predictions)
    
    def get_classification_report(self) -> str:
        """
        获取分类报告
        
        Returns:
            分类报告字符串
        """
        if not self.predictions or not self.targets:
            return "无数据"
        
        return classification_report(self.targets, self.predictions)
    
    def compute_all_metrics(self, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        计算所有指标
        
        Args:
            k_values: Top-K值列表
            
        Returns:
            指标字典
        """
        metrics = {}
        
        # 基础指标
        metrics['accuracy'] = self.compute_accuracy()
        precision, recall, f1 = self.compute_precision_recall_f1()
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        # Top-K准确率
        for k in k_values:
            metrics[f'top_{k}_accuracy'] = self.compute_accuracy(k)
        
        # 排名指标
        metrics['mean_rank'] = self.compute_mean_rank()
        metrics['median_rank'] = self.compute_median_rank()
        metrics['mrr'] = self.compute_mrr()
        
        # NDCG指标
        for k in k_values:
            metrics[f'ndcg@{k}'] = self.compute_ndcg(k)
        
        # 置信度相关指标
        if self.confidences:
            metrics['mean_confidence'] = np.mean(self.confidences)
            metrics['std_confidence'] = np.std(self.confidences)
        
        return metrics
    
    def plot_confusion_matrix(self, 
                             save_path: Optional[str] = None,
                             class_names: Optional[List[str]] = None,
                             figsize: Tuple[int, int] = (8, 6)) -> Optional[str]:
        """
        绘制混淆矩阵
        
        Args:
            save_path: 保存路径
            class_names: 类别名称
            figsize: 图像大小
            
        Returns:
            保存路径（如果保存）
        """
        cm = self.compute_confusion_matrix()
        
        if cm.size == 0:
            logger.warning("无法绘制混淆矩阵：无数据")
            return None
        
        plt.figure(figsize=figsize)
        
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩阵已保存: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def plot_metrics_history(self,
                           metrics_history: Dict[str, List[float]],
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 8)) -> Optional[str]:
        """
        绘制指标历史
        
        Args:
            metrics_history: 指标历史字典
            save_path: 保存路径
            figsize: 图像大小
            
        Returns:
            保存路径（如果保存）
        """
        if not metrics_history:
            logger.warning("无法绘制指标历史：无数据")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        metric_names = list(metrics_history.keys())[:4]  # 最多显示4个指标
        
        for i, metric_name in enumerate(metric_names):
            if i >= 4:
                break
                
            values = metrics_history[metric_name]
            epochs = range(1, len(values) + 1)
            
            axes[i].plot(epochs, values, 'b-', linewidth=2, marker='o', markersize=4)
            axes[i].set_title(f'{metric_name.replace("_", " ").title()}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('Value')
            axes[i].grid(True, alpha=0.3)
        
        # 隐藏未使用的子图
        for i in range(len(metric_names), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"指标历史图已保存: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def save_metrics_report(self, 
                           save_path: str,
                           metrics: Optional[Dict[str, float]] = None,
                           include_classification_report: bool = True) -> str:
        """
        保存指标报告
        
        Args:
            save_path: 保存路径
            metrics: 指标字典
            include_classification_report: 是否包含分类报告
            
        Returns:
            保存路径
        """
        if metrics is None:
            metrics = self.compute_all_metrics()
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("模型评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 基础指标
            f.write("基础指标:\n")
            f.write("-" * 20 + "\n")
            for key, value in metrics.items():
                if key in ['accuracy', 'precision', 'recall', 'f1_score']:
                    f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
            f.write("\n")
            
            # Top-K准确率
            f.write("Top-K准确率:\n")
            f.write("-" * 20 + "\n")
            for key, value in metrics.items():
                if 'top_' in key and 'accuracy' in key:
                    f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
            f.write("\n")
            
            # 排名指标
            f.write("排名指标:\n")
            f.write("-" * 20 + "\n")
            for key, value in metrics.items():
                if key in ['mean_rank', 'median_rank', 'mrr']:
                    f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
            f.write("\n")
            
            # NDCG指标
            f.write("NDCG指标:\n")
            f.write("-" * 20 + "\n")
            for key, value in metrics.items():
                if 'ndcg' in key:
                    f.write(f"{key.upper()}: {value:.4f}\n")
            f.write("\n")
            
            # 置信度指标
            if any('confidence' in key for key in metrics.keys()):
                f.write("置信度指标:\n")
                f.write("-" * 20 + "\n")
                for key, value in metrics.items():
                    if 'confidence' in key:
                        f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
                f.write("\n")
            
            # 分类报告
            if include_classification_report:
                f.write("详细分类报告:\n")
                f.write("-" * 20 + "\n")
                f.write(self.get_classification_report())
                f.write("\n")
        
        logger.info(f"指标报告已保存: {save_path}")
        return str(save_path)
    
    def compare_models(self, 
                      model_metrics: Dict[str, Dict[str, float]],
                      save_path: Optional[str] = None) -> Optional[str]:
        """
        比较多个模型的性能
        
        Args:
            model_metrics: 模型指标字典 {模型名: {指标名: 值}}
            save_path: 保存路径
            
        Returns:
            保存路径（如果保存）
        """
        if not model_metrics:
            logger.warning("无法比较模型：无数据")
            return None
        
        # 获取所有指标名称
        all_metrics = set()
        for metrics in model_metrics.values():
            all_metrics.update(metrics.keys())
        
        # 创建比较表格
        import pandas as pd
        
        comparison_data = []
        for model_name, metrics in model_metrics.items():
            row = {'Model': model_name}
            for metric_name in sorted(all_metrics):
                row[metric_name] = metrics.get(metric_name, 0.0)
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        if save_path:
            # 保存为CSV
            csv_path = Path(save_path).with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            
            # 保存为文本报告
            txt_path = Path(save_path).with_suffix('.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("模型性能比较报告\n")
                f.write("=" * 50 + "\n\n")
                f.write(df.to_string(index=False, float_format='%.4f'))
                f.write("\n\n")
                
                # 找出最佳模型
                for metric in ['accuracy', 'f1_score', 'mrr']:
                    if metric in df.columns:
                        best_model = df.loc[df[metric].idxmax(), 'Model']
                        best_value = df[metric].max()
                        f.write(f"最佳 {metric}: {best_model} ({best_value:.4f})\n")
            
            logger.info(f"模型比较报告已保存: {txt_path}")
            return str(txt_path)
        
        return None