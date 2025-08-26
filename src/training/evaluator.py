import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns

from src.features.image_features import ImageFeatureExtractor
from src.features.text_features import TextFeatureExtractor
from src.features.similarity import SimilarityCalculator
from src.data.loader import DataLoader as ChartDataLoader


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, 
                 image_extractor: ImageFeatureExtractor,
                 text_extractor: TextFeatureExtractor,
                 similarity_calculator: SimilarityCalculator):
        """
        初始化评估器
        
        Args:
            image_extractor: 图像特征提取器
            text_extractor: 文本特征提取器
            similarity_calculator: 相似度计算器
        """
        self.image_extractor = image_extractor
        self.text_extractor = text_extractor
        self.similarity_calculator = similarity_calculator
        
        logger.info("模型评估器初始化完成")
    
    def evaluate_retrieval(self, 
                          image_paths: List[str], 
                          texts: List[str], 
                          ground_truth: List[int] = None,
                          batch_size: int = 32) -> Dict[str, float]:
        """
        评估检索性能
        
        Args:
            image_paths: 图像路径列表
            texts: 文本列表
            ground_truth: 真实匹配标签（可选）
            batch_size: 批处理大小
            
        Returns:
            评估指标字典
        """
        logger.info(f"开始评估检索性能，图像数量: {len(image_paths)}, 文本数量: {len(texts)}")
        
        # 提取特征
        image_features = self.image_extractor.extract_features_from_paths(image_paths, batch_size)
        text_features = self.text_extractor.extract_features(texts, batch_size)
        
        # 计算相似度矩阵
        similarity_matrix = self.similarity_calculator.compute_similarity_matrix(
            image_features, text_features
        )
        
        metrics = {}
        
        # 如果有真实标签，计算准确率等指标
        if ground_truth is not None:
            metrics.update(self._compute_accuracy_metrics(similarity_matrix, ground_truth))
        
        # 计算检索指标
        metrics.update(self._compute_retrieval_metrics(similarity_matrix, ground_truth))
        
        # 计算相似度统计
        metrics.update(self._compute_similarity_stats(similarity_matrix))
        
        logger.info(f"评估完成，主要指标: {metrics}")
        return metrics
    
    def _compute_accuracy_metrics(self, similarity_matrix: np.ndarray, 
                                ground_truth: List[int]) -> Dict[str, float]:
        """计算准确率相关指标"""
        metrics = {}
        
        # Top-1 准确率
        top1_predictions = np.argmax(similarity_matrix, axis=1)
        top1_accuracy = accuracy_score(ground_truth, top1_predictions)
        metrics['top1_accuracy'] = float(top1_accuracy)
        
        # Top-k 准确率
        for k in [3, 5, 10]:
            if similarity_matrix.shape[1] >= k:
                topk_predictions = np.argsort(similarity_matrix, axis=1)[:, -k:]
                topk_accuracy = np.mean([
                    gt in topk_pred for gt, topk_pred in zip(ground_truth, topk_predictions)
                ])
                metrics[f'top{k}_accuracy'] = float(topk_accuracy)
        
        # 精确率、召回率、F1分数（将问题转化为多分类）
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth, top1_predictions, average='weighted', zero_division=0
        )
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1_score'] = float(f1)
        
        return metrics
    
    def _compute_retrieval_metrics(self, similarity_matrix: np.ndarray, 
                                 ground_truth: List[int] = None) -> Dict[str, float]:
        """计算检索相关指标"""
        metrics = {}
        
        if ground_truth is not None:
            # 平均排名
            ranks = []
            for i, gt in enumerate(ground_truth):
                similarities = similarity_matrix[i]
                rank = np.sum(similarities > similarities[gt]) + 1
                ranks.append(rank)
            
            metrics['mean_rank'] = float(np.mean(ranks))
            metrics['median_rank'] = float(np.median(ranks))
            
            # MRR (Mean Reciprocal Rank)
            mrr = np.mean([1.0 / rank for rank in ranks])
            metrics['mrr'] = float(mrr)
            
            # NDCG@k (Normalized Discounted Cumulative Gain)
            for k in [5, 10]:
                if similarity_matrix.shape[1] >= k:
                    ndcg = self._compute_ndcg_at_k(similarity_matrix, ground_truth, k)
                    metrics[f'ndcg@{k}'] = float(ndcg)
        
        # 互相最佳匹配数量
        mutual_matches = self.similarity_calculator.find_mutual_best_matches(
            similarity_matrix[:, :similarity_matrix.shape[0]], 
            similarity_matrix[:similarity_matrix.shape[0], :].T
        )
        metrics['mutual_matches_count'] = len(mutual_matches)
        metrics['mutual_matches_ratio'] = len(mutual_matches) / similarity_matrix.shape[0]
        
        return metrics
    
    def _compute_ndcg_at_k(self, similarity_matrix: np.ndarray, 
                          ground_truth: List[int], k: int) -> float:
        """计算NDCG@k"""
        ndcg_scores = []
        
        for i, gt in enumerate(ground_truth):
            similarities = similarity_matrix[i]
            
            # 获取top-k预测
            top_k_indices = np.argsort(similarities)[::-1][:k]
            
            # 计算DCG
            dcg = 0.0
            for j, idx in enumerate(top_k_indices):
                if idx == gt:
                    dcg += 1.0 / np.log2(j + 2)  # j+2 because log2(1) = 0
            
            # 计算IDCG (理想情况下的DCG)
            idcg = 1.0 / np.log2(2)  # 最佳情况下，正确答案在第一位
            
            # 计算NDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores)
    
    def _compute_similarity_stats(self, similarity_matrix: np.ndarray) -> Dict[str, float]:
        """计算相似度统计信息"""
        return {
            'mean_similarity': float(np.mean(similarity_matrix)),
            'std_similarity': float(np.std(similarity_matrix)),
            'max_similarity': float(np.max(similarity_matrix)),
            'min_similarity': float(np.min(similarity_matrix)),
            'median_similarity': float(np.median(similarity_matrix))
        }
    
    def evaluate_on_dataset(self, data_loader: ChartDataLoader, 
                           split: str = 'test') -> Dict[str, float]:
        """
        在数据集上评估模型
        
        Args:
            data_loader: 数据加载器
            split: 数据集分割 ('train', 'val', 'test')
            
        Returns:
            评估指标字典
        """
        if split == 'train':
            df = data_loader.train_df
        elif split == 'val':
            df = data_loader.val_df if hasattr(data_loader, 'val_df') else None
        elif split == 'test':
            df = data_loader.test_df
        else:
            raise ValueError(f"不支持的数据集分割: {split}")
        
        if df is None or df.empty:
            logger.warning(f"数据集 {split} 为空")
            return {}
        
        # 获取图像路径和文本
        image_paths = df['Source'].tolist()
        
        if 'Caption' in df.columns:
            texts = df['Caption'].tolist()
            # 对于训练集，ground_truth就是索引对应关系
            ground_truth = list(range(len(texts)))
        else:
            # 测试集没有Caption，需要从训练集获取所有可能的文本
            train_texts = data_loader.get_all_captions()
            texts = train_texts
            ground_truth = None
        
        return self.evaluate_retrieval(image_paths, texts, ground_truth)
    
    def generate_confusion_matrix(self, 
                                image_paths: List[str], 
                                texts: List[str], 
                                ground_truth: List[int],
                                save_path: Optional[str] = None) -> np.ndarray:
        """
        生成混淆矩阵
        
        Args:
            image_paths: 图像路径列表
            texts: 文本列表
            ground_truth: 真实标签
            save_path: 保存路径（可选）
            
        Returns:
            混淆矩阵
        """
        # 提取特征并预测
        image_features = self.image_extractor.extract_features_from_paths(image_paths)
        text_features = self.text_extractor.extract_features(texts)
        similarity_matrix = self.similarity_calculator.compute_similarity_matrix(
            image_features, text_features
        )
        
        predictions = np.argmax(similarity_matrix, axis=1)
        
        # 计算混淆矩阵
        n_classes = len(texts)
        confusion_matrix = np.zeros((n_classes, n_classes))
        
        for true_label, pred_label in zip(ground_truth, predictions):
            confusion_matrix[true_label, pred_label] += 1
        
        # 可视化
        if save_path:
            plt.figure(figsize=(12, 10))
            sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"混淆矩阵已保存到: {save_path}")
        
        return confusion_matrix
    
    def analyze_failure_cases(self, 
                            image_paths: List[str], 
                            texts: List[str], 
                            ground_truth: List[int],
                            top_k: int = 10) -> List[Dict]:
        """
        分析失败案例
        
        Args:
            image_paths: 图像路径列表
            texts: 文本列表
            ground_truth: 真实标签
            top_k: 返回前k个最差案例
            
        Returns:
            失败案例列表
        """
        # 提取特征并计算相似度
        image_features = self.image_extractor.extract_features_from_paths(image_paths)
        text_features = self.text_extractor.extract_features(texts)
        similarity_matrix = self.similarity_calculator.compute_similarity_matrix(
            image_features, text_features
        )
        
        failure_cases = []
        
        for i, gt in enumerate(ground_truth):
            similarities = similarity_matrix[i]
            predicted = np.argmax(similarities)
            
            if predicted != gt:
                # 计算正确答案的排名
                rank = np.sum(similarities > similarities[gt]) + 1
                
                failure_case = {
                    'image_idx': i,
                    'image_path': image_paths[i],
                    'true_text_idx': gt,
                    'true_text': texts[gt],
                    'predicted_text_idx': predicted,
                    'predicted_text': texts[predicted],
                    'true_similarity': float(similarities[gt]),
                    'predicted_similarity': float(similarities[predicted]),
                    'rank': int(rank),
                    'similarity_diff': float(similarities[predicted] - similarities[gt])
                }
                failure_cases.append(failure_case)
        
        # 按相似度差异排序，返回最差的案例
        failure_cases.sort(key=lambda x: x['similarity_diff'], reverse=True)
        
        logger.info(f"发现 {len(failure_cases)} 个失败案例")
        return failure_cases[:top_k]
    
    def generate_evaluation_report(self, 
                                 data_loader: ChartDataLoader,
                                 save_dir: str) -> Dict[str, any]:
        """
        生成完整的评估报告
        
        Args:
            data_loader: 数据加载器
            save_dir: 保存目录
            
        Returns:
            评估报告字典
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            'model_info': {
                'image_extractor': self.image_extractor.get_model_info(),
                'text_extractor': self.text_extractor.get_model_info(),
                'similarity_calculator': self.similarity_calculator.get_config()
            },
            'evaluation_results': {}
        }
        
        # 评估训练集
        if hasattr(data_loader, 'train_df') and not data_loader.train_df.empty:
            logger.info("评估训练集...")
            train_metrics = self.evaluate_on_dataset(data_loader, 'train')
            report['evaluation_results']['train'] = train_metrics
        
        # 评估验证集
        if hasattr(data_loader, 'val_df') and data_loader.val_df is not None and not data_loader.val_df.empty:
            logger.info("评估验证集...")
            val_metrics = self.evaluate_on_dataset(data_loader, 'val')
            report['evaluation_results']['val'] = val_metrics
        
        # 评估测试集
        if hasattr(data_loader, 'test_df') and not data_loader.test_df.empty:
            logger.info("评估测试集...")
            test_metrics = self.evaluate_on_dataset(data_loader, 'test')
            report['evaluation_results']['test'] = test_metrics
        
        # 保存报告
        report_path = save_dir / 'evaluation_report.json'
        import json
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"评估报告已保存到: {report_path}")
        return report
    
    def compare_models(self, other_evaluators: List['ModelEvaluator'], 
                      data_loader: ChartDataLoader,
                      model_names: List[str] = None) -> pd.DataFrame:
        """
        比较多个模型的性能
        
        Args:
            other_evaluators: 其他评估器列表
            data_loader: 数据加载器
            model_names: 模型名称列表
            
        Returns:
            比较结果DataFrame
        """
        all_evaluators = [self] + other_evaluators
        
        if model_names is None:
            model_names = [f'Model_{i+1}' for i in range(len(all_evaluators))]
        
        comparison_results = []
        
        for evaluator, name in zip(all_evaluators, model_names):
            metrics = evaluator.evaluate_on_dataset(data_loader, 'test')
            metrics['model_name'] = name
            comparison_results.append(metrics)
        
        df = pd.DataFrame(comparison_results)
        return df