import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.features.image_features import ImageFeatureExtractor
from src.features.text_features import TextFeatureExtractor
from src.features.similarity import SimilarityCalculator
from src.data.loader import DataLoader as ChartDataLoader
from src.data.preprocessor import ImagePreprocessor, TextPreprocessor


class ChartPredictor:
    """图表内容预测器"""
    
    def __init__(self, 
                 image_extractor: ImageFeatureExtractor,
                 text_extractor: TextFeatureExtractor,
                 similarity_calculator: SimilarityCalculator,
                 device: str = 'auto'):
        """
        初始化预测器
        
        Args:
            image_extractor: 图像特征提取器
            text_extractor: 文本特征提取器
            similarity_calculator: 相似度计算器
            device: 设备类型
        """
        self.image_extractor = image_extractor
        self.text_extractor = text_extractor
        self.similarity_calculator = similarity_calculator
        
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # 移动模型到指定设备
        self.image_extractor.to(self.device)
        self.text_extractor.to(self.device)
        
        # 设置为评估模式
        self.image_extractor.eval()
        self.text_extractor.eval()
        
        logger.info(f"预测器初始化完成，使用设备: {self.device}")
    
    def predict_single(self, 
                      image_path: str, 
                      candidate_texts: List[str],
                      return_scores: bool = False) -> Union[int, Tuple[int, List[float]]]:
        """
        对单个图像进行预测
        
        Args:
            image_path: 图像路径
            candidate_texts: 候选文本列表
            return_scores: 是否返回相似度分数
            
        Returns:
            最佳匹配的文本索引，可选返回所有相似度分数
        """
        with torch.no_grad():
            # 提取图像特征
            image_features = self.image_extractor.extract_features_from_paths([image_path])
            
            # 提取文本特征
            text_features = self.text_extractor.extract_features(candidate_texts)
            
            # 计算相似度
            similarities = self.similarity_calculator.compute_similarity(
                image_features, text_features
            )
            
            # 获取最佳匹配
            best_idx = int(np.argmax(similarities[0]))
            
            if return_scores:
                return best_idx, similarities[0].tolist()
            else:
                return best_idx
    
    def predict_batch(self, 
                     image_paths: List[str], 
                     candidate_texts: List[str],
                     batch_size: int = 32,
                     return_scores: bool = False) -> Union[List[int], Tuple[List[int], np.ndarray]]:
        """
        批量预测
        
        Args:
            image_paths: 图像路径列表
            candidate_texts: 候选文本列表
            batch_size: 批处理大小
            return_scores: 是否返回相似度分数
            
        Returns:
            最佳匹配的文本索引列表，可选返回相似度矩阵
        """
        logger.info(f"开始批量预测，图像数量: {len(image_paths)}, 候选文本数量: {len(candidate_texts)}")
        
        with torch.no_grad():
            # 提取图像特征
            image_features = self.image_extractor.extract_features_from_paths(
                image_paths, batch_size
            )
            
            # 提取文本特征
            text_features = self.text_extractor.extract_features(
                candidate_texts, batch_size
            )
            
            # 计算相似度矩阵
            similarity_matrix = self.similarity_calculator.compute_similarity_matrix(
                image_features, text_features
            )
            
            # 获取最佳匹配
            best_indices = np.argmax(similarity_matrix, axis=1).tolist()
            
            logger.info(f"批量预测完成，预测结果数量: {len(best_indices)}")
            
            if return_scores:
                return best_indices, similarity_matrix
            else:
                return best_indices
    
    def predict_with_confidence(self, 
                               image_paths: List[str], 
                               candidate_texts: List[str],
                               confidence_threshold: float = 0.5,
                               batch_size: int = 32) -> List[Dict[str, any]]:
        """
        带置信度的预测
        
        Args:
            image_paths: 图像路径列表
            candidate_texts: 候选文本列表
            confidence_threshold: 置信度阈值
            batch_size: 批处理大小
            
        Returns:
            预测结果列表，包含索引、置信度等信息
        """
        predictions, similarity_matrix = self.predict_batch(
            image_paths, candidate_texts, batch_size, return_scores=True
        )
        
        results = []
        for i, (pred_idx, similarities) in enumerate(zip(predictions, similarity_matrix)):
            # 计算置信度（最高分与第二高分的差值）
            sorted_scores = np.sort(similarities)[::-1]
            confidence = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else float(sorted_scores[0])
            
            # 获取top-k候选
            top_k_indices = np.argsort(similarities)[::-1][:5]
            top_k_scores = similarities[top_k_indices]
            
            result = {
                'image_idx': i,
                'image_path': image_paths[i],
                'predicted_text_idx': pred_idx,
                'predicted_text': candidate_texts[pred_idx],
                'confidence': confidence,
                'max_similarity': float(similarities[pred_idx]),
                'is_confident': confidence > confidence_threshold,
                'top_k_candidates': [
                    {
                        'text_idx': int(idx),
                        'text': candidate_texts[idx],
                        'similarity': float(score)
                    }
                    for idx, score in zip(top_k_indices, top_k_scores)
                ]
            }
            results.append(result)
        
        return results
    
    def predict_on_dataset(self, 
                          data_loader,  # 可以是ChartDataLoader或DataLoader
                          split: str = 'test',
                          batch_size: int = 32) -> pd.DataFrame:
        """
        在数据集上进行预测
        
        Args:
            data_loader: 数据加载器（ChartDataLoader或DataLoader）
            split: 数据集分割
            batch_size: 批处理大小
            
        Returns:
            预测结果DataFrame
        """
        # 处理不同类型的data_loader
        if hasattr(data_loader, 'test_df'):  # ChartDataLoader
            if split == 'test':
                df = data_loader.test_df
                candidate_texts = data_loader.get_candidate_captions()
            elif split == 'train':
                df = data_loader.train_df
                candidate_texts = df['Caption'].tolist()
            else:
                raise ValueError(f"不支持的数据集分割: {split}")
        else:  # 原始DataLoader类
            if split == 'test':
                df = data_loader.test_df
                candidate_texts = data_loader.get_candidate_captions()
            elif split == 'train':
                df = data_loader.train_df
                candidate_texts = df['Caption'].tolist()
            else:
                raise ValueError(f"不支持的数据集分割: {split}")
        
        if df.empty:
            logger.warning(f"数据集 {split} 为空")
            return pd.DataFrame()
        
        image_paths = df['Source'].tolist()
        
        # 执行预测
        predictions = self.predict_batch(image_paths, candidate_texts, batch_size)
        
        # 创建结果DataFrame
        result_df = df.copy()
        result_df['predicted_caption_idx'] = predictions
        result_df['predicted_caption'] = [candidate_texts[idx] for idx in predictions]
        
        return result_df
    
    def predict_with_ensemble(self, 
                             image_paths: List[str], 
                             candidate_texts: List[str],
                             ensemble_methods: List[str] = ['mean', 'max'],
                             batch_size: int = 32) -> List[int]:
        """
        集成预测
        
        Args:
            image_paths: 图像路径列表
            candidate_texts: 候选文本列表
            ensemble_methods: 集成方法列表
            batch_size: 批处理大小
            
        Returns:
            集成预测结果
        """
        # 获取基础相似度矩阵
        _, similarity_matrix = self.predict_batch(
            image_paths, candidate_texts, batch_size, return_scores=True
        )
        
        ensemble_scores = np.zeros_like(similarity_matrix)
        
        for method in ensemble_methods:
            if method == 'mean':
                # 简单平均
                ensemble_scores += similarity_matrix
            elif method == 'max':
                # 最大值
                ensemble_scores = np.maximum(ensemble_scores, similarity_matrix)
            elif method == 'weighted':
                # 加权平均（这里使用简单权重）
                ensemble_scores += similarity_matrix * 0.8
        
        # 归一化
        ensemble_scores /= len(ensemble_methods)
        
        # 获取最佳匹配
        predictions = np.argmax(ensemble_scores, axis=1).tolist()
        
        return predictions
    
    def predict_with_tta(self, 
                        image_paths: List[str], 
                        candidate_texts: List[str],
                        tta_transforms: Optional[List] = None,
                        batch_size: int = 32) -> List[int]:
        """
        测试时增强(TTA)预测
        
        Args:
            image_paths: 图像路径列表
            candidate_texts: 候选文本列表
            tta_transforms: TTA变换列表
            batch_size: 批处理大小
            
        Returns:
            TTA预测结果
        """
        if tta_transforms is None:
            # 默认TTA变换：原图 + 水平翻转
            tta_transforms = ['original', 'hflip']
        
        all_predictions = []
        
        for transform in tta_transforms:
            # 这里简化处理，实际应该在ImagePreprocessor中实现不同的变换
            predictions, similarity_matrix = self.predict_batch(
                image_paths, candidate_texts, batch_size, return_scores=True
            )
            all_predictions.append(similarity_matrix)
        
        # 平均所有TTA结果
        ensemble_matrix = np.mean(all_predictions, axis=0)
        final_predictions = np.argmax(ensemble_matrix, axis=1).tolist()
        
        return final_predictions
    
    def save_predictions(self, 
                        predictions: Union[List[int], pd.DataFrame], 
                        save_path: str,
                        format: str = 'csv'):
        """
        保存预测结果
        
        Args:
            predictions: 预测结果
            save_path: 保存路径
            format: 保存格式
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(predictions, list):
            # 简单列表格式
            df = pd.DataFrame({'prediction': predictions})
        else:
            # DataFrame格式
            df = predictions
        
        if format == 'csv':
            df.to_csv(save_path, index=False)
        elif format == 'json':
            df.to_json(save_path, orient='records', indent=2)
        elif format == 'pickle':
            df.to_pickle(save_path)
        else:
            raise ValueError(f"不支持的保存格式: {format}")
        
        logger.info(f"预测结果已保存到: {save_path}")
    
    def load_model_weights(self, checkpoint_path: str):
        """
        加载模型权重
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'image_extractor' in checkpoint:
            self.image_extractor.load_state_dict(checkpoint['image_extractor'])
        
        if 'text_extractor' in checkpoint:
            self.text_extractor.load_state_dict(checkpoint['text_extractor'])
        
        if 'similarity_calculator' in checkpoint:
            self.similarity_calculator.load_state_dict(checkpoint['similarity_calculator'])
        
        logger.info(f"模型权重已从 {checkpoint_path} 加载")
    
    def get_model_info(self) -> Dict[str, any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'device': self.device,
            'image_extractor': self.image_extractor.get_model_info(),
            'text_extractor': self.text_extractor.get_model_info(),
            'similarity_calculator': self.similarity_calculator.get_config()
        }