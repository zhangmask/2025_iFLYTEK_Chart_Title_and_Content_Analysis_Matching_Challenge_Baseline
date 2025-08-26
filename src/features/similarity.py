import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Union, Optional
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from loguru import logger

from config import MODEL_CONFIG


class SimilarityCalculator:
    """跨模态相似度计算器"""
    
    def __init__(self, similarity_method: str = 'cosine', temperature: float = 0.07):
        """
        初始化相似度计算器
        
        Args:
            similarity_method: 相似度计算方法 ('cosine', 'euclidean', 'dot_product', 'learned')
            temperature: 温度参数，用于缩放相似度分数
        """
        self.similarity_method = similarity_method
        self.temperature = temperature
        
        # 从配置获取参数
        config = MODEL_CONFIG.get('cross_modal', {})
        self.similarity_method = config.get('similarity_method', similarity_method)
        self.temperature = config.get('temperature', temperature)
        
        # 如果使用学习的相似度，初始化神经网络
        if self.similarity_method == 'learned':
            self._init_learned_similarity()
        
        logger.info(f"相似度计算器初始化完成: {self.similarity_method}, 温度: {self.temperature}")
    
    def _init_learned_similarity(self):
        """初始化学习的相似度网络"""
        feature_dim = MODEL_CONFIG['image_model']['feature_dim']
        
        self.similarity_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        device = torch.device(MODEL_CONFIG.get('device', 'cpu'))
        self.similarity_net.to(device)
    
    def compute_similarity_matrix(self, image_features: np.ndarray, 
                                text_features: np.ndarray) -> np.ndarray:
        """
        计算图像和文本特征之间的相似度矩阵
        
        Args:
            image_features: 图像特征矩阵 (N, D)
            text_features: 文本特征矩阵 (M, D)
            
        Returns:
            相似度矩阵 (N, M)
        """
        if self.similarity_method == 'cosine':
            similarity_matrix = cosine_similarity(image_features, text_features)
        
        elif self.similarity_method == 'euclidean':
            # 欧几里得距离转换为相似度
            distances = cdist(image_features, text_features, metric='euclidean')
            similarity_matrix = 1 / (1 + distances)
        
        elif self.similarity_method == 'dot_product':
            similarity_matrix = np.dot(image_features, text_features.T)
        
        elif self.similarity_method == 'learned':
            similarity_matrix = self._compute_learned_similarity(image_features, text_features)
        
        else:
            raise ValueError(f"不支持的相似度方法: {self.similarity_method}")
        
        # 应用温度缩放
        similarity_matrix = similarity_matrix / self.temperature
        
        return similarity_matrix
    
    def _compute_learned_similarity(self, image_features: np.ndarray, 
                                  text_features: np.ndarray) -> np.ndarray:
        """使用学习的网络计算相似度"""
        device = next(self.similarity_net.parameters()).device
        
        image_tensor = torch.from_numpy(image_features).float().to(device)
        text_tensor = torch.from_numpy(text_features).float().to(device)
        
        n_images, n_texts = len(image_features), len(text_features)
        similarity_matrix = np.zeros((n_images, n_texts))
        
        with torch.no_grad():
            for i in range(n_images):
                for j in range(n_texts):
                    # 拼接特征
                    combined_features = torch.cat([image_tensor[i], text_tensor[j]], dim=0)
                    similarity = self.similarity_net(combined_features)
                    similarity_matrix[i, j] = similarity.cpu().item()
        
        return similarity_matrix
    
    def find_best_matches(self, image_features: np.ndarray, 
                         text_features: np.ndarray, 
                         top_k: int = 1) -> List[List[Tuple[int, float]]]:
        """
        为每个图像找到最佳匹配的文本
        
        Args:
            image_features: 图像特征矩阵 (N, D)
            text_features: 文本特征矩阵 (M, D)
            top_k: 返回前k个最佳匹配
            
        Returns:
            每个图像的最佳匹配列表，格式为 [(text_idx, similarity_score), ...]
        """
        similarity_matrix = self.compute_similarity_matrix(image_features, text_features)
        
        best_matches = []
        for i in range(len(image_features)):
            # 获取第i个图像与所有文本的相似度
            similarities = similarity_matrix[i]
            
            # 找到top_k个最高相似度的索引
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # 构建匹配结果
            matches = [(int(idx), float(similarities[idx])) for idx in top_indices]
            best_matches.append(matches)
        
        return best_matches
    
    def find_mutual_best_matches(self, image_features: np.ndarray, 
                               text_features: np.ndarray, 
                               threshold: float = 0.5) -> List[Tuple[int, int, float]]:
        """
        找到互相最佳匹配的图像-文本对
        
        Args:
            image_features: 图像特征矩阵 (N, D)
            text_features: 文本特征矩阵 (M, D)
            threshold: 相似度阈值
            
        Returns:
            互相最佳匹配的对，格式为 [(image_idx, text_idx, similarity), ...]
        """
        similarity_matrix = self.compute_similarity_matrix(image_features, text_features)
        
        mutual_matches = []
        
        # 找到每个图像的最佳文本匹配
        image_to_text = np.argmax(similarity_matrix, axis=1)
        
        # 找到每个文本的最佳图像匹配
        text_to_image = np.argmax(similarity_matrix, axis=0)
        
        # 检查互相匹配
        for img_idx in range(len(image_features)):
            best_text_idx = image_to_text[img_idx]
            similarity_score = similarity_matrix[img_idx, best_text_idx]
            
            # 检查是否互相最佳匹配且超过阈值
            if (text_to_image[best_text_idx] == img_idx and 
                similarity_score >= threshold):
                mutual_matches.append((img_idx, best_text_idx, float(similarity_score)))
        
        return mutual_matches
    
    def compute_retrieval_metrics(self, image_features: np.ndarray, 
                                text_features: np.ndarray, 
                                ground_truth: List[int]) -> dict:
        """
        计算检索评估指标
        
        Args:
            image_features: 图像特征矩阵 (N, D)
            text_features: 文本特征矩阵 (M, D)
            ground_truth: 真实匹配标签，长度为N
            
        Returns:
            评估指标字典
        """
        similarity_matrix = self.compute_similarity_matrix(image_features, text_features)
        
        # 计算各种指标
        metrics = {}
        
        # Top-1 准确率
        top1_predictions = np.argmax(similarity_matrix, axis=1)
        top1_accuracy = np.mean(top1_predictions == ground_truth)
        metrics['top1_accuracy'] = float(top1_accuracy)
        
        # Top-5 准确率
        top5_predictions = np.argsort(similarity_matrix, axis=1)[:, -5:]
        top5_accuracy = np.mean([gt in top5_pred for gt, top5_pred in zip(ground_truth, top5_predictions)])
        metrics['top5_accuracy'] = float(top5_accuracy)
        
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
        
        return metrics
    
    def save_similarity_net(self, save_path: str):
        """保存学习的相似度网络"""
        if self.similarity_method == 'learned' and hasattr(self, 'similarity_net'):
            torch.save({
                'model_state_dict': self.similarity_net.state_dict(),
                'similarity_method': self.similarity_method,
                'temperature': self.temperature
            }, save_path)
            logger.info(f"相似度网络已保存到: {save_path}")
    
    def load_similarity_net(self, load_path: str):
        """加载学习的相似度网络"""
        if self.similarity_method == 'learned':
            checkpoint = torch.load(load_path)
            self.similarity_net.load_state_dict(checkpoint['model_state_dict'])
            self.temperature = checkpoint.get('temperature', self.temperature)
            logger.info(f"相似度网络已从 {load_path} 加载")
    
    def get_config(self) -> dict:
        """获取配置信息"""
        return {
            'similarity_method': self.similarity_method,
            'temperature': self.temperature,
            'has_learned_net': hasattr(self, 'similarity_net')
        }


class ContrastiveLoss(nn.Module):
    """对比学习损失函数"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        计算对比学习损失
        
        Args:
            image_features: 图像特征 (batch_size, feature_dim)
            text_features: 文本特征 (batch_size, feature_dim)
            
        Returns:
            对比学习损失
        """
        batch_size = image_features.size(0)
        
        # 归一化特征
        image_features = nn.functional.normalize(image_features, p=2, dim=1)
        text_features = nn.functional.normalize(text_features, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(image_features, text_features.T) / self.temperature
        
        # 标签（对角线为正样本）
        labels = torch.arange(batch_size, device=image_features.device)
        
        # 计算双向损失
        loss_i2t = self.criterion(similarity_matrix, labels)
        loss_t2i = self.criterion(similarity_matrix.T, labels)
        
        return (loss_i2t + loss_t2i) / 2