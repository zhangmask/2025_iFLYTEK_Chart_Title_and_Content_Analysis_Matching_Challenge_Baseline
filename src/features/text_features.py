import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Union, Optional, Dict
from pathlib import Path
from loguru import logger

from config import MODEL_CONFIG, TRAINING_CONFIG
from src.data.preprocessor import TextPreprocessor


class TextFeatureExtractor(nn.Module):
    """文本特征提取器，基于BERT模型"""
    
    def __init__(self, model_name: str = None, feature_dim: int = None):
        super().__init__()
        
        # 从配置获取参数
        if model_name is None:
            model_name = MODEL_CONFIG['text_model']['name']
        if feature_dim is None:
            feature_dim = MODEL_CONFIG['text_model']['feature_dim']
        
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.device = torch.device(TRAINING_CONFIG['device'])
        
        # 初始化模型
        self._init_transformer()
        
        # 文本预处理器
        self.preprocessor = TextPreprocessor()
        
        logger.info(f"文本特征提取器初始化完成: {model_name}, 特征维度: {feature_dim}")
    

    
    def _init_transformer(self):
        """初始化普通Transformer模型"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            # 获取模型输出维度
            model_dim = self.model.config.hidden_size
            
            # 添加投影层
            self.projection = nn.Sequential(
                nn.Linear(model_dim, self.feature_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.LayerNorm(self.feature_dim)
            ).to(self.device)
            
        except Exception as e:
            logger.error(f"Transformer初始化失败: {e}")
            raise
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """前向传播"""
        return self._forward_transformer(texts)
    

    
    def _forward_transformer(self, texts: List[str]) -> torch.Tensor:
        """普通Transformer前向传播"""
        # 预处理文本
        processed_texts = [self.preprocessor.preprocess_text(text) for text in texts]
        
        # 分词
        encoded = self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # 移动到设备
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(**encoded)
            
        # 使用[CLS] token的表示或平均池化
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            # 平均池化
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # 投影到目标维度
        embeddings = self.projection(embeddings)
        
        # L2归一化
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def extract_features(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量提取文本特征"""
        self.eval()
        all_features = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                try:
                    # 提取特征
                    features = self.forward(batch_texts)
                    all_features.append(features.cpu().numpy())
                    
                except Exception as e:
                    logger.warning(f"批次 {i//batch_size + 1} 处理失败: {e}")
                    # 添加零特征作为fallback
                    fallback_features = np.zeros((len(batch_texts), self.feature_dim))
                    all_features.append(fallback_features)
        
        return np.vstack(all_features)
    
    def extract_single_feature(self, text: str) -> np.ndarray:
        """提取单个文本的特征"""
        self.eval()
        
        with torch.no_grad():
            try:
                # 提取特征
                features = self.forward([text])
                return features.cpu().numpy().squeeze()
                
            except Exception as e:
                logger.error(f"单个文本特征提取失败: {e}")
                return np.zeros(self.feature_dim)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        feature1 = self.extract_single_feature(text1)
        feature2 = self.extract_single_feature(text2)
        
        # 余弦相似度
        similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
        return float(similarity)
    
    def save_model(self, save_path: Union[str, Path]):
        """保存模型"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存完整模型
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'projection_state_dict': self.projection.state_dict(),
            'model_name': self.model_name,
            'feature_dim': self.feature_dim
        }, save_path / 'model.pt')
        
        logger.info(f"文本特征提取器已保存到: {save_path}")
    
    def load_model(self, load_path: Union[str, Path]):
        """加载模型"""
        load_path = Path(load_path)
        
        # 加载模型
        checkpoint = torch.load(load_path / 'model.pt', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.projection.load_state_dict(checkpoint['projection_state_dict'])
        
        logger.info(f"文本特征提取器已从 {load_path} 加载")
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        total_params += sum(p.numel() for p in self.projection.parameters())
        
        return {
            'model_name': self.model_name,
            'feature_dim': self.feature_dim,
            'total_parameters': total_params,
            'device': str(self.device)
        }