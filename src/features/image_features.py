import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from typing import List, Union, Optional
from pathlib import Path
from loguru import logger

from config import MODEL_CONFIG, TRAINING_CONFIG
from src.data.preprocessor import ImagePreprocessor


class ImageFeatureExtractor(nn.Module):
    """图像特征提取器，基于ResNet预训练模型"""
    
    def __init__(self, model_name: str = None, pretrained: bool = True, feature_dim: int = None):
        super().__init__()
        
        # 从配置获取参数
        if model_name is None:
            model_name = MODEL_CONFIG['image_model']['name']
        if feature_dim is None:
            feature_dim = MODEL_CONFIG['image_model']['feature_dim']
        
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.device = torch.device(TRAINING_CONFIG['device'])
        
        # 加载预训练模型
        self.backbone = self._load_backbone(model_name, pretrained)
        
        # 获取backbone输出维度
        backbone_dim = self._get_backbone_dim()
        
        # 添加特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # 图像预处理器
        self.preprocessor = ImagePreprocessor()
        
        self.to(self.device)
        logger.info(f"图像特征提取器初始化完成: {model_name}, 特征维度: {feature_dim}")
    
    def _load_backbone(self, model_name: str, pretrained: bool) -> nn.Module:
        """加载backbone模型"""
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            # 移除最后的分类层
            model = nn.Sequential(*list(model.children())[:-1])
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
            model = nn.Sequential(*list(model.children())[:-1])
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=pretrained)
            model = nn.Sequential(*list(model.children())[:-1])
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier = nn.Identity()
        elif model_name == 'vit_b_16':
            model = models.vit_b_16(pretrained=pretrained)
            model.heads = nn.Identity()
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        return model
    
    def _get_backbone_dim(self) -> int:
        """获取backbone输出维度"""
        if 'resnet' in self.model_name:
            if 'resnet50' in self.model_name or 'resnet101' in self.model_name or 'resnet152' in self.model_name:
                return 2048
        elif 'efficientnet_b0' in self.model_name:
            return 1280
        elif 'vit_b_16' in self.model_name:
            return 768
        else:
            # 通过前向传播获取维度
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                output = self.backbone(dummy_input)
                return output.view(output.size(0), -1).size(1)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 提取backbone特征
        features = self.backbone(images)
        
        # 展平特征
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        
        # 特征投影
        features = self.feature_projection(features)
        
        # L2归一化
        features = nn.functional.normalize(features, p=2, dim=1)
        
        return features
    
    def extract_features_from_paths(self, image_paths: List[Union[str, Path]], 
                                  batch_size: int = 32) -> np.ndarray:
        """从图像路径批量提取特征"""
        self.eval()
        all_features = []
        
        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                
                try:
                    # 使用安全的批量预处理图像
                    batch_images = self.preprocessor.batch_preprocess_safe(batch_paths)
                    batch_images = batch_images.to(self.device)
                    
                    # 提取特征
                    features = self.forward(batch_images)
                    all_features.append(features.cpu().numpy())
                    
                except Exception as e:
                    logger.warning(f"批次 {i//batch_size + 1} 处理失败: {e}")
                    # 添加零特征作为fallback
                    fallback_features = np.zeros((len(batch_paths), self.feature_dim))
                    all_features.append(fallback_features)
        
        return np.vstack(all_features)
    
    def extract_single_feature(self, image_path: Union[str, Path]) -> np.ndarray:
        """提取单个图像的特征"""
        self.eval()
        
        with torch.no_grad():
            try:
                # 预处理图像
                image_tensor = self.preprocessor.preprocess_from_path(image_path)
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                
                # 提取特征
                features = self.forward(image_tensor)
                return features.cpu().numpy().squeeze()
                
            except Exception as e:
                logger.error(f"单个图像特征提取失败 {image_path}: {e}")
                return np.zeros(self.feature_dim)
    
    def save_model(self, save_path: Union[str, Path]):
        """保存模型"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'feature_dim': self.feature_dim,
            'config': MODEL_CONFIG['image_model']
        }, save_path)
        
        logger.info(f"图像特征提取器已保存到: {save_path}")
    
    def load_model(self, load_path: Union[str, Path]):
        """加载模型"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"图像特征提取器已从 {load_path} 加载")
    
    @classmethod
    def from_pretrained(cls, load_path: Union[str, Path]):
        """从预训练模型加载"""
        checkpoint = torch.load(load_path)
        
        model = cls(
            model_name=checkpoint['model_name'],
            feature_dim=checkpoint['feature_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'feature_dim': self.feature_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device)
        }