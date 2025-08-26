#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试特征提取和相似度计算
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import torch
import sys

# 添加项目根目录到Python路径
sys.path.append('.')

from src.data.loader import DataLoader
from src.features.image_features import ImageFeatureExtractor
from src.features.text_features import TextFeatureExtractor
from src.features.similarity import SimilarityCalculator

def debug_features():
    """调试特征提取过程"""
    logger.info("开始调试特征提取过程")
    
    # 初始化组件
    data_loader = DataLoader()
    image_extractor = ImageFeatureExtractor()
    text_extractor = TextFeatureExtractor()
    similarity_calculator = SimilarityCalculator()
    
    # 获取测试数据
    test_df = data_loader.test_df
    candidate_captions = data_loader.get_candidate_captions()
    
    logger.info(f"测试图像数量: {len(test_df)}")
    logger.info(f"候选标题数量: {len(candidate_captions)}")
    
    # 取前10个图像进行调试
    debug_images = test_df['Source'].head(10).tolist()
    debug_image_paths = [str(Path('dataset') / Path(img).name) for img in debug_images]
    
    logger.info(f"调试图像路径: {debug_image_paths[:3]}...")  # 只显示前3个
    
    # 提取图像特征
    logger.info("提取图像特征...")
    image_features = image_extractor.extract_features_from_paths(debug_image_paths, batch_size=5)
    logger.info(f"图像特征形状: {image_features.shape}")
    logger.info(f"图像特征统计: mean={np.mean(image_features):.4f}, std={np.std(image_features):.4f}")
    
    # 检查是否有相同的特征向量
    unique_features = np.unique(image_features, axis=0)
    logger.info(f"唯一特征向量数量: {len(unique_features)} / {len(image_features)}")
    
    if len(unique_features) < len(image_features):
        logger.warning("发现重复的图像特征向量！")
        # 找出重复的特征
        for i in range(len(image_features)):
            for j in range(i+1, len(image_features)):
                if np.allclose(image_features[i], image_features[j], atol=1e-6):
                    logger.warning(f"图像 {i} 和 {j} 的特征向量相同")
                    logger.warning(f"路径: {debug_image_paths[i]} vs {debug_image_paths[j]}")
    
    # 提取文本特征（取前50个候选标题）
    logger.info("提取文本特征...")
    debug_captions = candidate_captions[:50]
    text_features = text_extractor.extract_features(debug_captions, batch_size=10)
    logger.info(f"文本特征形状: {text_features.shape}")
    logger.info(f"文本特征统计: mean={np.mean(text_features):.4f}, std={np.std(text_features):.4f}")
    
    # 计算相似度矩阵
    logger.info("计算相似度矩阵...")
    similarity_matrix = similarity_calculator.compute_similarity_matrix(image_features, text_features)
    logger.info(f"相似度矩阵形状: {similarity_matrix.shape}")
    logger.info(f"相似度统计: mean={np.mean(similarity_matrix):.4f}, std={np.std(similarity_matrix):.4f}")
    logger.info(f"相似度范围: [{np.min(similarity_matrix):.4f}, {np.max(similarity_matrix):.4f}]")
    
    # 分析预测结果
    predictions = np.argmax(similarity_matrix, axis=1)
    logger.info(f"预测结果: {predictions}")
    
    # 统计预测分布
    unique_predictions, counts = np.unique(predictions, return_counts=True)
    logger.info(f"唯一预测数量: {len(unique_predictions)} / {len(predictions)}")
    
    for pred, count in zip(unique_predictions, counts):
        if count > 1:
            logger.warning(f"标题索引 {pred} 被预测了 {count} 次: '{debug_captions[pred][:50]}...'")
    
    # 显示每个图像的top-3预测
    logger.info("\n每个图像的top-3预测:")
    for i in range(len(image_features)):
        similarities = similarity_matrix[i]
        top3_indices = np.argsort(similarities)[::-1][:3]
        top3_scores = similarities[top3_indices]
        
        logger.info(f"\n图像 {i}: {debug_image_paths[i]}")
        for j, (idx, score) in enumerate(zip(top3_indices, top3_scores)):
            logger.info(f"  Top-{j+1}: 索引{idx}, 分数{score:.4f}, '{debug_captions[idx][:50]}...'")
    
    logger.info("调试完成")

if __name__ == "__main__":
    debug_features()