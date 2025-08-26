#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接预测脚本 - 跳过训练，直接生成预测结果
"""

import os
import sys
import pandas as pd
from pathlib import Path
from loguru import logger
import torch

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG
from src.data.loader import DataLoader
from src.features.image_features import ImageFeatureExtractor
from src.features.text_features import TextFeatureExtractor
from src.features.similarity import SimilarityCalculator
from src.prediction.predictor import ChartPredictor

def main():
    """主函数"""
    logger.info("=== 开始直接预测 ===")
    
    # 1. 加载数据
    logger.info("1. 加载数据...")
    data_loader = DataLoader()
    
    # 加载训练数据作为候选文本
    train_df = data_loader.load_train_data()
    candidate_texts = train_df['Caption'].tolist()
    logger.info(f"候选文本数量: {len(candidate_texts)}")
    
    # 加载测试数据
    test_df = data_loader.load_test_data()
    logger.info(f"测试数据数量: {len(test_df)}")
    
    # 2. 初始化模型
    logger.info("2. 初始化模型...")
    
    # 图像特征提取器
    image_extractor = ImageFeatureExtractor(
        model_name=MODEL_CONFIG['image_model']['name']
    )
    
    # 文本特征提取器
    text_extractor = TextFeatureExtractor(
        model_name=MODEL_CONFIG['text_model']['name']
    )
    
    # 相似度计算器
    similarity_calculator = SimilarityCalculator(
        similarity_method=MODEL_CONFIG['cross_modal']['similarity_metric'],
        temperature=MODEL_CONFIG['cross_modal']['temperature']
    )
    
    # 预测器
    predictor = ChartPredictor(
        image_extractor=image_extractor,
        text_extractor=text_extractor,
        similarity_calculator=similarity_calculator
    )
    
    logger.info("模型初始化完成")
    
    # 3. 生成预测
    logger.info("3. 开始预测...")
    
    # 提取候选文本特征
    logger.info("提取候选文本特征...")
    candidate_features = text_extractor.extract_features(candidate_texts)
    
    # 预测测试数据
    logger.info("预测测试数据...")
    test_image_paths = test_df['Source'].tolist()
    
    # 批量预测
    batch_size = 32
    all_predictions = []
    
    for i in range(0, len(test_image_paths), batch_size):
        batch_paths = test_image_paths[i:i+batch_size]
        logger.info(f"处理批次 {i//batch_size + 1}/{(len(test_image_paths) + batch_size - 1)//batch_size}")
        
        try:
            # 提取图像特征
            image_features = image_extractor.extract_features_from_paths(batch_paths)
            
            # 计算相似度并预测
            # 将torch tensor转换为numpy array
            if isinstance(image_features, torch.Tensor):
                image_features_np = image_features.cpu().numpy()
            else:
                image_features_np = image_features
                
            if isinstance(candidate_features, torch.Tensor):
                candidate_features_np = candidate_features.cpu().numpy()
            else:
                candidate_features_np = candidate_features
            
            similarities = similarity_calculator.compute_similarity_matrix(
                image_features_np, candidate_features_np
            )
            
            # 获取最佳匹配
            similarities_tensor = torch.from_numpy(similarities)
            best_matches = torch.argmax(similarities_tensor, dim=1)
            
            # 添加到结果列表
            for j, match_idx in enumerate(best_matches):
                prediction = {
                    'Source': batch_paths[j],
                    'Caption': candidate_texts[match_idx.item()]
                }
                all_predictions.append(prediction)
                
        except Exception as e:
            logger.error(f"批次 {i//batch_size + 1} 预测失败: {e}")
            # 为失败的批次添加默认预测
            for path in batch_paths:
                prediction = {
                    'Source': path,
                    'Caption': candidate_texts[0]  # 使用第一个候选文本作为默认
                }
                all_predictions.append(prediction)
    
    # 4. 保存结果
    logger.info("4. 保存预测结果...")
    
    # 创建提交文件
    submission_df = pd.DataFrame(all_predictions)
    
    # 确保输出目录存在
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    # 保存提交文件
    submission_path = output_dir / 'submission.csv'
    submission_df.to_csv(submission_path, index=False, encoding='utf-8')
    
    logger.info(f"预测完成！结果已保存到: {submission_path}")
    logger.info(f"预测数量: {len(submission_df)}")
    
    # 显示前几行结果
    logger.info("前5行预测结果:")
    print(submission_df.head())
    
    # 检查预测多样性
    unique_captions = submission_df['Caption'].nunique()
    logger.info(f"唯一预测数量: {unique_captions}/{len(submission_df)}")
    
    if unique_captions < len(submission_df) * 0.1:
        logger.warning("预测多样性较低，可能需要调整模型参数")
    else:
        logger.info("预测具有良好的多样性")

if __name__ == '__main__':
    main()