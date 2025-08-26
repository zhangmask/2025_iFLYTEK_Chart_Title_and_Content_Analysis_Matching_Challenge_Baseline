#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试预测逻辑问题的脚本
专门检查为什么所有预测结果都相同
"""

import torch
import numpy as np
import pandas as pd
from src.data.loader import DataLoader
from src.features.image_features import ImageFeatureExtractor
from src.features.text_features import TextFeatureExtractor
from src.features.similarity import SimilarityCalculator
from config import MODEL_CONFIG

def main():
    print("=== 调试预测逻辑问题 ===\n")
    
    # 初始化组件
    data_loader = DataLoader()
    image_extractor = ImageFeatureExtractor()
    text_extractor = TextFeatureExtractor()
    similarity_calc = SimilarityCalculator()
    
    # 1. 加载训练数据（候选文本）
    print("1. 加载训练数据...")
    train_data = data_loader.load_train_data()
    candidate_texts = train_data['Caption'].tolist()
    print(f"候选文本数量: {len(candidate_texts)}")
    print(f"前3个候选文本:")
    for i in range(min(3, len(candidate_texts))):
        print(f"  {i}: {candidate_texts[i][:100]}...")
    
    # 2. 加载少量测试数据
    print("\n2. 加载测试数据...")
    test_data = data_loader.load_test_data()
    print(f"测试数据总数: {len(test_data)}")
    
    # 只处理前5个测试样本
    test_samples = test_data.head(5)
    print(f"处理前5个测试样本:")
    for i, row in test_samples.iterrows():
        print(f"  {i}: {row['Source']}")
    
    # 3. 提取候选文本特征
    print("\n3. 提取候选文本特征...")
    text_features = text_extractor.extract_features(candidate_texts)
    print(f"文本特征形状: {text_features.shape}")
    print(f"文本特征范围: [{text_features.min():.6f}, {text_features.max():.6f}]")
    
    # 4. 逐个处理测试图像
    print("\n4. 逐个处理测试图像...")
    predictions = []
    
    for idx, row in test_samples.iterrows():
        image_path = row['Source']
        # 转换相对路径为绝对路径
        if image_path.startswith('./'):
            relative_path = image_path[2:]  # 移除 './'
        else:
            relative_path = image_path
        full_image_path = f"d:\\Desktop\\作品\\2025\\2025讯飞系列\\图表标题与内容的解析匹配挑战赛\\图表文件\\{relative_path}"
        print(f"\n--- 处理图像 {idx}: {image_path} ---")
        print(f"完整路径: {full_image_path}")
        
        try:
            # 4.1 提取图像特征
            image_features = image_extractor.extract_single_feature(full_image_path)
            image_features_tensor = torch.tensor(image_features).unsqueeze(0)
            print(f"图像特征形状: {image_features_tensor.shape}")
            print(f"图像特征范围: [{image_features_tensor.min():.6f}, {image_features_tensor.max():.6f}]")
            
            # 4.2 计算相似度
            image_np = image_features_tensor.cpu().numpy()
            similarities = similarity_calc.compute_similarity_matrix(image_np, text_features)
            similarities_tensor = torch.tensor(similarities).squeeze()
            
            print(f"相似度形状: {similarities_tensor.shape}")
            print(f"相似度范围: [{similarities_tensor.min():.6f}, {similarities_tensor.max():.6f}]")
            print(f"相似度均值: {similarities_tensor.mean():.6f}")
            print(f"相似度标准差: {similarities_tensor.std():.6f}")
            
            # 4.3 找到最佳匹配
            best_match_idx = torch.argmax(similarities_tensor).item()
            best_similarity = similarities_tensor[best_match_idx].item()
            best_text = candidate_texts[best_match_idx]
            
            print(f"最佳匹配索引: {best_match_idx}")
            print(f"最佳相似度: {best_similarity:.6f}")
            print(f"匹配文本: {best_text[:100]}...")
            
            # 4.4 显示前5个最高相似度
            top5_indices = torch.argsort(similarities_tensor, descending=True)[:5]
            print("前5个最高相似度:")
            for i, text_idx in enumerate(top5_indices):
                sim_score = similarities_tensor[text_idx].item()
                text = candidate_texts[text_idx]
                print(f"  {i+1}. 索引{text_idx}: {sim_score:.6f} - {text[:50]}...")
            
            # 4.5 检查相似度分布
            unique_similarities = torch.unique(similarities_tensor)
            print(f"唯一相似度值数量: {len(unique_similarities)}")
            if len(unique_similarities) == 1:
                print(f"⚠️ 警告：所有相似度都相同！值为: {unique_similarities[0]:.6f}")
            elif len(unique_similarities) < 10:
                print(f"⚠️ 警告：相似度值种类很少: {unique_similarities.tolist()}")
            
            # 保存预测结果
            predictions.append({
                'Source': row['Source'],  # 保持原始相对路径
                'best_match_idx': best_match_idx,
                'best_similarity': best_similarity,
                'Caption': best_text
            })
            
        except Exception as e:
            print(f"❌ 处理图像失败: {e}")
            predictions.append({
                'Source': row['Source'],  # 保持原始相对路径
                'best_match_idx': -1,
                'best_similarity': 0.0,
                'Caption': 'ERROR'
            })
    
    # 5. 分析预测结果
    print("\n5. 分析预测结果...")
    print("预测结果汇总:")
    for i, pred in enumerate(predictions):
        print(f"图像 {i+1}: 索引{pred['best_match_idx']}, 相似度{pred['best_similarity']:.6f}")
        print(f"  文本: {pred['Caption'][:80]}...")
    
    # 检查是否所有预测都相同
    unique_predictions = set(pred['best_match_idx'] for pred in predictions if pred['best_match_idx'] != -1)
    print(f"\n唯一预测索引数量: {len(unique_predictions)}")
    
    if len(unique_predictions) == 1:
        print("⚠️ 发现问题：所有图像都预测为同一个文本！")
        common_idx = list(unique_predictions)[0]
        print(f"共同预测的文本索引: {common_idx}")
        print(f"共同预测的文本: {candidate_texts[common_idx]}")
    elif len(unique_predictions) < len(predictions) / 2:
        print(f"⚠️ 警告：预测多样性较低，{len(predictions)}个图像只有{len(unique_predictions)}种不同预测")
    else:
        print("✓ 预测具有良好的多样性")
    
    # 6. 检查SimilarityCalculator的配置
    print("\n6. 检查SimilarityCalculator配置...")
    print(f"相似度方法: {similarity_calc.similarity_method}")
    print(f"温度参数: {similarity_calc.temperature}")
    
    # 7. 生成小规模提交文件进行验证
    print("\n7. 生成小规模提交文件...")
    submission_df = pd.DataFrame(predictions)[['Source', 'Caption']]
    submission_df.to_csv('debug_submission.csv', index=False)
    print("已生成 debug_submission.csv")
    print("前几行内容:")
    print(submission_df.head())

if __name__ == "__main__":
    main()