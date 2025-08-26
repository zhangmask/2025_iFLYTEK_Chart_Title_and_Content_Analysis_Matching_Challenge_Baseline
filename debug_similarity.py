#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试相似度计算问题的脚本
"""

import torch
import numpy as np
from src.data.loader import DataLoader
from src.features.image_features import ImageFeatureExtractor
from src.features.text_features import TextFeatureExtractor
from src.features.similarity import SimilarityCalculator
from config import MODEL_CONFIG

def main():
    print("=== 调试相似度计算问题 ===")
    
    # 初始化组件
    data_loader = DataLoader()
    image_extractor = ImageFeatureExtractor()
    text_extractor = TextFeatureExtractor()
    similarity_calc = SimilarityCalculator()
    
    # 加载数据
    print("\n1. 加载数据...")
    train_data = data_loader.load_train_data()
    candidate_texts = train_data['Caption'].tolist()
    print(f"候选文本数量: {len(candidate_texts)}")
    
    # 测试数据
    dataset_path = "D:/Desktop/作品/2025/2025讯飞系列/图表标题与内容的解析匹配挑战赛/图表文件/dataset"
    test_data = [
        {'image_path': f'{dataset_path}/bedcf7d3-a91c-4c3b-b266-9c4a24efa24c.png', 'id': 'test1'},
        {'image_path': f'{dataset_path}/de788cb6-6929-4152-8795-84208728ec9c.jpg', 'id': 'test2'},
        {'image_path': f'{dataset_path}/05c5deb7-4d90-449a-bdfe-8ddae0b273b6.png', 'id': 'test3'}
    ]
    
    print(f"测试图像数量: {len(test_data)}")
    
    # 2. 提取文本特征
    print("\n2. 提取文本特征...")
    text_features = text_extractor.extract_features(candidate_texts)
    print(f"文本特征形状: {text_features.shape}")
    print(f"文本特征范围: [{text_features.min():.6f}, {text_features.max():.6f}]")
    print(f"文本特征均值: {text_features.mean():.6f}")
    print(f"文本特征标准差: {text_features.std():.6f}")
    
    # 检查文本特征的多样性
    print("\n3. 检查文本特征多样性...")
    # 计算前10个文本特征之间的相似度
    sample_text_features = torch.tensor(text_features[:10])
    text_similarity_matrix = torch.mm(sample_text_features, sample_text_features.T)
    print(f"前10个文本特征相似度矩阵:")
    print(text_similarity_matrix.numpy())
    
    # 检查是否所有文本特征都相同
    text_features_tensor = torch.tensor(text_features)
    unique_features = torch.unique(text_features_tensor, dim=0)
    print(f"\n唯一文本特征数量: {unique_features.shape[0]} / {text_features_tensor.shape[0]}")
    
    if unique_features.shape[0] == 1:
        print("⚠️ 警告：所有文本特征都相同！")
    elif unique_features.shape[0] < text_features_tensor.shape[0] * 0.1:
        print("⚠️ 警告：文本特征多样性很低！")
    else:
        print("✓ 文本特征具有良好的多样性")
    
    # 4. 提取图像特征
    print("\n4. 提取图像特征...")
    image_features_list = []
    for i, sample in enumerate(test_data):
        try:
            features = image_extractor.extract_single_feature(sample['image_path'])
            features_tensor = torch.tensor(features).unsqueeze(0)  # 添加batch维度
            image_features_list.append(features_tensor)
            print(f"图像 {i+1} 特征形状: {features_tensor.shape}")
            print(f"图像 {i+1} 特征范围: [{features_tensor.min():.6f}, {features_tensor.max():.6f}]")
        except Exception as e:
            print(f"图像 {i+1} 特征提取失败: {e}")
    
    if not image_features_list:
        print("❌ 没有成功提取任何图像特征")
        return
    
    # 5. 计算相似度
    print("\n5. 计算相似度...")
    for i, image_features in enumerate(image_features_list):
        print(f"\n--- 图像 {i+1} ---")
        
        # 计算相似度
        image_np = image_features.cpu().numpy()
        text_np = text_features
        similarities = similarity_calc.compute_similarity_matrix(image_np, text_np)
        similarities_tensor = torch.tensor(similarities).squeeze()
        print(f"相似度形状: {similarities_tensor.shape}")
        print(f"相似度范围: [{similarities_tensor.min():.6f}, {similarities_tensor.max():.6f}]")
        print(f"相似度均值: {similarities_tensor.mean():.6f}")
        print(f"相似度标准差: {similarities_tensor.std():.6f}")
        
        # 检查相似度是否都相同
        unique_similarities = torch.unique(similarities_tensor)
        print(f"唯一相似度值数量: {len(unique_similarities)}")
        if len(unique_similarities) == 1:
            print(f"警告：所有相似度都相同！值为: {unique_similarities[0]:.6f}")
        else:
            print(f"相似度值范围: {unique_similarities[:5]}...")
        
        # 找到最高相似度的索引
        best_match_idx = torch.argmax(similarities_tensor)
        best_similarity = similarities_tensor[best_match_idx]
        
        print(f"最佳匹配索引: {best_match_idx}, 相似度: {best_similarity:.6f}")
        print(f"匹配文本: {candidate_texts[best_match_idx][:100]}...")
        
        # 显示前5个最高相似度
        top5_indices = torch.argsort(similarities_tensor, descending=True)[:5]
        print("\n前5个最高相似度:")
        for i, idx in enumerate(top5_indices):
            print(f"  {i+1}. 索引{idx}: {similarities_tensor[idx]:.6f} - {candidate_texts[idx][:50]}...")
    
    # 6. 检查特征归一化
    print("\n6. 检查特征归一化...")
    
    # 检查图像特征是否已归一化
    for i, image_features in enumerate(image_features_list):
        norm = torch.norm(image_features, dim=1)
        print(f"图像 {i+1} 特征L2范数: {norm.item():.6f}")
    
    # 检查文本特征是否已归一化
    text_norms = torch.norm(text_features_tensor, dim=1)
    print(f"文本特征L2范数范围: [{text_norms.min():.6f}, {text_norms.max():.6f}]")
    print(f"文本特征L2范数均值: {text_norms.mean():.6f}")
    
    # 7. 手动计算相似度验证
    print("\n7. 手动验证相似度计算...")
    if image_features_list:
        image_features = image_features_list[0]
        
        # 手动计算点积
        manual_similarities = torch.mm(image_features, text_features_tensor.T).squeeze()
        print(f"手动计算相似度形状: {manual_similarities.shape}")
        print(f"手动计算相似度范围: [{manual_similarities.min():.6f}, {manual_similarities.max():.6f}]")
        
        # 与SimilarityCalculator的结果比较
        image_np = image_features.cpu().numpy()
        calc_similarities = similarity_calc.compute_similarity_matrix(image_np, text_features)
        calc_similarities_tensor = torch.tensor(calc_similarities).squeeze()
        diff = torch.abs(manual_similarities - calc_similarities_tensor)
        print(f"计算差异最大值: {diff.max():.6f}")
        
        if diff.max() < 1e-6:
            print("✓ 相似度计算一致")
        else:
            print("⚠️ 相似度计算存在差异")

if __name__ == "__main__":
    main()