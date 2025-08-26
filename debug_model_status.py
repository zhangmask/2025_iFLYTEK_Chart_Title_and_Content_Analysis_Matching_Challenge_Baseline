#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型状态调试脚本
检查图像和文本特征提取器的详细状态
"""

import sys
sys.path.append('.')

import torch
import numpy as np
from pathlib import Path

from src.features.image_features import ImageFeatureExtractor
from src.features.text_features import TextFeatureExtractor
from src.data.loader import DataLoader

def debug_model_status():
    print("=" * 50)
    print("模型状态调试")
    print("=" * 50)
    
    # 1. 检查设备
    print("\n1. 设备信息:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  当前设备: {device}")
    if torch.cuda.is_available():
        print(f"  GPU名称: {torch.cuda.get_device_name()}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 2. 初始化图像特征提取器
    print("\n2. 初始化图像特征提取器...")
    try:
        image_extractor = ImageFeatureExtractor()
        print(f"  模型信息: {image_extractor.get_model_info()}")
        
        # 检查模型参数
        total_params = sum(p.numel() for p in image_extractor.parameters())
        trainable_params = sum(p.numel() for p in image_extractor.parameters() if p.requires_grad)
        print(f"  总参数数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        # 检查backbone
        print(f"  Backbone类型: {type(image_extractor.backbone)}")
        print(f"  投影层: {image_extractor.feature_projection}")
        
        # 测试前向传播
        print("\n  测试图像特征提取...")
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            features = image_extractor.forward(dummy_input)
            print(f"  输出特征形状: {features.shape}")
            print(f"  输出特征范围: [{features.min().item():.6f}, {features.max().item():.6f}]")
            print(f"  输出特征均值: {features.mean().item():.6f}")
            print(f"  输出特征标准差: {features.std().item():.6f}")
            print(f"  是否包含NaN: {torch.isnan(features).any().item()}")
            print(f"  是否包含Inf: {torch.isinf(features).any().item()}")
            
    except Exception as e:
        print(f"  图像特征提取器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 初始化文本特征提取器
    print("\n3. 初始化文本特征提取器...")
    try:
        text_extractor = TextFeatureExtractor()
        print(f"  模型信息: {text_extractor.get_model_info()}")
        
        # 检查模型组件
        print(f"  Transformer模型: {type(text_extractor.model)}")
        print(f"  投影层: {text_extractor.projection}")
        print(f"  分词器: {type(text_extractor.tokenizer)}")
        
        # 测试前向传播
        print("\n  测试文本特征提取...")
        test_texts = ["This is a test sentence.", "Another test sentence."]
        with torch.no_grad():
            features = text_extractor.forward(test_texts)
            print(f"  输出特征形状: {features.shape}")
            print(f"  输出特征范围: [{features.min().item():.6f}, {features.max().item():.6f}]")
            print(f"  输出特征均值: {features.mean().item():.6f}")
            print(f"  输出特征标准差: {features.std().item():.6f}")
            print(f"  是否包含NaN: {torch.isnan(features).any().item()}")
            print(f"  是否包含Inf: {torch.isinf(features).any().item()}")
            
    except Exception as e:
        print(f"  文本特征提取器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 测试实际数据
    print("\n4. 测试实际数据...")
    try:
        # 加载数据
        data_loader = DataLoader()
        data_loader.load_test_data()
        test_data = data_loader.test_data.head(2)
        
        print(f"  测试数据形状: {test_data.shape}")
        print(f"  测试图像路径: {test_data['Source'].tolist()}")
        
        # 测试图像特征提取
        for idx, row in test_data.iterrows():
            image_path = row['Source']
            print(f"\n  测试图像 {idx}: {image_path}")
            
            # 检查文件是否存在
            if not Path(image_path).exists():
                print(f"    警告: 文件不存在!")
                continue
                
            try:
                features = image_extractor.extract_single_feature(image_path)
                print(f"    特征形状: {features.shape}")
                print(f"    特征范围: [{features.min():.6f}, {features.max():.6f}]")
                print(f"    特征均值: {features.mean():.6f}")
                print(f"    特征标准差: {features.std():.6f}")
                print(f"    是否全零: {np.allclose(features, 0)}")
            except Exception as e:
                print(f"    特征提取失败: {e}")
                
        # 测试文本特征提取
        data_loader.load_train_data()
        candidate_texts = data_loader.get_all_captions()[:5]
        print(f"\n  候选文本数量: {len(candidate_texts)}")
        print(f"  前3个文本: {candidate_texts[:3]}")
        
        try:
            text_features = text_extractor.extract_features(candidate_texts)
            print(f"  文本特征形状: {text_features.shape}")
            print(f"  文本特征范围: [{text_features.min():.6f}, {text_features.max():.6f}]")
            print(f"  文本特征均值: {text_features.mean():.6f}")
            print(f"  文本特征标准差: {text_features.std():.6f}")
            print(f"  是否全零: {np.allclose(text_features, 0)}")
        except Exception as e:
            print(f"  文本特征提取失败: {e}")
            
    except Exception as e:
        print(f"  实际数据测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=" * 50)
    print("调试完成")
    print("=" * 50)

if __name__ == "__main__":
    debug_model_status()