#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 简化版本，避免进度条导致的卡顿问题
"""

import os
import sys
import torch
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.loader import DataLoader
from src.data.dataset import create_dataloaders
from src.features.image_features import ImageFeatureExtractor
from src.features.text_features import TextFeatureExtractor
from src.features.similarity import SimilarityCalculator
from src.prediction.predictor import ChartPredictor
from src.utils.config_manager import ConfigManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    try:
        logger.info("开始快速测试...")
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {device}")
        
        # 创建配置管理器
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # 修改配置以加快测试
        config.training.num_epochs = 2  # 只训练2个epoch
        config.training.batch_size = 16  # 减小批次大小
        
        logger.info("加载数据...")
        
        # 创建数据加载器
        data_loader = DataLoader()
        
        # 加载数据
        train_data = data_loader.load_train_data()
        test_data = data_loader.load_test_data()
        
        # 划分训练和验证集
        train_split, val_split = data_loader.split_train_validation(validation_ratio=0.2)
        
        # 创建数据加载器
        dataloaders = create_dataloaders(
            train_data=train_split,
            val_data=val_split,
            test_data=test_data,
            data_loader_instance=data_loader
        )
        
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        test_loader = dataloaders['test']
        
        logger.info(f"数据加载完成: 训练集 {len(train_loader.dataset)} 样本")
        
        # 创建模型
        logger.info("创建模型...")
        
        image_extractor = ImageFeatureExtractor(
            model_name=config.image.model_name,
            feature_dim=config.image.feature_dim
        )
        
        text_extractor = TextFeatureExtractor(
            model_name=config.text.model_name,
            feature_dim=config.text.feature_dim
        )
        
        similarity_calculator = SimilarityCalculator(
            similarity_method=config.similarity.method,
            temperature=config.similarity.temperature
        )
        
        logger.info("模型创建完成")
        
        # 简化的训练过程（不使用tqdm）
        logger.info("开始简化训练...")
        
        image_extractor.train()
        text_extractor.train()
        
        # 设置优化器
        params = list(image_extractor.parameters())
        if hasattr(text_extractor, 'projection') and text_extractor.projection is not None:
            params.extend(text_extractor.projection.parameters())
        
        optimizer = torch.optim.AdamW(params, lr=config.training.learning_rate)
        
        # 简单的训练循环
        for epoch in range(config.training.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{config.training.num_epochs}")
            
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    # 获取批次数据
                    images = batch['image'].to(device)
                    texts = batch['captions']
                    
                    # 前向传播
                    optimizer.zero_grad()
                    
                    # 提取特征
                    image_features = image_extractor(images)
                    text_features = text_extractor(texts)
                    
                    # 计算对比损失
                    similarity_matrix = similarity_calculator.compute_similarity_matrix(
                        image_features, text_features
                    )
                    
                    # 简单的对比损失
                    batch_size = similarity_matrix.size(0)
                    labels = torch.arange(batch_size).to(device)
                    
                    loss_i2t = torch.nn.functional.cross_entropy(similarity_matrix, labels)
                    loss_t2i = torch.nn.functional.cross_entropy(similarity_matrix.t(), labels)
                    loss = (loss_i2t + loss_t2i) / 2
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if batch_idx % 10 == 0:
                        logger.info(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
                    
                except Exception as e:
                    logger.error(f"训练批次 {batch_idx} 失败: {e}")
                    continue
            
            avg_loss = total_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch + 1} 完成, 平均损失: {avg_loss:.4f}")
        
        logger.info("训练完成，开始预测...")
        
        # 创建预测器
        predictor = ChartPredictor(
            image_extractor=image_extractor,
            text_extractor=text_extractor,
            similarity_calculator=similarity_calculator,
            device=device
        )
        
        # 生成预测结果
        logger.info("生成预测结果...")
        predictions_df = predictor.predict_on_dataset(data_loader, split='test')
        
        # 保存结果
        output_dir = Path('outputs/submissions')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        submission_path = output_dir / 'quick_submission.csv'
        predictions_df.to_csv(submission_path, index=False)
        
        logger.info(f"预测完成！结果已保存到: {submission_path}")
        logger.info(f"生成了 {len(predictions_df)} 个预测结果")
        
        # 显示前几个预测结果
        logger.info("前5个预测结果:")
        for i, row in predictions_df.head().iterrows():
            logger.info(f"  {i}: {row['predicted_caption'][:50]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"快速测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("快速测试成功完成！")
    else:
        logger.error("快速测试失败！")
        sys.exit(1)