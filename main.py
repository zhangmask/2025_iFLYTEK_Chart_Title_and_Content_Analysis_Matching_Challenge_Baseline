#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图表标题与内容解析匹配挑战赛 - 主程序入口

这是一个完整的跨模态图表-文本匹配解决方案，包括：
- 数据加载和预处理
- 图像特征提取（ResNet等预训练模型）
- 文本特征提取（BERT/Sentence-BERT）
- 跨模态相似度计算和匹配
- 模型训练和评估
- 预测结果生成和导出

作者: SOLO Coding
日期: 2025
"""

import argparse
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from src.utils.logger import setup_logger, get_logger
from src.utils.config_manager import ConfigManager
from src.data import ChartDataLoader, create_dataloaders
from src.features import ImageFeatureExtractor, TextFeatureExtractor, SimilarityCalculator
from src.training import CrossModalTrainer, ModelEvaluator
from src.prediction import ChartPredictor, SubmissionGenerator, InferenceEngine
from src.utils import CheckpointManager, MetricsCalculator, Visualizer


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="图表标题与内容解析匹配挑战赛解决方案",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基本参数
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'predict', 'all'],
                       default='all', help='运行模式')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='输出目录路径')
    
    # 模型参数
    parser.add_argument('--image_model', type=str, default='resnet50',
                       help='图像特征提取模型')
    parser.add_argument('--text_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                       help='文本特征提取模型')
    parser.add_argument('--similarity_method', type=str, default='cosine',
                       choices=['cosine', 'euclidean', 'dot', 'learned'],
                       help='相似度计算方法')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='验证集比例')
    
    # 预测参数
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型检查点路径')
    parser.add_argument('--ensemble', action='store_true',
                       help='是否使用集成预测')
    parser.add_argument('--tta', action='store_true',
                       help='是否使用测试时增强')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备 (cpu/cuda/auto)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出')
    
    return parser.parse_args()


def setup_environment(args):
    """设置运行环境"""
    # 设置随机种子
    import random
    import numpy as np
    import torch
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'submissions'), exist_ok=True)
    
    # 设置日志
    setup_logger(
        log_file=os.path.join(args.output_dir, 'logs', 'main.log'),
        log_level='DEBUG' if args.verbose else 'INFO'
    )
    
    logger = get_logger(__name__)
    logger.info(f"环境设置完成，使用设备: {device}")
    logger.info(f"输出目录: {args.output_dir}")
    
    return device, logger


def load_config(args, logger):
    """加载配置"""
    try:
        config_manager = ConfigManager.from_args(args)
        config = config_manager.config
        
        # 更新配置
        config.data.data_dir = args.data_dir
        config.system.output_dir = args.output_dir
        config.training.epochs = args.epochs
        config.training.batch_size = args.batch_size
        config.training.learning_rate = args.learning_rate
        config.training.val_split = args.val_split
        config.system.num_workers = args.num_workers
        
        logger.info("配置加载完成")
        return config_manager, config
        
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        logger.info("使用默认配置")
        
        # 创建默认配置
        config_manager = ConfigManager()
        config = config_manager.config
        
        # 更新基本配置
        config.data.data_dir = args.data_dir
        config.system.output_dir = args.output_dir
        config.training.epochs = args.epochs
        config.training.batch_size = args.batch_size
        config.training.learning_rate = args.learning_rate
        config.training.val_split = args.val_split
        config.system.num_workers = args.num_workers
        
        return config_manager, config


def load_data(config, logger):
    """加载数据"""
    logger.info("开始加载数据...")
    
    try:
        # 创建数据加载器
        from src.data.loader import DataLoader
        data_loader = DataLoader()
        
        # 加载训练和测试数据
        train_data = data_loader.load_train_data()
        test_data = data_loader.load_test_data()
        
        # 划分训练集和验证集
        train_split, val_split = data_loader.split_train_validation(
            validation_ratio=config.training.val_split
        )
        
        # 创建数据集和数据加载器
        from src.data.dataset import create_dataloaders
        dataloaders = create_dataloaders(
            train_data=train_split,
            val_data=val_split,
            test_data=test_data
        )
        
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        test_loader = dataloaders['test']
        
        logger.info(f"数据加载完成:")
        logger.info(f"  训练集: {len(train_loader.dataset)} 样本")
        logger.info(f"  验证集: {len(val_loader.dataset)} 样本")
        logger.info(f"  测试集: {len(test_loader.dataset)} 样本")
        
        return data_loader, train_loader, val_loader, test_loader
        
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        raise


def create_models(config, device, logger):
    """创建模型"""
    logger.info("开始创建模型...")
    
    try:
        # 创建图像特征提取器
        image_extractor = ImageFeatureExtractor(
            model_name=config.image.model_name,
            feature_dim=config.image.feature_dim,
            pretrained=config.image.pretrained
        )
        
        # 创建文本特征提取器
        text_extractor = TextFeatureExtractor(
            model_name=config.text.model_name,
            feature_dim=config.text.feature_dim
        )
        
        # 创建相似度计算器
        similarity_calculator = SimilarityCalculator(
            similarity_method=config.similarity.method,
            temperature=config.similarity.temperature
        )
        
        logger.info("模型创建完成")
        logger.info(f"  图像模型: {config.image.model_name}")
        logger.info(f"  文本模型: {config.text.model_name}")
        logger.info(f"  相似度方法: {config.similarity.method}")
        
        return image_extractor, text_extractor, similarity_calculator
        
    except Exception as e:
        logger.error(f"模型创建失败: {e}")
        raise


def train_model(image_extractor, text_extractor, similarity_calculator,
               train_loader, val_loader, config, device, logger):
    """训练模型"""
    logger.info("开始训练模型...")
    
    try:
        # 创建训练器
        from dataclasses import asdict
        training_config = asdict(config.training)
        training_config['device'] = device  # 添加device配置
        trainer = CrossModalTrainer(
            image_extractor=image_extractor,
            text_extractor=text_extractor,
            similarity_calculator=similarity_calculator,
            config=training_config
        )
        
        # 创建检查点管理器
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=os.path.join(config.system.output_dir, 'checkpoints'),
            max_checkpoints=5  # 使用默认值
        )
        
        # 创建可视化器
        visualizer = Visualizer(
            output_dir=os.path.join(config.system.output_dir, 'visualizations')
        )
        
        # 训练模型
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.training.num_epochs,
            save_dir=os.path.join(config.system.output_dir, 'checkpoints')
        )
        
        # 可视化训练历史
        visualizer.plot_training_history(
            history=history,
            title="训练历史",
            save_name="training_history.png"
        )
        
        logger.info("模型训练完成")
        return trainer, history
        
    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        raise


def evaluate_model(image_extractor, text_extractor, similarity_calculator,
                  val_loader, config, device, logger):
    """评估模型"""
    logger.info("开始评估模型...")
    
    try:
        # 创建评估器
        evaluator = ModelEvaluator(
            image_extractor=image_extractor,
            text_extractor=text_extractor,
            similarity_calculator=similarity_calculator,
            device=device
        )
        
        # 评估模型
        metrics = evaluator.evaluate_dataset(
            dataloader=val_loader,
            top_k_list=[1, 3, 5, 10]
        )
        
        # 创建可视化器
        visualizer = Visualizer(
            output_dir=os.path.join(config.system.output_dir, 'visualizations')
        )
        
        # 生成评估报告
        report = evaluator.generate_report(
            metrics=metrics,
            save_path=os.path.join(config.system.output_dir, 'evaluation_report.json')
        )
        
        logger.info("模型评估完成")
        logger.info(f"主要指标:")
        logger.info(f"  Top-1 准确率: {metrics.get('top_1_accuracy', 0):.4f}")
        logger.info(f"  Top-5 准确率: {metrics.get('top_5_accuracy', 0):.4f}")
        logger.info(f"  平均排名: {metrics.get('mean_rank', 0):.2f}")
        logger.info(f"  MRR: {metrics.get('mrr', 0):.4f}")
        
        return metrics, report
        
    except Exception as e:
        logger.error(f"模型评估失败: {e}")
        raise


def predict_and_submit(image_extractor, text_extractor, similarity_calculator,
                      data_loader, test_loader, config, device, args, logger):
    """预测并生成提交文件"""
    logger.info("开始预测并生成提交文件...")
    
    try:
        # 创建预测器
        predictor = ChartPredictor(
            image_extractor=image_extractor,
            text_extractor=text_extractor,
            similarity_calculator=similarity_calculator,
            device=device
        )
        
        # 创建提交文件生成器
        submission_generator = SubmissionGenerator(
            data_loader=data_loader,
            predictor=predictor
        )
        
        # 生成预测结果
        if args.ensemble:
            logger.info("使用集成预测")
            submission_df = submission_generator.generate_submission(
                use_ensemble=True,
                use_tta=args.tta
            )
        else:
            submission_df = submission_generator.generate_submission(
                use_tta=args.tta
            )
        
        # 提交文件已自动保存
        submission_path = os.path.join(
            config.system.output_dir, 'submissions', 'submission.csv'
        )
        
        # 生成详细提交文件
        detailed_submission_results = submission_generator.generate_detailed_submission(
            k=5,
            include_confidence=True
        )
        detailed_submission_df = detailed_submission_results['standard']
        
        detailed_submission_path = os.path.join(
            config.system.output_dir, 'submissions', 'detailed_submission.csv'
        )
        # 详细提交文件已自动保存
        
        logger.info(f"提交文件已保存:")
        logger.info(f"  标准提交: {submission_path}")
        logger.info(f"  详细提交: {detailed_submission_path}")
        
        return submission_df, detailed_submission_df
        
    except Exception as e:
        logger.error(f"预测和提交文件生成失败: {e}")
        raise


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置环境
    device, logger = setup_environment(args)
    
    try:
        logger.info("=" * 60)
        logger.info("图表标题与内容解析匹配挑战赛解决方案")
        logger.info("=" * 60)
        
        # 加载配置
        config_manager, config = load_config(args, logger)
        
        # 加载数据
        data_loader, train_loader, val_loader, test_loader = load_data(config, logger)
        
        # 创建模型
        image_extractor, text_extractor, similarity_calculator = create_models(
            config, device, logger
        )
        
        # 加载检查点（如果指定）
        if args.checkpoint:
            logger.info(f"加载检查点: {args.checkpoint}")
            checkpoint_manager = CheckpointManager(
                checkpoint_dir=os.path.dirname(args.checkpoint)
            )
            checkpoint_manager.load_checkpoint(
                checkpoint_path=args.checkpoint,
                image_extractor=image_extractor,
                text_extractor=text_extractor,
                similarity_calculator=similarity_calculator
            )
        
        # 根据模式执行相应操作
        if args.mode in ['train', 'all']:
            trainer, history = train_model(
                image_extractor, text_extractor, similarity_calculator,
                train_loader, val_loader, config, device, logger
            )
        
        if args.mode in ['eval', 'all']:
            metrics, report = evaluate_model(
                image_extractor, text_extractor, similarity_calculator,
                val_loader, config, device, logger
            )
        
        if args.mode in ['predict', 'all']:
            submission_df, detailed_submission_df = predict_and_submit(
                image_extractor, text_extractor, similarity_calculator,
                data_loader, test_loader, config, device, args, logger
            )
        
        logger.info("=" * 60)
        logger.info("程序执行完成！")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()