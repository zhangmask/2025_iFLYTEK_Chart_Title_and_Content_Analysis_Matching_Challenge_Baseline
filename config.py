import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据路径配置
DATA_CONFIG = {
    'train_csv': PROJECT_ROOT / 'train.csv',
    'test_csv': PROJECT_ROOT / 'test.csv',
    'sample_submit_csv': PROJECT_ROOT / 'sample_submit.csv',
    'dataset_dir': PROJECT_ROOT / '图表文件' / 'dataset',
    'output_dir': PROJECT_ROOT / 'outputs',
    'models_dir': PROJECT_ROOT / 'models',
    'logs_dir': PROJECT_ROOT / 'logs'
}

# 模型配置
MODEL_CONFIG = {
    # 图像特征提取模型
    'image_model': {
        'name': 'resnet50',
        'pretrained': True,
        'feature_dim': 512,  # 使用投影层将2048维降到512维
        'backbone_dim': 2048,  # ResNet50实际输出维度
        'input_size': (224, 224),
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225]
    },
    
    # 文本特征提取模型
    'text_model': {
        'name': 'sentence-transformers/all-MiniLM-L6-v2',
        'feature_dim': 512,  # 使用投影层将384维升到512维
        'backbone_dim': 384,  # sentence-transformers实际输出维度
        'max_length': 512
    },
    
    # 跨模态匹配模型
    'cross_modal': {
        'embedding_dim': 512,
        'temperature': 0.07,
        'similarity_metric': 'cosine'  # cosine, euclidean, dot_product
    }
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 100,
    'early_stopping_patience': 10,
    'validation_split': 0.2,
    'random_seed': 42,
    'device': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
    'num_workers': 4,
    'pin_memory': True
}

# 预测配置
PREDICTION_CONFIG = {
    'batch_size': 64,
    'top_k': 5,  # 返回前k个最相似的结果
    'confidence_threshold': 0.5
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}',
    'rotation': '10 MB',
    'retention': '7 days'
}

# 路径配置（为了兼容性）
PATHS = {
    'data': PROJECT_ROOT,
    'models': DATA_CONFIG['models_dir'],
    'outputs': DATA_CONFIG['output_dir'],
    'logs': DATA_CONFIG['logs_dir'],
    'dataset': DATA_CONFIG['dataset_dir']
}

# 创建必要的目录
for dir_path in [DATA_CONFIG['output_dir'], DATA_CONFIG['models_dir'], DATA_CONFIG['logs_dir']]:
    dir_path.mkdir(exist_ok=True)