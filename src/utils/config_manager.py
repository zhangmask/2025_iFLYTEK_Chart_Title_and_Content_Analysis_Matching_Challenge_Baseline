import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from loguru import logger
import argparse


@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "data"
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    sample_submit_csv: str = "sample_submit.csv"
    chart_dir: str = "图表文件/dataset"
    output_dir: str = "outputs"
    models_dir: str = "models"
    logs_dir: str = "logs"
    cache_dir: str = "cache"


@dataclass
class ImageConfig:
    """图像配置"""
    model_name: str = "resnet50"
    pretrained: bool = True
    feature_dim: int = 512
    input_size: int = 224
    normalize_mean: list = None
    normalize_std: list = None
    augmentation: bool = True
    
    def __post_init__(self):
        if self.normalize_mean is None:
            self.normalize_mean = [0.485, 0.456, 0.406]
        if self.normalize_std is None:
            self.normalize_std = [0.229, 0.224, 0.225]


@dataclass
class TextConfig:
    """文本配置"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    feature_dim: int = 512
    max_length: int = 128
    use_projection: bool = True
    cache_dir: str = "cache/text_models"


@dataclass
class SimilarityConfig:
    """相似度配置"""
    method: str = "cosine"
    temperature: float = 1.0
    use_learned_similarity: bool = False
    hidden_dim: int = 256


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 50
    val_split: float = 0.2
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    patience: int = 10
    min_delta: float = 1e-4
    gradient_clip_norm: float = 1.0
    accumulation_steps: int = 1
    mixed_precision: bool = True
    
    # 优化器配置
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    
    # 学习率调度器配置
    scheduler: str = "cosine"
    scheduler_params: dict = None
    
    # 损失函数配置
    loss_function: str = "contrastive"
    margin: float = 0.2
    
    def __post_init__(self):
        if self.scheduler_params is None:
            self.scheduler_params = {}


@dataclass
class PredictionConfig:
    """预测配置"""
    batch_size: int = 64
    confidence_threshold: float = 0.5
    use_ensemble: bool = False
    use_tta: bool = False
    tta_transforms: int = 5
    top_k: int = 5


@dataclass
class SystemConfig:
    """系统配置"""
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = True
    log_level: str = "INFO"
    save_checkpoints: bool = True
    checkpoint_interval: int = 5


@dataclass
class Config:
    """主配置类"""
    data: DataConfig = None
    image: ImageConfig = None
    text: TextConfig = None
    similarity: SimilarityConfig = None
    training: TrainingConfig = None
    prediction: PredictionConfig = None
    system: SystemConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.image is None:
            self.image = ImageConfig()
        if self.text is None:
            self.text = TextConfig()
        if self.similarity is None:
            self.similarity = SimilarityConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.prediction is None:
            self.prediction = PredictionConfig()
        if self.system is None:
            self.system = SystemConfig()


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = Config()
        
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
        
        logger.info(f"配置管理器初始化完成，配置文件: {config_path}")
    
    def load_config(self, config_path: str) -> Config:
        """
        从文件加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置对象
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_dict = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_dict = json.load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
            
            self.config = self._dict_to_config(config_dict)
            self.config_path = str(config_path)
            
            logger.info(f"配置文件加载成功: {config_path}")
            return self.config
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def save_config(self, config_path: Optional[str] = None, format: str = 'yaml') -> str:
        """
        保存配置到文件
        
        Args:
            config_path: 配置文件路径
            format: 文件格式 ('yaml' 或 'json')
            
        Returns:
            保存的文件路径
        """
        if config_path is None:
            config_path = self.config_path or f"config.{format}"
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self._config_to_dict(self.config)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if format.lower() in ['yaml', 'yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
                elif format.lower() == 'json':
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"不支持的格式: {format}")
            
            logger.info(f"配置文件保存成功: {config_path}")
            return str(config_path)
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            raise
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> Config:
        """
        将字典转换为配置对象
        
        Args:
            config_dict: 配置字典
            
        Returns:
            配置对象
        """
        config = Config()
        
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        
        if 'image' in config_dict:
            config.image = ImageConfig(**config_dict['image'])
        
        if 'text' in config_dict:
            config.text = TextConfig(**config_dict['text'])
        
        if 'similarity' in config_dict:
            config.similarity = SimilarityConfig(**config_dict['similarity'])
        
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        
        if 'prediction' in config_dict:
            config.prediction = PredictionConfig(**config_dict['prediction'])
        
        if 'system' in config_dict:
            config.system = SystemConfig(**config_dict['system'])
        
        return config
    
    def _config_to_dict(self, config: Config) -> Dict[str, Any]:
        """
        将配置对象转换为字典
        
        Args:
            config: 配置对象
            
        Returns:
            配置字典
        """
        return {
            'data': asdict(config.data),
            'image': asdict(config.image),
            'text': asdict(config.text),
            'similarity': asdict(config.similarity),
            'training': asdict(config.training),
            'prediction': asdict(config.prediction),
            'system': asdict(config.system)
        }
    
    def update_config(self, **kwargs):
        """
        更新配置
        
        Args:
            **kwargs: 配置更新参数
        """
        for section, updates in kwargs.items():
            if hasattr(self.config, section):
                section_config = getattr(self.config, section)
                for key, value in updates.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
                        logger.debug(f"更新配置: {section}.{key} = {value}")
                    else:
                        logger.warning(f"未知配置项: {section}.{key}")
            else:
                logger.warning(f"未知配置节: {section}")
    
    def get_config(self) -> Config:
        """
        获取配置对象
        
        Returns:
            配置对象
        """
        return self.config
    
    def validate_config(self) -> bool:
        """
        验证配置有效性
        
        Returns:
            配置是否有效
        """
        try:
            # 验证路径
            data_paths = [
                self.config.data.train_csv,
                self.config.data.test_csv,
                self.config.data.sample_submit_csv,
                self.config.data.chart_dir
            ]
            
            missing_paths = []
            for path in data_paths:
                if not Path(path).exists():
                    missing_paths.append(path)
            
            if missing_paths:
                logger.warning(f"以下路径不存在: {missing_paths}")
            
            # 验证参数范围
            if self.config.training.learning_rate <= 0:
                logger.error("学习率必须大于0")
                return False
            
            if self.config.training.batch_size <= 0:
                logger.error("批处理大小必须大于0")
                return False
            
            if self.config.image.feature_dim <= 0:
                logger.error("图像特征维度必须大于0")
                return False
            
            if self.config.text.feature_dim <= 0:
                logger.error("文本特征维度必须大于0")
                return False
            
            logger.info("配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
    
    def create_directories(self):
        """
        创建必要的目录
        """
        directories = [
            self.config.data.output_dir,
            self.config.data.models_dir,
            self.config.data.logs_dir,
            self.config.data.cache_dir,
            self.config.text.cache_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"创建目录: {directory}")
        
        logger.info("必要目录创建完成")
    
    def get_model_save_path(self, model_name: str, epoch: Optional[int] = None) -> str:
        """
        获取模型保存路径
        
        Args:
            model_name: 模型名称
            epoch: 训练轮次
            
        Returns:
            模型保存路径
        """
        models_dir = Path(self.config.data.models_dir)
        
        if epoch is not None:
            filename = f"{model_name}_epoch_{epoch}.pth"
        else:
            filename = f"{model_name}_best.pth"
        
        return str(models_dir / filename)
    
    def get_output_path(self, filename: str) -> str:
        """
        获取输出文件路径
        
        Args:
            filename: 文件名
            
        Returns:
            输出文件路径
        """
        output_dir = Path(self.config.data.output_dir)
        return str(output_dir / filename)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'ConfigManager':
        """
        从命令行参数创建配置管理器
        
        Args:
            args: 命令行参数
            
        Returns:
            配置管理器实例
        """
        config_manager = cls()
        
        # 从命令行参数更新配置
        if hasattr(args, 'config') and args.config:
            config_manager.load_config(args.config)
        
        # 覆盖特定参数
        updates = {}
        
        # 数据配置
        if hasattr(args, 'data_dir') and args.data_dir:
            updates.setdefault('data', {})['data_dir'] = args.data_dir
        
        if hasattr(args, 'output_dir') and args.output_dir:
            updates.setdefault('data', {})['output_dir'] = args.output_dir
        
        # 训练配置
        if hasattr(args, 'batch_size') and args.batch_size:
            updates.setdefault('training', {})['batch_size'] = args.batch_size
        
        if hasattr(args, 'learning_rate') and args.learning_rate:
            updates.setdefault('training', {})['learning_rate'] = args.learning_rate
        
        if hasattr(args, 'epochs') and args.epochs:
            updates.setdefault('training', {})['num_epochs'] = args.epochs
        
        if hasattr(args, 'val_split') and args.val_split:
            updates.setdefault('training', {})['val_split'] = args.val_split
        
        # 系统配置
        if hasattr(args, 'device') and args.device:
            updates.setdefault('system', {})['device'] = args.device
        
        if hasattr(args, 'num_workers') and args.num_workers:
            updates.setdefault('system', {})['num_workers'] = args.num_workers
        
        if updates:
            config_manager.update_config(**updates)
        
        return config_manager
    
    def create_default_config_file(self, config_path: str = "config.yaml"):
        """
        创建默认配置文件
        
        Args:
            config_path: 配置文件路径
        """
        self.config = Config()  # 使用默认配置
        self.save_config(config_path)
        logger.info(f"默认配置文件已创建: {config_path}")