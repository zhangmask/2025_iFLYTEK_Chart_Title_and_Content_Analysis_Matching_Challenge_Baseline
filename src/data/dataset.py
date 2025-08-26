import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
from loguru import logger

from .preprocessor import ImagePreprocessor, TextPreprocessor
from config import DATA_CONFIG, TRAINING_CONFIG

if TYPE_CHECKING:
    from .loader import DataLoader as CustomDataLoader


class ChartDataset(Dataset):
    """图表数据集类，用于PyTorch训练和推理"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 dataset_dir: Path,
                 mode: str = 'train',
                 augment: bool = False):
        """
        Args:
            data: 包含Source和Caption列的DataFrame
            dataset_dir: 数据集目录路径
            mode: 'train', 'val', 'test'
            augment: 是否使用数据增强
        """
        self.data = data
        self.dataset_dir = Path(dataset_dir)
        self.mode = mode
        self.augment = augment
        
        self.image_preprocessor = ImagePreprocessor()
        self.text_preprocessor = TextPreprocessor()
        
        # 验证文件路径
        self._validate_files()
        
        logger.info(f"数据集初始化完成: {mode} 模式，{len(self.data)} 个样本")
    
    def _validate_files(self):
        """验证文件是否存在"""
        valid_indices = []
        for idx, row in self.data.iterrows():
            file_path = self._get_file_path(row['Source'])
            if file_path.exists():
                valid_indices.append(idx)
            else:
                logger.warning(f"文件不存在: {file_path}")
        
        self.data = self.data.loc[valid_indices].reset_index(drop=True)
        logger.info(f"验证后保留 {len(self.data)} 个有效样本")
    
    def _get_file_path(self, source: str) -> Path:
        """获取文件的完整路径"""
        # 移除 './dataset/' 前缀
        relative_path = source.replace('./dataset/', '')
        return self.dataset_dir / relative_path
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取单个样本"""
        row = self.data.iloc[idx]
        
        # 加载和预处理图像
        file_path = self._get_file_path(row['Source'])
        try:
            image_tensor = self.image_preprocessor.preprocess_from_path(
                file_path, augment=self.augment
            )
        except Exception as e:
            logger.error(f"图像处理失败 {file_path}: {e}")
            # 返回零张量作为fallback
            image_tensor = torch.zeros(3, *self.image_preprocessor.input_size)
        
        sample = {
            'image': image_tensor,
            'source': row['Source'],
            'file_path': str(file_path)
        }
        
        # 如果有标注文本（训练/验证模式）
        if 'Caption' in row and pd.notna(row['Caption']):
            caption = self.text_preprocessor.preprocess_text(row['Caption'])
            sample['caption'] = caption
        
        return sample
    
    def get_all_captions(self) -> List[str]:
        """获取所有标注文本"""
        if 'Caption' not in self.data.columns:
            return []
        
        captions = self.data['Caption'].dropna().tolist()
        return self.text_preprocessor.batch_preprocess_text(captions)


class ChartDataLoader:
    """图表数据加载器"""
    
    def __init__(self, 
                 dataset: ChartDataset,
                 batch_size: int = None,
                 shuffle: bool = False,
                 num_workers: int = None,
                 pin_memory: bool = None,
                 data_loader_instance: Optional['CustomDataLoader'] = None):
        """
        初始化数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批处理大小
            shuffle: 是否打乱数据
            num_workers: 工作进程数
            pin_memory: 是否固定内存
            data_loader_instance: DataLoader实例的引用
        """
        # 使用配置文件中的默认值
        self.batch_size = batch_size or TRAINING_CONFIG['batch_size']
        self.num_workers = num_workers or TRAINING_CONFIG['num_workers']
        self.pin_memory = pin_memory if pin_memory is not None else TRAINING_CONFIG['pin_memory']
        
        self.dataset = dataset
        self.data_loader_instance = data_loader_instance
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """自定义批处理函数"""
        # 收集图像张量
        images = torch.stack([item['image'] for item in batch])
        
        # 收集其他信息
        sources = [item['source'] for item in batch]
        file_paths = [item['file_path'] for item in batch]
        
        result = {
            'image': images,
            'sources': sources,
            'file_paths': file_paths
        }
        
        # 如果有标注文本
        if 'caption' in batch[0]:
            captions = [item['caption'] for item in batch]
            result['captions'] = captions
        
        return result
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    @property
    def train_df(self) -> pd.DataFrame:
        """获取训练数据DataFrame"""
        if self.data_loader_instance:
            return self.data_loader_instance.train_df
        return pd.DataFrame()
    
    @property
    def test_df(self) -> pd.DataFrame:
        """获取测试数据DataFrame"""
        if self.data_loader_instance:
            return self.data_loader_instance.test_df
        return pd.DataFrame()
    
    def get_candidate_captions(self) -> List[str]:
        """获取候选标题集合，用于测试时预测"""
        if self.data_loader_instance:
            return self.data_loader_instance.get_candidate_captions()
        return []


def create_dataloaders(train_data: pd.DataFrame, 
                      val_data: pd.DataFrame = None,
                      test_data: pd.DataFrame = None,
                      data_loader_instance: Optional['DataLoader'] = None) -> Dict[str, ChartDataLoader]:
    """创建训练、验证、测试数据加载器"""
    dataloaders = {}
    
    # 训练数据加载器
    if train_data is not None:
        train_dataset = ChartDataset(
            data=train_data,
            dataset_dir=DATA_CONFIG['dataset_dir'],
            mode='train',
            augment=True
        )
        dataloaders['train'] = ChartDataLoader(
            dataset=train_dataset,
            shuffle=True,
            data_loader_instance=data_loader_instance
        )
    
    # 验证数据加载器
    if val_data is not None:
        val_dataset = ChartDataset(
            data=val_data,
            dataset_dir=DATA_CONFIG['dataset_dir'],
            mode='val',
            augment=False
        )
        dataloaders['val'] = ChartDataLoader(
            dataset=val_dataset,
            shuffle=False,
            data_loader_instance=data_loader_instance
        )
    
    # 测试数据加载器
    if test_data is not None:
        test_dataset = ChartDataset(
            data=test_data,
            dataset_dir=DATA_CONFIG['dataset_dir'],
            mode='test',
            augment=False
        )
        dataloaders['test'] = ChartDataLoader(
            dataset=test_dataset,
            shuffle=False,
            data_loader_instance=data_loader_instance
        )
    
    logger.info(f"创建了 {list(dataloaders.keys())} 数据加载器")
    return dataloaders