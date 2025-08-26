import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger

from config import DATA_CONFIG


class DataLoader:
    """数据加载器，负责读取和管理训练、测试数据"""
    
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.sample_submit = None
    
    @property
    def train_df(self) -> pd.DataFrame:
        """获取训练数据DataFrame"""
        if self.train_data is None:
            self.load_train_data()
        return self.train_data
    
    @property
    def test_df(self) -> pd.DataFrame:
        """获取测试数据DataFrame"""
        if self.test_data is None:
            self.load_test_data()
        return self.test_data
        
    def load_train_data(self) -> pd.DataFrame:
        """加载训练数据"""
        try:
            self.train_data = pd.read_csv(DATA_CONFIG['train_csv'])
            logger.info(f"训练数据加载成功，共 {len(self.train_data)} 条记录")
            return self.train_data
        except Exception as e:
            logger.error(f"训练数据加载失败: {e}")
            raise
    
    def load_test_data(self) -> pd.DataFrame:
        """加载测试数据"""
        try:
            self.test_data = pd.read_csv(DATA_CONFIG['test_csv'])
            logger.info(f"测试数据加载成功，共 {len(self.test_data)} 条记录")
            return self.test_data
        except Exception as e:
            logger.error(f"测试数据加载失败: {e}")
            raise
    
    def load_sample_submit(self) -> pd.DataFrame:
        """加载提交样例"""
        try:
            self.sample_submit = pd.read_csv(DATA_CONFIG['sample_submit_csv'])
            logger.info(f"提交样例加载成功，共 {len(self.sample_submit)} 条记录")
            return self.sample_submit
        except Exception as e:
            logger.error(f"提交样例加载失败: {e}")
            raise
    
    def get_file_statistics(self) -> Dict:
        """获取文件统计信息"""
        if self.train_data is None:
            self.load_train_data()
        
        # 统计文件类型分布
        file_types = {}
        for source in self.train_data['Source']:
            ext = Path(source).suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
        
        # 统计标注文本长度
        caption_lengths = self.train_data['Caption'].str.len()
        
        stats = {
            'total_files': len(self.train_data),
            'file_types': file_types,
            'caption_stats': {
                'mean_length': caption_lengths.mean(),
                'median_length': caption_lengths.median(),
                'min_length': caption_lengths.min(),
                'max_length': caption_lengths.max()
            }
        }
        
        logger.info(f"数据统计: {stats}")
        return stats
    
    def validate_file_paths(self) -> Tuple[List[str], List[str]]:
        """验证文件路径是否存在"""
        if self.train_data is None:
            self.load_train_data()
        if self.test_data is None:
            self.load_test_data()
        
        missing_files = []
        existing_files = []
        
        # 检查训练集文件
        for source in self.train_data['Source']:
            file_path = DATA_CONFIG['dataset_dir'] / source.replace('./dataset/', '')
            if file_path.exists():
                existing_files.append(str(file_path))
            else:
                missing_files.append(str(file_path))
        
        # 检查测试集文件
        for source in self.test_data['Source']:
            file_path = DATA_CONFIG['dataset_dir'] / source.replace('./dataset/', '')
            if file_path.exists():
                existing_files.append(str(file_path))
            else:
                missing_files.append(str(file_path))
        
        if missing_files:
            logger.warning(f"发现 {len(missing_files)} 个缺失文件")
        else:
            logger.info("所有文件路径验证通过")
        
        return existing_files, missing_files
    
    def split_train_validation(self, validation_ratio: float = 0.2, random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """划分训练集和验证集"""
        if self.train_data is None:
            self.load_train_data()
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 随机划分
        indices = np.random.permutation(len(self.train_data))
        split_idx = int(len(self.train_data) * (1 - validation_ratio))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_split = self.train_data.iloc[train_indices].reset_index(drop=True)
        val_split = self.train_data.iloc[val_indices].reset_index(drop=True)
        
        logger.info(f"数据划分完成: 训练集 {len(train_split)} 条，验证集 {len(val_split)} 条")
        
        return train_split, val_split
    
    def get_all_captions(self) -> List[str]:
        """获取所有标注文本，用于文本特征提取"""
        if self.train_data is None:
            self.load_train_data()
        
        return self.train_data['Caption'].tolist()
    
    def get_candidate_captions(self) -> List[str]:
        """获取候选标题集合，用于测试时预测"""
        if self.sample_submit is None:
            self.load_sample_submit()
        
        return self.sample_submit['Caption'].tolist()
    
    def create_submission_template(self) -> pd.DataFrame:
        """创建提交文件模板"""
        if self.test_data is None:
            self.load_test_data()
        
        submission = pd.DataFrame({
            'Source': self.test_data['Source'],
            'Caption': [''] * len(self.test_data)  # 空的预测结果
        })
        
        return submission