import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path
import json
from datetime import datetime
from loguru import logger

from src.data.loader import DataLoader as ChartDataLoader
from .predictor import ChartPredictor


class SubmissionGenerator:
    """提交文件生成器"""
    
    def __init__(self, 
                 data_loader: ChartDataLoader,
                 predictor: ChartPredictor):
        """
        初始化提交文件生成器
        
        Args:
            data_loader: 数据加载器
            predictor: 预测器
        """
        self.data_loader = data_loader
        self.predictor = predictor
        
        logger.info("提交文件生成器初始化完成")
    
    def generate_submission(self, 
                          output_path: str = 'submission.csv',
                          batch_size: int = 32,
                          use_ensemble: bool = False,
                          use_tta: bool = False) -> pd.DataFrame:
        """
        生成提交文件
        
        Args:
            output_path: 输出路径
            batch_size: 批处理大小
            use_ensemble: 是否使用集成方法
            use_tta: 是否使用测试时增强
            
        Returns:
            提交DataFrame
        """
        logger.info("开始生成提交文件...")
        
        # 获取测试数据
        test_df = self.data_loader.test_data
        if test_df.empty:
            raise ValueError("测试数据为空")
        
        # 获取候选文本（从sample_submit.csv中获取，而不是训练数据）
        candidate_texts = self.data_loader.get_candidate_captions()
        image_paths = test_df['Source'].tolist()
        
        logger.info(f"测试图像数量: {len(image_paths)}, 候选文本数量: {len(candidate_texts)}")
        
        # 执行预测
        if use_ensemble:
            predictions = self.predictor.predict_with_ensemble(
                image_paths, candidate_texts, batch_size=batch_size
            )
        elif use_tta:
            predictions = self.predictor.predict_with_tta(
                image_paths, candidate_texts, batch_size=batch_size
            )
        else:
            predictions = self.predictor.predict_batch(
                image_paths, candidate_texts, batch_size=batch_size
            )
        
        # 创建提交DataFrame
        submission_df = self._create_submission_dataframe(
            test_df, predictions, candidate_texts
        )
        
        # 保存提交文件
        self._save_submission(submission_df, output_path)
        
        logger.info(f"提交文件生成完成: {output_path}")
        return submission_df
    
    def _create_submission_dataframe(self, 
                                   test_df: pd.DataFrame,
                                   predictions: List[int],
                                   candidate_texts: List[str]) -> pd.DataFrame:
        """
        创建提交DataFrame
        
        Args:
            test_df: 测试数据DataFrame
            predictions: 预测结果列表
            candidate_texts: 候选文本列表
            
        Returns:
            提交DataFrame
        """
        # 检查预测结果数量
        if len(predictions) != len(test_df):
            raise ValueError(f"预测结果数量({len(predictions)})与测试数据数量({len(test_df)})不匹配")
        
        # 创建提交格式
        submission_data = []
        
        for i, (_, row) in enumerate(test_df.iterrows()):
            pred_idx = predictions[i]
            
            # 确保预测索引在有效范围内
            if pred_idx >= len(candidate_texts):
                logger.warning(f"预测索引 {pred_idx} 超出候选文本范围，使用索引 0")
                pred_idx = 0
            
            submission_data.append({
                'Source': row['Source'],
                'Caption': candidate_texts[pred_idx]
            })
        
        submission_df = pd.DataFrame(submission_data)
        
        # 验证提交格式
        self._validate_submission_format(submission_df)
        
        return submission_df
    
    def _validate_submission_format(self, submission_df: pd.DataFrame):
        """
        验证提交文件格式
        
        Args:
            submission_df: 提交DataFrame
        """
        required_columns = ['Source', 'Caption']
        
        # 检查必需列
        for col in required_columns:
            if col not in submission_df.columns:
                raise ValueError(f"提交文件缺少必需列: {col}")
        
        # 检查空值
        if submission_df.isnull().any().any():
            null_counts = submission_df.isnull().sum()
            logger.warning(f"提交文件包含空值: {null_counts.to_dict()}")
        
        # 检查重复
        duplicates = submission_df.duplicated(subset=['Source']).sum()
        if duplicates > 0:
            logger.warning(f"提交文件包含 {duplicates} 个重复的Source")
        
        logger.info("提交文件格式验证通过")
    
    def _save_submission(self, submission_df: pd.DataFrame, output_path: str):
        """
        保存提交文件
        
        Args:
            submission_df: 提交DataFrame
            output_path: 输出路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为CSV格式
        submission_df.to_csv(output_path, index=False, encoding='utf-8')
        
        # 同时保存一份带时间戳的备份
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = output_path.parent / f"{output_path.stem}_{timestamp}{output_path.suffix}"
        submission_df.to_csv(backup_path, index=False, encoding='utf-8')
        
        logger.info(f"提交文件已保存: {output_path}")
        logger.info(f"备份文件已保存: {backup_path}")
    
    def generate_detailed_submission(self, 
                                   output_dir: str = 'submission_detailed',
                                   batch_size: int = 32,
                                   include_confidence: bool = True,
                                   include_top_k: bool = True,
                                   k: int = 5) -> Dict[str, pd.DataFrame]:
        """
        生成详细的提交文件，包含置信度和候选项
        
        Args:
            output_dir: 输出目录
            batch_size: 批处理大小
            include_confidence: 是否包含置信度信息
            include_top_k: 是否包含top-k候选
            k: top-k的k值
            
        Returns:
            包含多个DataFrame的字典
        """
        logger.info("开始生成详细提交文件...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取测试数据
        test_df = self.data_loader.test_data
        candidate_texts = self.data_loader.get_candidate_captions()
        image_paths = test_df['Source'].tolist()
        
        # 执行带置信度的预测
        detailed_predictions = self.predictor.predict_with_confidence(
            image_paths, candidate_texts, batch_size=batch_size
        )
        
        results = {}
        
        # 1. 标准提交文件
        standard_submission = pd.DataFrame([
            {
                'Source': pred['image_path'],
                'Caption': pred['predicted_text']
            }
            for pred in detailed_predictions
        ])
        results['standard'] = standard_submission
        standard_submission.to_csv(output_dir / 'submission.csv', index=False)
        
        # 2. 带置信度的提交文件
        if include_confidence:
            confidence_submission = pd.DataFrame([
                {
                    'Source': pred['image_path'],
                    'Caption': pred['predicted_text'],
                    'confidence': pred['confidence'],
                    'max_similarity': pred['max_similarity'],
                    'is_confident': pred['is_confident']
                }
                for pred in detailed_predictions
            ])
            results['confidence'] = confidence_submission
            confidence_submission.to_csv(output_dir / 'submission_with_confidence.csv', index=False)
        
        # 3. Top-k候选文件
        if include_top_k:
            topk_data = []
            for pred in detailed_predictions:
                base_row = {
                    'Source': pred['image_path'],
                    'predicted_caption': pred['predicted_text']
                }
                
                for i, candidate in enumerate(pred['top_k_candidates'][:k]):
                    base_row[f'candidate_{i+1}'] = candidate['text']
                    base_row[f'similarity_{i+1}'] = candidate['similarity']
                
                topk_data.append(base_row)
            
            topk_submission = pd.DataFrame(topk_data)
            results['topk'] = topk_submission
            topk_submission.to_csv(output_dir / f'submission_top{k}.csv', index=False)
        
        # 4. 统计信息
        stats = self._generate_submission_stats(detailed_predictions)
        results['stats'] = pd.DataFrame([stats])
        
        # 保存统计信息
        with open(output_dir / 'submission_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"详细提交文件已生成到目录: {output_dir}")
        return results
    
    def _generate_submission_stats(self, predictions: List[Dict]) -> Dict[str, any]:
        """
        生成提交统计信息
        
        Args:
            predictions: 预测结果列表
            
        Returns:
            统计信息字典
        """
        confidences = [pred['confidence'] for pred in predictions]
        similarities = [pred['max_similarity'] for pred in predictions]
        confident_count = sum(pred['is_confident'] for pred in predictions)
        
        stats = {
            'total_predictions': len(predictions),
            'confident_predictions': confident_count,
            'confident_ratio': confident_count / len(predictions),
            'confidence_stats': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'median': float(np.median(confidences))
            },
            'similarity_stats': {
                'mean': float(np.mean(similarities)),
                'std': float(np.std(similarities)),
                'min': float(np.min(similarities)),
                'max': float(np.max(similarities)),
                'median': float(np.median(similarities))
            },
            'generation_time': datetime.now().isoformat()
        }
        
        return stats
    
    def compare_with_sample(self, 
                          submission_df: pd.DataFrame,
                          sample_submission_path: str = 'sample_submit.csv') -> Dict[str, any]:
        """
        与样例提交文件比较
        
        Args:
            submission_df: 生成的提交DataFrame
            sample_submission_path: 样例提交文件路径
            
        Returns:
            比较结果字典
        """
        try:
            sample_df = pd.read_csv(sample_submission_path)
            
            comparison = {
                'format_match': True,
                'row_count_match': len(submission_df) == len(sample_df),
                'column_match': list(submission_df.columns) == list(sample_df.columns),
                'source_match': submission_df['Source'].equals(sample_df['Source']),
                'differences': []
            }
            
            # 检查格式差异
            if not comparison['row_count_match']:
                comparison['differences'].append(
                    f"行数不匹配: 生成{len(submission_df)}, 样例{len(sample_df)}"
                )
            
            if not comparison['column_match']:
                comparison['differences'].append(
                    f"列名不匹配: 生成{list(submission_df.columns)}, 样例{list(sample_df.columns)}"
                )
            
            if not comparison['source_match']:
                comparison['differences'].append("Source列内容不匹配")
            
            comparison['format_match'] = len(comparison['differences']) == 0
            
            logger.info(f"与样例提交文件比较完成，格式匹配: {comparison['format_match']}")
            
        except Exception as e:
            logger.error(f"比较样例提交文件时出错: {e}")
            comparison = {'error': str(e)}
        
        return comparison
    
    def validate_submission_file(self, submission_path: str) -> Dict[str, any]:
        """
        验证提交文件
        
        Args:
            submission_path: 提交文件路径
            
        Returns:
            验证结果字典
        """
        try:
            df = pd.read_csv(submission_path)
            
            validation = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'stats': {
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'columns': list(df.columns)
                }
            }
            
            # 检查必需列
            required_columns = ['Source', 'Caption']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation['errors'].append(f"缺少必需列: {missing_columns}")
                validation['valid'] = False
            
            # 检查空值
            null_counts = df.isnull().sum()
            if null_counts.any():
                validation['warnings'].append(f"包含空值: {null_counts.to_dict()}")
            
            # 检查重复
            if 'Source' in df.columns:
                duplicates = df.duplicated(subset=['Source']).sum()
                if duplicates > 0:
                    validation['warnings'].append(f"包含 {duplicates} 个重复的Source")
            
            # 检查文件大小
            file_size = Path(submission_path).stat().st_size
            validation['stats']['file_size_mb'] = file_size / (1024 * 1024)
            
            logger.info(f"提交文件验证完成，有效: {validation['valid']}")
            
        except Exception as e:
            logger.error(f"验证提交文件时出错: {e}")
            validation = {
                'valid': False,
                'errors': [str(e)],
                'warnings': [],
                'stats': {}
            }
        
        return validation