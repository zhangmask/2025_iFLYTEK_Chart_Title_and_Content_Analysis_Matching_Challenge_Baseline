import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import time
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

from src.features.image_features import ImageFeatureExtractor
from src.features.text_features import TextFeatureExtractor
from src.features.similarity import SimilarityCalculator
from src.data.preprocessor import ImagePreprocessor, TextPreprocessor


class InferenceEngine:
    """推理引擎 - 高效的批量推理处理"""
    
    def __init__(self,
                 image_extractor: ImageFeatureExtractor,
                 text_extractor: TextFeatureExtractor,
                 similarity_calculator: SimilarityCalculator,
                 device: str = 'auto',
                 max_batch_size: int = 32,
                 num_workers: int = 4,
                 enable_mixed_precision: bool = True):
        """
        初始化推理引擎
        
        Args:
            image_extractor: 图像特征提取器
            text_extractor: 文本特征提取器
            similarity_calculator: 相似度计算器
            device: 设备类型
            max_batch_size: 最大批处理大小
            num_workers: 工作线程数
            enable_mixed_precision: 是否启用混合精度
        """
        self.image_extractor = image_extractor
        self.text_extractor = text_extractor
        self.similarity_calculator = similarity_calculator
        
        # 设备配置
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.max_batch_size = max_batch_size
        self.num_workers = num_workers
        self.enable_mixed_precision = enable_mixed_precision
        
        # 移动模型到设备
        self._move_models_to_device()
        
        # 设置评估模式
        self._set_eval_mode()
        
        # 预处理器
        self.image_preprocessor = ImagePreprocessor()
        self.text_preprocessor = TextPreprocessor()
        
        # 性能统计
        self.stats = {
            'total_images_processed': 0,
            'total_texts_processed': 0,
            'total_inference_time': 0.0,
            'average_image_time': 0.0,
            'average_text_time': 0.0
        }
        
        logger.info(f"推理引擎初始化完成，设备: {self.device}")
    
    def _move_models_to_device(self):
        """移动模型到指定设备"""
        self.image_extractor.to(self.device)
        self.text_extractor.to(self.device)
        self.similarity_calculator.to(self.device)
    
    def _set_eval_mode(self):
        """设置模型为评估模式"""
        self.image_extractor.eval()
        self.text_extractor.eval()
        self.similarity_calculator.eval()
    
    @torch.no_grad()
    def extract_image_features_batch(self, 
                                   image_paths: List[str],
                                   batch_size: Optional[int] = None) -> torch.Tensor:
        """
        批量提取图像特征
        
        Args:
            image_paths: 图像路径列表
            batch_size: 批处理大小
            
        Returns:
            图像特征张量
        """
        if batch_size is None:
            batch_size = self.max_batch_size
        
        start_time = time.time()
        all_features = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # 预处理图像
            batch_images = []
            for path in batch_paths:
                try:
                    image = self.image_preprocessor.load_and_preprocess(path)
                    batch_images.append(image)
                except Exception as e:
                    logger.warning(f"加载图像失败 {path}: {e}")
                    # 使用零张量作为占位符
                    dummy_image = torch.zeros(3, 224, 224)
                    batch_images.append(dummy_image)
            
            if batch_images:
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                # 混合精度推理
                if self.enable_mixed_precision and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        features = self.image_extractor(batch_tensor)
                else:
                    features = self.image_extractor(batch_tensor)
                
                all_features.append(features.cpu())
            
            # 清理GPU内存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 合并所有特征
        if all_features:
            result = torch.cat(all_features, dim=0)
        else:
            result = torch.empty(0, self.image_extractor.feature_dim)
        
        # 更新统计信息
        inference_time = time.time() - start_time
        self.stats['total_images_processed'] += len(image_paths)
        self.stats['total_inference_time'] += inference_time
        self.stats['average_image_time'] = inference_time / len(image_paths)
        
        logger.info(f"批量提取 {len(image_paths)} 个图像特征，耗时: {inference_time:.2f}s")
        return result
    
    @torch.no_grad()
    def extract_text_features_batch(self, 
                                  texts: List[str],
                                  batch_size: Optional[int] = None) -> torch.Tensor:
        """
        批量提取文本特征
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            文本特征张量
        """
        if batch_size is None:
            batch_size = self.max_batch_size
        
        start_time = time.time()
        all_features = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 预处理文本
            processed_texts = [self.text_preprocessor.preprocess(text) for text in batch_texts]
            
            # 混合精度推理
            if self.enable_mixed_precision and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    features = self.text_extractor.extract_features_batch(processed_texts)
            else:
                features = self.text_extractor.extract_features_batch(processed_texts)
            
            all_features.append(features.cpu())
            
            # 清理GPU内存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 合并所有特征
        if all_features:
            result = torch.cat(all_features, dim=0)
        else:
            result = torch.empty(0, self.text_extractor.feature_dim)
        
        # 更新统计信息
        inference_time = time.time() - start_time
        self.stats['total_texts_processed'] += len(texts)
        self.stats['total_inference_time'] += inference_time
        self.stats['average_text_time'] = inference_time / len(texts)
        
        logger.info(f"批量提取 {len(texts)} 个文本特征，耗时: {inference_time:.2f}s")
        return result
    
    def compute_similarity_matrix(self, 
                                image_features: torch.Tensor,
                                text_features: torch.Tensor,
                                method: str = 'cosine',
                                temperature: float = 1.0) -> torch.Tensor:
        """
        计算相似度矩阵
        
        Args:
            image_features: 图像特征
            text_features: 文本特征
            method: 相似度计算方法
            temperature: 温度参数
            
        Returns:
            相似度矩阵
        """
        start_time = time.time()
        
        # 移动到设备
        image_features = image_features.to(self.device)
        text_features = text_features.to(self.device)
        
        # 计算相似度
        with torch.no_grad():
            if self.enable_mixed_precision and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    similarity_matrix = self.similarity_calculator.compute_similarity_matrix(
                        image_features, text_features, method=method, temperature=temperature
                    )
            else:
                similarity_matrix = self.similarity_calculator.compute_similarity_matrix(
                    image_features, text_features, method=method, temperature=temperature
                )
        
        inference_time = time.time() - start_time
        logger.info(f"计算相似度矩阵 ({image_features.shape[0]}x{text_features.shape[0]})，耗时: {inference_time:.2f}s")
        
        return similarity_matrix.cpu()
    
    def predict_batch(self, 
                     image_paths: List[str],
                     candidate_texts: List[str],
                     batch_size: Optional[int] = None,
                     return_similarities: bool = False) -> Union[List[int], Tuple[List[int], torch.Tensor]]:
        """
        批量预测
        
        Args:
            image_paths: 图像路径列表
            candidate_texts: 候选文本列表
            batch_size: 批处理大小
            return_similarities: 是否返回相似度矩阵
            
        Returns:
            预测结果列表，可选相似度矩阵
        """
        logger.info(f"开始批量预测: {len(image_paths)} 图像 x {len(candidate_texts)} 文本")
        
        # 提取特征
        image_features = self.extract_image_features_batch(image_paths, batch_size)
        text_features = self.extract_text_features_batch(candidate_texts, batch_size)
        
        # 计算相似度
        similarity_matrix = self.compute_similarity_matrix(image_features, text_features)
        
        # 获取最佳匹配
        predictions = torch.argmax(similarity_matrix, dim=1).tolist()
        
        logger.info(f"批量预测完成，预测结果数量: {len(predictions)}")
        
        if return_similarities:
            return predictions, similarity_matrix
        else:
            return predictions
    
    def predict_with_confidence(self, 
                              image_paths: List[str],
                              candidate_texts: List[str],
                              batch_size: Optional[int] = None,
                              confidence_threshold: float = 0.5) -> List[Dict]:
        """
        带置信度的批量预测
        
        Args:
            image_paths: 图像路径列表
            candidate_texts: 候选文本列表
            batch_size: 批处理大小
            confidence_threshold: 置信度阈值
            
        Returns:
            包含预测结果和置信度的字典列表
        """
        predictions, similarity_matrix = self.predict_batch(
            image_paths, candidate_texts, batch_size, return_similarities=True
        )
        
        results = []
        for i, pred_idx in enumerate(predictions):
            similarities = similarity_matrix[i]
            max_similarity = float(similarities[pred_idx])
            
            # 计算置信度（基于最大相似度与次大相似度的差异）
            sorted_similarities, sorted_indices = torch.sort(similarities, descending=True)
            if len(sorted_similarities) > 1:
                confidence = float(sorted_similarities[0] - sorted_similarities[1])
            else:
                confidence = float(sorted_similarities[0])
            
            # 获取top-k候选
            top_k_candidates = []
            for j in range(min(5, len(sorted_indices))):
                idx = int(sorted_indices[j])
                sim = float(sorted_similarities[j])
                top_k_candidates.append({
                    'text': candidate_texts[idx],
                    'similarity': sim,
                    'index': idx
                })
            
            results.append({
                'image_path': image_paths[i],
                'predicted_text': candidate_texts[pred_idx],
                'predicted_index': pred_idx,
                'confidence': confidence,
                'max_similarity': max_similarity,
                'is_confident': confidence > confidence_threshold,
                'top_k_candidates': top_k_candidates
            })
        
        return results
    
    def predict_parallel(self, 
                        image_paths: List[str],
                        candidate_texts: List[str],
                        batch_size: Optional[int] = None) -> List[int]:
        """
        并行预测（适用于大规模数据）
        
        Args:
            image_paths: 图像路径列表
            candidate_texts: 候选文本列表
            batch_size: 批处理大小
            
        Returns:
            预测结果列表
        """
        if batch_size is None:
            batch_size = self.max_batch_size
        
        logger.info(f"开始并行预测: {len(image_paths)} 图像")
        
        # 预先提取所有文本特征（一次性）
        text_features = self.extract_text_features_batch(candidate_texts, batch_size)
        
        # 分批处理图像
        all_predictions = []
        
        def process_image_batch(batch_paths):
            """处理单个图像批次"""
            batch_image_features = self.extract_image_features_batch(batch_paths, len(batch_paths))
            batch_similarity = self.compute_similarity_matrix(batch_image_features, text_features)
            return torch.argmax(batch_similarity, dim=1).tolist()
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                future = executor.submit(process_image_batch, batch_paths)
                futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    batch_predictions = future.result()
                    all_predictions.extend(batch_predictions)
                except Exception as e:
                    logger.error(f"并行预测批次失败: {e}")
        
        logger.info(f"并行预测完成，预测结果数量: {len(all_predictions)}")
        return all_predictions
    
    def benchmark_performance(self, 
                            num_images: int = 100,
                            num_texts: int = 1000,
                            batch_sizes: List[int] = [1, 8, 16, 32, 64]) -> Dict[str, any]:
        """
        性能基准测试
        
        Args:
            num_images: 测试图像数量
            num_texts: 测试文本数量
            batch_sizes: 测试的批处理大小列表
            
        Returns:
            性能测试结果
        """
        logger.info("开始性能基准测试...")
        
        # 生成测试数据
        dummy_image_features = torch.randn(num_images, self.image_extractor.feature_dim)
        dummy_text_features = torch.randn(num_texts, self.text_extractor.feature_dim)
        
        results = {
            'device': str(self.device),
            'mixed_precision': self.enable_mixed_precision,
            'batch_size_results': {}
        }
        
        for batch_size in batch_sizes:
            logger.info(f"测试批处理大小: {batch_size}")
            
            start_time = time.time()
            
            # 测试相似度计算
            with torch.no_grad():
                for i in range(0, num_images, batch_size):
                    batch_img_features = dummy_image_features[i:i + batch_size].to(self.device)
                    
                    if self.enable_mixed_precision and self.device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            similarity = self.compute_similarity_matrix(
                                batch_img_features, dummy_text_features.to(self.device)
                            )
                    else:
                        similarity = self.compute_similarity_matrix(
                            batch_img_features, dummy_text_features.to(self.device)
                        )
                    
                    # 清理内存
                    del similarity
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            total_time = time.time() - start_time
            throughput = num_images / total_time
            
            results['batch_size_results'][batch_size] = {
                'total_time': total_time,
                'throughput': throughput,
                'time_per_image': total_time / num_images
            }
            
            logger.info(f"批处理大小 {batch_size}: {throughput:.2f} images/sec")
        
        # 找到最佳批处理大小
        best_batch_size = max(
            results['batch_size_results'].keys(),
            key=lambda x: results['batch_size_results'][x]['throughput']
        )
        results['best_batch_size'] = best_batch_size
        
        logger.info(f"性能基准测试完成，最佳批处理大小: {best_batch_size}")
        return results
    
    def get_memory_usage(self) -> Dict[str, any]:
        """
        获取内存使用情况
        
        Returns:
            内存使用信息
        """
        memory_info = {
            'device': str(self.device)
        }
        
        if self.device.type == 'cuda':
            memory_info.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
                'gpu_max_memory_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
            })
        
        # CPU内存（需要psutil）
        try:
            import psutil
            process = psutil.Process()
            memory_info['cpu_memory_mb'] = process.memory_info().rss / 1024**2  # MB
        except ImportError:
            memory_info['cpu_memory_mb'] = 'psutil not available'
        
        return memory_info
    
    def clear_cache(self):
        """清理缓存"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("缓存已清理")
    
    def get_stats(self) -> Dict[str, any]:
        """获取性能统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置性能统计信息"""
        self.stats = {
            'total_images_processed': 0,
            'total_texts_processed': 0,
            'total_inference_time': 0.0,
            'average_image_time': 0.0,
            'average_text_time': 0.0
        }
        logger.info("性能统计信息已重置")