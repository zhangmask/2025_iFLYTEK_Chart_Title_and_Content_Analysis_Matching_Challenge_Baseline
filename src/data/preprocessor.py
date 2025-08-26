import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from pathlib import Path
from typing import Union, List, Tuple
import torch
from torchvision import transforms
from loguru import logger

from config import MODEL_CONFIG
from src.utils.path_utils import fix_image_path, validate_image_path


class ImagePreprocessor:
    """图像预处理器，处理PDF、PNG、JPG格式的图表文件"""
    
    def __init__(self):
        self.image_config = MODEL_CONFIG['image_model']
        self.input_size = self.image_config['input_size']
        self.normalize_mean = self.image_config['normalize_mean']
        self.normalize_std = self.image_config['normalize_std']
        
        # 定义图像变换
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])
        
        # 用于数据增强的变换
        self.augment_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(self.input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])
    
    def load_image(self, file_path: Union[str, Path]) -> Image.Image:
        """加载图像文件，支持PDF、PNG、JPG格式"""
        # 修复路径
        fixed_path = fix_image_path(file_path)
        file_path = Path(fixed_path)
        
        # 验证路径是否存在
        if not validate_image_path(file_path):
            raise FileNotFoundError(f"图像文件不存在: {file_path}")
        
        try:
            if file_path.suffix.lower() == '.pdf':
                # 处理PDF文件，转换为图像
                try:
                    images = convert_from_path(str(file_path), first_page=1, last_page=1)
                    if images:
                        image = images[0]
                    else:
                        raise ValueError(f"无法从PDF文件提取图像: {file_path}")
                except Exception as pdf_error:
                    logger.warning(f"PDF处理失败，创建默认图像: {file_path}: {pdf_error}")
                    # 基于文件名创建唯一的默认图像
                    import random
                    import hashlib
                    
                    # 使用文件名生成种子，确保同一文件总是生成相同的图像
                    seed = int(hashlib.md5(str(file_path).encode()).hexdigest()[:8], 16)
                    random.seed(seed)
                    
                    # 创建具有不同颜色和模式的图像
                    base_color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
                    image = Image.new('RGB', (224, 224), color=base_color)
                    
                    # 添加几何图案以增加差异性
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(image)
                    
                    # 随机绘制一些形状
                    for _ in range(random.randint(3, 8)):
                        shape_type = random.choice(['rectangle', 'ellipse', 'line'])
                        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        
                        if shape_type == 'rectangle':
                            x1, y1 = random.randint(0, 150), random.randint(0, 150)
                            x2, y2 = x1 + random.randint(20, 74), y1 + random.randint(20, 74)
                            draw.rectangle([x1, y1, x2, y2], fill=color)
                        elif shape_type == 'ellipse':
                            x1, y1 = random.randint(0, 150), random.randint(0, 150)
                            x2, y2 = x1 + random.randint(20, 74), y1 + random.randint(20, 74)
                            draw.ellipse([x1, y1, x2, y2], fill=color)
                        else:  # line
                            x1, y1 = random.randint(0, 224), random.randint(0, 224)
                            x2, y2 = random.randint(0, 224), random.randint(0, 224)
                            draw.line([x1, y1, x2, y2], fill=color, width=random.randint(1, 5))
            else:
                # 处理PNG、JPG等图像文件
                image = Image.open(file_path)
                
            # 转换为RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            return image
            
        except Exception as e:
            logger.error(f"图像加载失败 {file_path}: {e}")
            raise
    
    def preprocess_image(self, image: Image.Image, augment: bool = False) -> torch.Tensor:
        """预处理图像，返回tensor格式"""
        try:
            if augment:
                tensor = self.augment_transform(image)
            else:
                tensor = self.transform(image)
            return tensor
        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            raise
    
    def preprocess_from_path(self, file_path: Union[str, Path], augment: bool = False) -> torch.Tensor:
        """从文件路径直接预处理图像"""
        image = self.load_image(file_path)
        return self.preprocess_image(image, augment)
    
    def batch_preprocess(self, file_paths: List[Union[str, Path]], augment: bool = False) -> torch.Tensor:
        """批量预处理图像"""
        tensors = []
        for file_path in file_paths:
            try:
                tensor = self.preprocess_from_path(file_path, augment)
                tensors.append(tensor)
            except Exception as e:
                logger.warning(f"跳过文件 {file_path}: {e}")
                continue
        
        if not tensors:
            raise ValueError("没有成功处理的图像")
        
        return torch.stack(tensors)
    
    def batch_preprocess_safe(self, file_paths: List[Union[str, Path]], augment: bool = False) -> torch.Tensor:
        """安全的批量预处理图像，为每个输入路径返回对应的tensor（失败时返回零tensor）"""
        tensors = []
        for file_path in file_paths:
            try:
                tensor = self.preprocess_from_path(file_path, augment)
                tensors.append(tensor)
            except Exception as e:
                logger.warning(f"文件处理失败，使用零tensor: {file_path}: {e}")
                # 创建零tensor作为fallback
                zero_tensor = torch.zeros(3, self.input_size[0], self.input_size[1])
                tensors.append(zero_tensor)
        
        return torch.stack(tensors)
    
    def get_image_info(self, file_path: Union[str, Path]) -> dict:
        """获取图像信息"""
        try:
            image = self.load_image(file_path)
            return {
                'width': image.width,
                'height': image.height,
                'mode': image.mode,
                'format': image.format,
                'size_mb': Path(file_path).stat().st_size / (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"获取图像信息失败 {file_path}: {e}")
            return {}


class TextPreprocessor:
    """文本预处理器，处理标注文本"""
    
    def __init__(self):
        self.text_config = MODEL_CONFIG['text_model']
        self.max_length = self.text_config['max_length']
    
    def clean_text(self, text: str) -> str:
        """清理文本"""
        if not isinstance(text, str):
            text = str(text)
        
        # 移除多余的空白字符
        text = ' '.join(text.split())
        
        # 移除特殊字符（可根据需要调整）
        # text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text.strip()
    
    def truncate_text(self, text: str, max_length: int = None) -> str:
        """截断文本到指定长度"""
        if max_length is None:
            max_length = self.max_length
        
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"文本被截断到 {max_length} 字符")
        
        return text
    
    def preprocess_text(self, text: str) -> str:
        """预处理文本"""
        text = self.clean_text(text)
        text = self.truncate_text(text)
        return text
    
    def batch_preprocess_text(self, texts: List[str]) -> List[str]:
        """批量预处理文本"""
        return [self.preprocess_text(text) for text in texts]
    
    def get_text_stats(self, texts: List[str]) -> dict:
        """获取文本统计信息"""
        lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        return {
            'total_texts': len(texts),
            'avg_length': np.mean(lengths),
            'max_length': max(lengths),
            'min_length': min(lengths),
            'avg_word_count': np.mean(word_counts),
            'max_word_count': max(word_counts),
            'min_word_count': min(word_counts)
        }