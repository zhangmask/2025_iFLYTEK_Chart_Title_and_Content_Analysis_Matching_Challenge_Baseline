import os
from pathlib import Path
from typing import Union


def fix_image_path(image_path: Union[str, Path]) -> str:
    """
    修复图像路径，将相对路径转换为正确的绝对路径
    
    Args:
        image_path: 原始图像路径（可能是相对路径）
        
    Returns:
        修复后的绝对路径
    """
    image_path = str(image_path)
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    
    # 处理相对路径
    if image_path.startswith('./'):
        relative_path = image_path[2:]  # 移除 './'
    else:
        relative_path = image_path
    
    # 构建完整路径：项目根目录/图表文件/相对路径
    full_path = project_root / "图表文件" / relative_path
    
    return str(full_path)


def validate_image_path(image_path: Union[str, Path]) -> bool:
    """
    验证图像路径是否存在
    
    Args:
        image_path: 图像路径
        
    Returns:
        路径是否存在
    """
    return Path(image_path).exists()


def get_project_root() -> Path:
    """
    获取项目根目录
    
    Returns:
        项目根目录路径
    """
    return Path(__file__).parent.parent.parent