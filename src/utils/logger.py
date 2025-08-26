import sys
import os
from pathlib import Path
from loguru import logger
from typing import Optional, Union
from datetime import datetime


def setup_logger(log_level: str = "INFO",
                log_file: Optional[str] = None,
                log_dir: str = "logs",
                rotation: str = "10 MB",
                retention: str = "7 days",
                format_string: Optional[str] = None) -> None:
    """
    设置日志配置
    
    Args:
        log_level: 日志级别
        log_file: 日志文件名
        log_dir: 日志目录
        rotation: 日志轮转大小
        retention: 日志保留时间
        format_string: 自定义格式字符串
    """
    # 移除默认处理器
    logger.remove()
    
    # 默认格式
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    
    # 控制台输出
    logger.add(
        sys.stdout,
        level=log_level,
        format=format_string,
        colorize=True
    )
    
    # 文件输出
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"chart_matching_{timestamp}.log"
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_path / log_file,
        level=log_level,
        format=format_string,
        rotation=rotation,
        retention=retention,
        encoding="utf-8"
    )
    
    # 错误日志单独文件
    error_log_file = log_file.replace('.log', '_error.log')
    logger.add(
        log_path / error_log_file,
        level="ERROR",
        format=format_string,
        rotation=rotation,
        retention=retention,
        encoding="utf-8"
    )
    
    logger.info(f"日志系统初始化完成，日志目录: {log_path}")
    logger.info(f"日志级别: {log_level}")
    logger.info(f"主日志文件: {log_file}")
    logger.info(f"错误日志文件: {error_log_file}")


def get_logger(name: str = None) -> logger:
    """
    获取日志器实例
    
    Args:
        name: 日志器名称
        
    Returns:
        日志器实例
    """
    if name:
        return logger.bind(name=name)
    return logger


class LoggerContext:
    """日志上下文管理器"""
    
    def __init__(self, 
                 name: str,
                 level: str = "INFO",
                 extra_info: Optional[dict] = None):
        """
        初始化日志上下文
        
        Args:
            name: 上下文名称
            level: 日志级别
            extra_info: 额外信息
        """
        self.name = name
        self.level = level
        self.extra_info = extra_info or {}
        self.logger = get_logger(name)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(self.level, f"开始执行: {self.name}")
        if self.extra_info:
            self.logger.log(self.level, f"额外信息: {self.extra_info}")
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.log(self.level, f"完成执行: {self.name}，耗时: {duration:.2f}s")
        else:
            self.logger.error(f"执行失败: {self.name}，耗时: {duration:.2f}s，错误: {exc_val}")
        
        return False  # 不抑制异常


class PerformanceLogger:
    """性能日志记录器"""
    
    def __init__(self, name: str = "performance"):
        self.logger = get_logger(name)
        self.metrics = {}
    
    def start_timer(self, operation: str):
        """开始计时"""
        self.metrics[operation] = {
            'start_time': datetime.now(),
            'end_time': None,
            'duration': None
        }
        self.logger.debug(f"开始计时: {operation}")
    
    def end_timer(self, operation: str):
        """结束计时"""
        if operation not in self.metrics:
            self.logger.warning(f"未找到计时器: {operation}")
            return
        
        self.metrics[operation]['end_time'] = datetime.now()
        duration = (self.metrics[operation]['end_time'] - 
                   self.metrics[operation]['start_time']).total_seconds()
        self.metrics[operation]['duration'] = duration
        
        self.logger.info(f"操作完成: {operation}，耗时: {duration:.3f}s")
        return duration
    
    def log_memory_usage(self, operation: str = "current"):
        """记录内存使用情况"""
        try:
            import psutil
            import torch
            
            process = psutil.Process()
            cpu_memory = process.memory_info().rss / 1024**2  # MB
            
            memory_info = {
                'operation': operation,
                'cpu_memory_mb': cpu_memory
            }
            
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
                gpu_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
                memory_info.update({
                    'gpu_memory_mb': gpu_memory,
                    'gpu_reserved_mb': gpu_reserved
                })
            
            self.logger.info(f"内存使用情况 [{operation}]: {memory_info}")
            return memory_info
            
        except ImportError:
            self.logger.warning("psutil未安装，无法获取内存信息")
            return None
    
    def get_summary(self) -> dict:
        """获取性能摘要"""
        summary = {
            'total_operations': len(self.metrics),
            'operations': {}
        }
        
        total_time = 0
        for op, metrics in self.metrics.items():
            if metrics['duration'] is not None:
                summary['operations'][op] = {
                    'duration': metrics['duration'],
                    'start_time': metrics['start_time'].isoformat(),
                    'end_time': metrics['end_time'].isoformat()
                }
                total_time += metrics['duration']
        
        summary['total_time'] = total_time
        return summary
    
    def reset(self):
        """重置性能指标"""
        self.metrics.clear()
        self.logger.info("性能指标已重置")


def log_function_call(func):
    """函数调用日志装饰器"""
    def wrapper(*args, **kwargs):
        func_logger = get_logger(func.__module__)
        func_name = f"{func.__module__}.{func.__name__}"
        
        # 记录函数调用
        func_logger.debug(f"调用函数: {func_name}")
        func_logger.debug(f"参数: args={args}, kwargs={kwargs}")
        
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            func_logger.debug(f"函数完成: {func_name}，耗时: {duration:.3f}s")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            func_logger.error(f"函数异常: {func_name}，耗时: {duration:.3f}s，错误: {e}")
            raise
    
    return wrapper


def log_exception(func):
    """异常日志装饰器"""
    def wrapper(*args, **kwargs):
        func_logger = get_logger(func.__module__)
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            func_logger.exception(f"函数 {func.__name__} 发生异常: {e}")
            raise
    
    return wrapper