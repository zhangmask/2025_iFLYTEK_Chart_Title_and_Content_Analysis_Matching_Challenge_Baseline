import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from PIL import Image
import torch
from loguru import logger
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')


class Visualizer:
    """可视化工具类"""
    
    def __init__(self, 
                 output_dir: str = "visualizations",
                 figsize: Tuple[int, int] = (10, 8),
                 dpi: int = 300):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
            figsize: 默认图像大小
            dpi: 图像分辨率
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.figsize = figsize
        self.dpi = dpi
        
        # 颜色配置
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        logger.info(f"可视化器初始化完成，输出目录: {output_dir}")
    
    def plot_training_history(self,
                             history: Dict[str, List[float]],
                             title: str = "训练历史",
                             save_name: Optional[str] = None,
                             show_validation: bool = True) -> str:
        """
        绘制训练历史
        
        Args:
            history: 训练历史字典
            title: 图表标题
            save_name: 保存文件名
            show_validation: 是否显示验证指标
            
        Returns:
            保存路径
        """
        if not history:
            logger.warning("训练历史为空")
            return ""
        
        # 分离训练和验证指标
        train_metrics = {}
        val_metrics = {}
        
        for key, values in history.items():
            if key.startswith('val_'):
                val_metrics[key[4:]] = values  # 移除'val_'前缀
            else:
                train_metrics[key] = values
        
        # 确定子图数量
        n_metrics = len(train_metrics)
        if n_metrics == 0:
            logger.warning("没有找到训练指标")
            return ""
        
        # 计算子图布局
        cols = min(2, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(self.figsize[0] * cols, self.figsize[1] * rows))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        # 绘制每个指标
        for i, (metric_name, train_values) in enumerate(train_metrics.items()):
            ax = axes[i]
            epochs = range(1, len(train_values) + 1)
            
            # 绘制训练指标
            ax.plot(epochs, train_values, 
                   label=f'训练 {metric_name}', 
                   color=self.colors['primary'],
                   linewidth=2, marker='o', markersize=4)
            
            # 绘制验证指标（如果存在）
            if show_validation and metric_name in val_metrics:
                val_values = val_metrics[metric_name]
                if len(val_values) == len(train_values):
                    ax.plot(epochs, val_values, 
                           label=f'验证 {metric_name}', 
                           color=self.colors['secondary'],
                           linewidth=2, marker='s', markersize=4)
            
            ax.set_title(f'{metric_name.replace("_", " ").title()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图像
        if save_name is None:
            save_name = "training_history.png"
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"训练历史图已保存: {save_path}")
        return str(save_path)
    
    def plot_confusion_matrix(self,
                             cm: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             title: str = "混淆矩阵",
                             save_name: Optional[str] = None,
                             normalize: bool = False) -> str:
        """
        绘制混淆矩阵
        
        Args:
            cm: 混淆矩阵
            class_names: 类别名称
            title: 图表标题
            save_name: 保存文件名
            normalize: 是否归一化
            
        Returns:
            保存路径
        """
        if cm.size == 0:
            logger.warning("混淆矩阵为空")
            return ""
        
        plt.figure(figsize=self.figsize)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': '比例' if normalize else '数量'})
        
        plt.title(title)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        
        # 保存图像
        if save_name is None:
            save_name = "confusion_matrix.png"
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"混淆矩阵已保存: {save_path}")
        return str(save_path)
    
    def plot_similarity_distribution(self,
                                   similarities: np.ndarray,
                                   labels: Optional[np.ndarray] = None,
                                   title: str = "相似度分布",
                                   save_name: Optional[str] = None) -> str:
        """
        绘制相似度分布
        
        Args:
            similarities: 相似度数组
            labels: 标签数组（用于区分正负样本）
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            保存路径
        """
        plt.figure(figsize=self.figsize)
        
        if labels is not None:
            # 分别绘制正负样本的相似度分布
            positive_sims = similarities[labels == 1]
            negative_sims = similarities[labels == 0]
            
            plt.hist(positive_sims, bins=50, alpha=0.7, 
                    label='正样本', color=self.colors['success'], density=True)
            plt.hist(negative_sims, bins=50, alpha=0.7, 
                    label='负样本', color=self.colors['danger'], density=True)
            plt.legend()
        else:
            plt.hist(similarities, bins=50, alpha=0.7, 
                    color=self.colors['primary'], density=True)
        
        plt.title(title)
        plt.xlabel('相似度')
        plt.ylabel('密度')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图像
        if save_name is None:
            save_name = "similarity_distribution.png"
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"相似度分布图已保存: {save_path}")
        return str(save_path)
    
    def plot_feature_tsne(self,
                         features: np.ndarray,
                         labels: Optional[np.ndarray] = None,
                         title: str = "特征t-SNE可视化",
                         save_name: Optional[str] = None,
                         perplexity: int = 30) -> str:
        """
        绘制特征的t-SNE可视化
        
        Args:
            features: 特征数组
            labels: 标签数组
            title: 图表标题
            save_name: 保存文件名
            perplexity: t-SNE参数
            
        Returns:
            保存路径
        """
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            logger.error("需要安装scikit-learn来使用t-SNE")
            return ""
        
        if features.shape[0] < 4:  # t-SNE需要至少4个样本
            logger.warning("样本数量太少，无法进行t-SNE可视化")
            return ""
        
        # 执行t-SNE
        tsne = TSNE(n_components=2, perplexity=min(perplexity, features.shape[0] - 1), 
                   random_state=42)
        features_2d = tsne.fit_transform(features)
        
        plt.figure(figsize=self.figsize)
        
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                           c=[colors[i]], label=f'类别 {label}', alpha=0.7)
            plt.legend()
        else:
            plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                       c=self.colors['primary'], alpha=0.7)
        
        plt.title(title)
        plt.xlabel('t-SNE 维度 1')
        plt.ylabel('t-SNE 维度 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图像
        if save_name is None:
            save_name = "feature_tsne.png"
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"t-SNE可视化已保存: {save_path}")
        return str(save_path)
    
    def plot_model_comparison(self,
                             model_metrics: Dict[str, Dict[str, float]],
                             metrics_to_plot: Optional[List[str]] = None,
                             title: str = "模型性能比较",
                             save_name: Optional[str] = None) -> str:
        """
        绘制模型性能比较图
        
        Args:
            model_metrics: 模型指标字典
            metrics_to_plot: 要绘制的指标列表
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            保存路径
        """
        if not model_metrics:
            logger.warning("模型指标为空")
            return ""
        
        # 获取所有指标
        all_metrics = set()
        for metrics in model_metrics.values():
            all_metrics.update(metrics.keys())
        
        if metrics_to_plot is None:
            metrics_to_plot = list(all_metrics)[:6]  # 最多显示6个指标
        
        # 准备数据
        models = list(model_metrics.keys())
        n_metrics = len(metrics_to_plot)
        
        if n_metrics == 0:
            logger.warning("没有指标可绘制")
            return ""
        
        # 计算子图布局
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(self.figsize[0] * cols, self.figsize[1] * rows))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        # 绘制每个指标
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            values = [model_metrics[model].get(metric, 0) for model in models]
            colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
            
            bars = ax.bar(models, values, color=colors, alpha=0.8)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图像
        if save_name is None:
            save_name = "model_comparison.png"
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"模型比较图已保存: {save_path}")
        return str(save_path)
    
    def plot_prediction_examples(self,
                               images: List[Union[str, np.ndarray, Image.Image]],
                               texts: List[str],
                               predictions: List[str],
                               confidences: List[float],
                               ground_truths: Optional[List[str]] = None,
                               title: str = "预测示例",
                               save_name: Optional[str] = None,
                               max_examples: int = 8) -> str:
        """
        绘制预测示例
        
        Args:
            images: 图像列表
            texts: 文本列表
            predictions: 预测结果列表
            confidences: 置信度列表
            ground_truths: 真实标签列表
            title: 图表标题
            save_name: 保存文件名
            max_examples: 最大示例数
            
        Returns:
            保存路径
        """
        n_examples = min(len(images), max_examples)
        if n_examples == 0:
            logger.warning("没有示例可显示")
            return ""
        
        # 计算子图布局
        cols = min(4, n_examples)
        rows = (n_examples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(self.figsize[0] * cols, self.figsize[1] * rows))
        if n_examples == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        for i in range(n_examples):
            ax = axes[i]
            
            # 加载和显示图像
            if isinstance(images[i], str):
                # 文件路径
                try:
                    img = Image.open(images[i])
                    ax.imshow(img)
                except Exception as e:
                    logger.warning(f"无法加载图像 {images[i]}: {e}")
                    ax.text(0.5, 0.5, '图像加载失败', ha='center', va='center', transform=ax.transAxes)
            elif isinstance(images[i], np.ndarray):
                ax.imshow(images[i])
            elif isinstance(images[i], Image.Image):
                ax.imshow(images[i])
            
            ax.axis('off')
            
            # 添加文本信息
            info_text = f"文本: {texts[i][:30]}...\n"
            info_text += f"预测: {predictions[i]}\n"
            info_text += f"置信度: {confidences[i]:.3f}"
            
            if ground_truths:
                info_text += f"\n真实: {ground_truths[i]}"
                # 根据预测是否正确设置颜色
                color = 'green' if predictions[i] == ground_truths[i] else 'red'
            else:
                color = 'blue'
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   color=color)
        
        # 隐藏多余的子图
        for i in range(n_examples, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图像
        if save_name is None:
            save_name = "prediction_examples.png"
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"预测示例图已保存: {save_path}")
        return str(save_path)
    
    def create_interactive_dashboard(self,
                                   training_history: Dict[str, List[float]],
                                   model_metrics: Dict[str, Dict[str, float]],
                                   save_name: Optional[str] = None) -> str:
        """
        创建交互式仪表板
        
        Args:
            training_history: 训练历史
            model_metrics: 模型指标
            save_name: 保存文件名
            
        Returns:
            保存路径
        """
        try:
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('训练历史', '模型比较', '指标分布', '性能雷达图'),
                specs=[[{"secondary_y": True}, {}],
                       [{}, {"type": "polar"}]]
            )
            
            # 1. 训练历史
            if training_history:
                for metric_name, values in training_history.items():
                    epochs = list(range(1, len(values) + 1))
                    fig.add_trace(
                        go.Scatter(x=epochs, y=values, name=metric_name, mode='lines+markers'),
                        row=1, col=1
                    )
            
            # 2. 模型比较
            if model_metrics:
                models = list(model_metrics.keys())
                accuracy_values = [metrics.get('accuracy', 0) for metrics in model_metrics.values()]
                
                fig.add_trace(
                    go.Bar(x=models, y=accuracy_values, name='准确率'),
                    row=1, col=2
                )
            
            # 3. 指标分布
            if model_metrics:
                all_accuracies = [metrics.get('accuracy', 0) for metrics in model_metrics.values()]
                fig.add_trace(
                    go.Histogram(x=all_accuracies, name='准确率分布'),
                    row=2, col=1
                )
            
            # 4. 性能雷达图
            if model_metrics and len(model_metrics) > 0:
                first_model = list(model_metrics.keys())[0]
                metrics = model_metrics[first_model]
                
                metric_names = list(metrics.keys())[:6]  # 最多6个指标
                metric_values = [metrics[name] for name in metric_names]
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=metric_values,
                        theta=metric_names,
                        fill='toself',
                        name=first_model
                    ),
                    row=2, col=2
                )
            
            # 更新布局
            fig.update_layout(
                title_text="模型训练与评估仪表板",
                showlegend=True,
                height=800
            )
            
            # 保存为HTML
            if save_name is None:
                save_name = "dashboard.html"
            
            save_path = self.output_dir / save_name
            fig.write_html(save_path)
            
            logger.info(f"交互式仪表板已保存: {save_path}")
            return str(save_path)
            
        except ImportError:
            logger.warning("需要安装plotly来创建交互式仪表板")
            return ""
        except Exception as e:
            logger.error(f"创建交互式仪表板失败: {e}")
            return ""
    
    def save_figure_to_base64(self, fig) -> str:
        """
        将matplotlib图像转换为base64字符串
        
        Args:
            fig: matplotlib图像对象
            
        Returns:
            base64编码的图像字符串
        """
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        
        return image_base64
    
    def create_report_html(self,
                          title: str,
                          sections: List[Dict[str, Any]],
                          save_name: Optional[str] = None) -> str:
        """
        创建HTML报告
        
        Args:
            title: 报告标题
            sections: 报告章节列表
            save_name: 保存文件名
            
        Returns:
            保存路径
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; text-align: center; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .image {{ text-align: center; margin: 20px 0; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
        """
        
        for section in sections:
            html_content += f'<div class="section">'
            html_content += f'<h2>{section.get("title", "未命名章节")}</h2>'
            
            if 'content' in section:
                html_content += f'<p>{section["content"]}</p>'
            
            if 'metrics' in section:
                html_content += '<div class="metric">'
                for key, value in section['metrics'].items():
                    html_content += f'<strong>{key}:</strong> {value}<br>'
                html_content += '</div>'
            
            if 'image_path' in section:
                html_content += f'<div class="image"><img src="{section["image_path"]}"></div>'
            
            if 'table' in section:
                html_content += '<table>'
                table_data = section['table']
                if 'headers' in table_data:
                    html_content += '<tr>'
                    for header in table_data['headers']:
                        html_content += f'<th>{header}</th>'
                    html_content += '</tr>'
                
                if 'rows' in table_data:
                    for row in table_data['rows']:
                        html_content += '<tr>'
                        for cell in row:
                            html_content += f'<td>{cell}</td>'
                        html_content += '</tr>'
                html_content += '</table>'
            
            html_content += '</div>'
        
        html_content += """
        </body>
        </html>
        """
        
        # 保存HTML文件
        if save_name is None:
            save_name = "report.html"
        
        save_path = self.output_dir / save_name
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML报告已保存: {save_path}")
        return str(save_path)