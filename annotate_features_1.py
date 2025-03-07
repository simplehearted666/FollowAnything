import math
import os
import random
import sys
import warnings
import traceback
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import argparse
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from tqdm import tqdm

# DINO模块导入
from DINO.dino_wrapper import get_dino_pixel_wise_features_model, preprocess_frame
import threading
import time

# ----------------------- 全局状态管理 -----------------------
class GlobalState:
    def __init__(self):
        self.exit_flag = False
        self.click_coords = (-1, -1)
        self.current_mode = 'auto'

global_state = GlobalState()
# ----------------------- 美观化控制台输出 -----------------------
class ConsoleStyle:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(title):
    print(f"\n{ConsoleStyle.BOLD}{ConsoleStyle.OKBLUE}=== {title.upper()} ==={ConsoleStyle.ENDC}")

def print_progress(desc, current, total):
    progress = f"{current}/{total}"
    print(f"{ConsoleStyle.OKGREEN}[PROGRESS]{ConsoleStyle.ENDC} {desc.ljust(40)} {progress.rjust(10)}")

def print_success(msg):
    print(f"{ConsoleStyle.OKGREEN}✓ {msg}{ConsoleStyle.ENDC}")

def print_warning(msg):
    print(f"{ConsoleStyle.WARNING}⚠ {msg}{ConsoleStyle.ENDC}")

def print_error(msg):
    print(f"{ConsoleStyle.FAIL}✗ {msg}{ConsoleStyle.ENDC}")

# ----------------------- 可视化样式配置 -----------------------
def configure_visuals():
    """兼容不同Matplotlib版本的样式设置"""
    available_styles = plt.style.available
    preferred_styles = [
        'seaborn-v0_8',    # Matplotlib >=3.6
        'seaborn',         # 旧版本
        'ggplot',          # 备选方案
        'default'
    ]
    
    # 寻找第一个可用的样式
    selected_style = next((s for s in preferred_styles if s in available_styles), 'default')
    
    try:
        plt.style.use(selected_style)
        # 如果使用seaborn相关样式，建议安装seaborn包
        if 'seaborn' in selected_style and not hasattr(plt, 'seaborn'):
            print_warning("For better visualization, install seaborn package: pip install seaborn")
    except Exception as e:
        print_warning(f"Style {selected_style} not available: {str(e)}")
        plt.style.use('default')

    # 通用参数设置
    matplotlib.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'figure.dpi': 150,
        'figure.figsize': (12, 6),
        'image.cmap': 'viridis'
    })
# ----------------------- 命令行参数解析 -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description='自动特征标注工具', add_help=False)
    
    # 必需参数
    parser.add_argument('--mode', required=True, 
                      choices=['click', 'text', 'auto'],
                      help='标注模式: click-点击/text-文本/auto-自动标注')
    
    # 路径参数
    parser.add_argument('--path_to_images', required=True,
                      help='图像序列目录路径（如uav0000076_00241_s）')
    parser.add_argument('--annotations_dir', required=('auto' in sys.argv),
                      help='标注文件目录（仅auto模式需要）')
    parser.add_argument('--queries_dir', default='./queries',
                      help='特征存储目录')
    
    # 处理参数
    parser.add_argument('--sample_size', type=int, default=5,
                      help='每序列采样帧数')
    parser.add_argument('--min_bbox_size', type=int, default=40,
                      help='有效标注的最小像素尺寸')
    
    # 硬件参数
    parser.add_argument('--cpu', action='store_true',  # 确保存在该定义
                      help='强制使用CPU')
    parser.add_argument('--use_16bit', action='store_true',
                      help='启用16位精度模式')
    
    # 可视化参数
    parser.add_argument('--plot_results', action='store_true',
                      help='显示可视化热力图')
    
    # 模型参数
    parser.add_argument('--desired_height', default=480, type=int, 
                   help='提高输入分辨率')
    parser.add_argument('--desired_width', type=int, default=640,
                      help='输入图像宽度')
    parser.add_argument('--dino_strides', default=4, type=int, 
                        help='较小步长保留更多空间信息')
    return parser.parse_args()

# ----------------------- 设备管理 -----------------------
class DeviceManager:
    def __init__(self, use_cpu=False):
        self.device = torch.device("cuda" if not use_cpu and torch.cuda.is_available() else "cpu")
        self._configure_device()
    
    def _configure_device(self):
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
    
    def move_to_device(self, tensor, non_blocking=True):
        return tensor.to(self.device, non_blocking=non_blocking)

# ----------------------- 特征提取器 -----------------------
class FeatureExtractor:
    def __init__(self, args, device_manager):
        self.args = args
        self.dm = device_manager
        self._init_models()
    
    def _init_models(self):
        """Initialize DINO model with safety checks"""
        try:
            model_config = {
                'desired_height': self.args.desired_height,
                'desired_width': self.args.desired_width,
                'dino_strides': self.args.dino_strides,
                'use_16bit': self.args.use_16bit
            }
            self.dino_model = get_dino_pixel_wise_features_model(model_config, self.dm.device)
            torch.cuda.empty_cache()
        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def extract_features(self, image):
        """Main feature extraction method with memory management"""
        try:
            # Preprocessing with size validation
            processed = preprocess_frame(image, {
                'desired_height': self.args.desired_height,
                'desired_width': self.args.desired_width,
                'use_16bit': self.args.use_16bit  # 必须包含
            })
            # 添加批次维度
            if processed.dim() == 3:
                processed = processed.unsqueeze(0)  # [1, C, H, W]
            # Device transfer with optimized settings
            if self.args.use_16bit:
                processed = processed.half()
            processed = self.dm.move_to_device(processed)
            
            # Dynamic memory management
            with torch.cuda.amp.autocast(enabled=self.args.use_16bit):
                if self._should_use_safe_mode():
                    with torch.no_grad():
                        features = self.dino_model(processed)
                else:
                    features = self.dino_model(processed)
            
            # 再次验证输出维度
            if features.dim() == 3:
                features = features.unsqueeze(0)  # [1,D,Hf,Wf]
            
            # 在通道维度归一化 (dim=1)
            normalized_features = F.normalize(features, dim=1)
        
            return normalized_features.detach()  # 返回4D张量
        
        except RuntimeError as e:
            self._handle_memory_error(e)
    
    def _should_use_safe_mode(self):
        """Determine if should use memory-safe mode"""
        if torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            return free_mem < 1.5  # 1.5GB threshold
        return False
    

    def _handle_memory_error(self, error):
        """用户友好的内存错误提示"""
        print_error("Insufficient GPU memory! Try these solutions:")
        print(f" 1. Enable {ConsoleStyle.UNDERLINE}--use_16bit{ConsoleStyle.ENDC} for memory efficiency")
        print(f" 2. Reduce input size (current: {self.args.desired_height}x{self.args.desired_width})")
        print(f" 3. Use smaller sample size (current: {self.args.sample_size})")
        sys.exit(1)


# ----------------------- 新增特征存储类 -----------------------
class FeatureStorage:
    def __init__(self):
        self.feature_dict = defaultdict(list)
    
    def add_feature(self, class_label: int, feature: torch.Tensor):
        """添加特征到对应类别"""
        self.feature_dict[class_label].append(feature.squeeze(0).cpu().detach())
    
    def save_features(self, output_dir: Union[str, Path]):
        """保存所有特征到磁盘"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for label, features in self.feature_dict.items():
            save_path = output_dir / f"feat{label}.pt"
            torch.save(features, save_path)
            print_success(f"保存类别 {label} 特征至: {ConsoleStyle.UNDERLINE}{save_path}{ConsoleStyle.ENDC}")

# ----------------------- 自动标注处理器 -----------------------
class AutoAnnotator:
    def __init__(self, args, feature_storage: FeatureStorage):
        self.args = args
        self.feature_storage = feature_storage
        self.device = torch.device("cuda" if not args.cpu and torch.cuda.is_available() else "cpu")
        self.model = self._init_dino_model()
        if args.use_16bit:
            self.model = self.model.half()
        torch.backends.cudnn.benchmark = True
    def _save_auto_features(self, features, bbox):
        """自动模式下的特征保存逻辑"""
        x, y, w, h = map(int, bbox)
        
        # 提取ROI区域特征
        roi_features = features[..., y:y+h, x:x+w]
        avg_feature = roi_features.mean(dim=(2,3))  # 空间维度平均
        
        # 假设自动标注为类别0
        self.feature_storage.add_feature(class_label=0, feature=avg_feature)
        
    def _init_dino_model(self):
        cfg = {
            'desired_height': self.args.desired_height,
            'desired_width': self.args.desired_width,
            'dino_strides': self.args.dino_strides,
            'use_16bit': self.args.use_16bit
        }
        self.model = get_dino_pixel_wise_features_model(cfg, self.device)
        if self.args.use_16bit:
            self.model = self.model.half()
        return self.model
    
    def _resolve_windows_path(self, path):
        """处理Windows长路径问题"""
        path_str = str(path.absolute())
        if os.name == 'nt' and len(path_str) > 260:
            return Path(f"\\\\?\\{path_str}")
        return path
    
    def process_sequence(self, seq_path):
        seq_path = SafePath.long(Path(seq_path))
        anno_path = SafePath.long(
            Path(self.args.annotations_dir) / f"{seq_path.name}.txt"
        )
        
        # 加载标注数据
        annotations = self._load_visdrone_annotations(anno_path)
        
        # 处理图像帧
        frame_files = sorted(seq_path.glob("img*.jpg"), 
                           key=lambda x: int(x.stem[3:]))
        
        with tqdm(frame_files[:self.args.sample_size], desc="处理进度") as pbar:
            for frame_path in pbar:
                frame_idx = int(frame_path.stem[3:])
                bbox = annotations.get(frame_idx)
                if not bbox:
                    print_warning(f"跳过无标注帧: {frame_path.name}")
                    continue
                
                try:
                    self.process_frame(frame_path, bbox)
                except Exception as e:
                    print_error(f"处理失败: {frame_path.name} - {str(e)}")
                
                # 实时保存进度
                if pbar.n % 5 == 0:
                    self.feature_storage.save_features(self.args.queries_dir)
    
    def _load_visdrone_annotations(self, annotation_path):
        """修正索引偏移问题"""
        annotations = {}
        with open(annotation_path, 'r') as f:
            for line_idx, line in enumerate(f):
                parts = list(map(float, line.strip().split(',')))
                if len(parts) < 4:
                    continue
                # 行号从0开始，对应img000001.jpg的索引1
                annotations[line_idx + 1] = parts[:4]  
        return annotations
    
    def process_frame(self, frame_path, bbox):
        """处理单帧的核心方法（兼容VisDrone标注格式）"""
        try:
            # 加载图像
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise ValueError(f"无法读取图像文件: {frame_path}")

            # 预处理
            processed = preprocess_frame(frame, {
                'desired_height': self.args.desired_height,
                'desired_width': self.args.desired_width,
                'use_16bit': self.args.use_16bit
            }).to(self.device)

            # 特征提取
            with torch.no_grad():
                features = self.model(processed)
            
            # 验证特征维度
            if features.dim() != 4:
                raise ValueError(f"无效特征维度: {features.shape}")

            # 坐标转换（三步转换）
            orig_h, orig_w = frame.shape[:2]
            prep_h, prep_w = self.args.desired_height, self.args.desired_width
            feat_h, feat_w = features.shape[2], features.shape[3]

            # Step 1: 原始 -> 预处理坐标
            x_orig, y_orig, w_orig, h_orig = map(int, bbox)
            x_prep = int(x_orig * (prep_w / orig_w))
            y_prep = int(y_orig * (prep_h / orig_h))
            w_prep = int(w_orig * (prep_w / orig_w))
            h_prep = int(h_orig * (prep_h / orig_h))

            # Step 2: 预处理 -> 特征图坐标
            fx = int(x_prep * (feat_w / prep_w))
            fy = int(y_prep * (feat_h / prep_h))
            fw = max(1, int(w_prep * (feat_w / prep_w)))
            fh = max(1, int(h_prep * (feat_h / prep_h)))

            # Step 3: 安全边界校验
            fx = min(max(fx, 0), feat_w - 1)
            fy = min(max(fy, 0), feat_h - 1)
            fw = min(fw, feat_w - fx)
            fh = min(fh, feat_h - fy)

            # ROI提取防御性编程
            y_start, y_end = max(0, fy), min(feat_h, fy + fh)
            x_start, x_end = max(0, fx), min(feat_w, fx + fw)
            if (y_end - y_start) <= 0 or (x_end - x_start) <= 0:
                raise ValueError(f"无效ROI区域: y[{y_start}:{y_end}] x[{x_start}:{x_end}]")

            roi_features = features[:, :, y_start:y_end, x_start:x_end]
            assert roi_features.size(2) == (y_end - y_start), "高度维度不匹配"
            assert roi_features.size(3) == (x_end - x_start), "宽度维度不匹配"

            # 特征增强与聚合
            enhanced = self._enhance_features(roi_features)
            aggregated = self._aggregate_roi(enhanced)
            assert aggregated.shape[1] == 6144, f"特征维度应为6144，实际为{aggregated.shape[1]}"
            self.feature_storage.add_feature(0, aggregated)

            # 可视化处理
            if self.args.plot_results:
                self._show_annotation(
                    original_frame=frame,  # 关键修复：参数名对齐
                    original_bbox=(x_orig, y_orig, w_orig, h_orig),
                    prep_bbox=(x_prep, y_prep, w_prep, h_prep),
                    feature_map=features,  # 参数名同步为 feature_map
                    feature_bbox=(fx, fy, fw, fh),
                    frame_path=frame_path,
                    show_plot=self.args.plot_results  # 新增参数传递
                )
        except Exception as e:
            raise RuntimeError(f"帧处理失败 [{frame_path.name}]: {str(e)}")

    def _enhance_features(self, features):
        """维度安全的特征增强"""
        assert features.dim() == 4, "输入特征必须是4D张量"
        channel_weights = torch.mean(features, dim=(-2, -1), keepdim=True)
        return features * channel_weights

    def _aggregate_roi(self, features):
        """单尺度4x4池化"""
        # 使用固定4x4网格池化
        pooled_feat = F.adaptive_avg_pool2d(features, 4)
        
        # 验证输出维度（调试用）
        assert pooled_feat.shape[2] == 4 and pooled_feat.shape[3] == 4, \
            f"池化后维度异常: {pooled_feat.shape}"
        
        # 展平并归一化
        flattened = pooled_feat.flatten(start_dim=1)
        return F.normalize(flattened, p=2, dim=1)
    
    def _get_feature_scale_factor(self, orig_h, orig_w):
        """计算特征图与原图的尺寸比例"""
        feat_h = self.args.desired_height // self.args.dino_strides
        feat_w = self.args.desired_width // self.args.dino_strides
        return feat_h / orig_h, feat_w / orig_w

    def _bbox_to_feature_coords(self, bbox, orig_h, orig_w):
        """精确的特征图坐标转换（直接从特征张量获取尺寸）"""
        # 从特征张量获取真实尺寸
        _, _, feat_h, feat_w = self._get_feature_shape()
            
        x, y, w, h = bbox
            
        # 计算缩放比例（使用浮点计算）
        scale_w = feat_w / orig_w
        scale_h = feat_h / orig_h
            
        # 应用缩放并四舍五入
        fx = round(x * scale_w)
        fy = round(y * scale_h)
        fw = round(w * scale_w)
        fh = round(h * scale_h)
            
        # 边界安全限制
        fx = max(0, min(fx, feat_w - 1))
        fy = max(0, min(fy, feat_h - 1))
        fw = max(1, min(fw, feat_w - fx))
        fh = max(1, min(fh, feat_h - fy))
            
        print(f"[坐标转换] 预处理尺寸:{orig_w}x{orig_h} 特征图:{feat_w}x{feat_h}")
        print(f"预处理坐标:({x},{y},{w},{h}) → 转换后:({fx},{fy},{fw},{fh})")
            
        return ((int(fx), int(fy), int(fw), int(fh)), feat_w, feat_h)


    def _get_feature_shape(self):
        """获取模型实际输出尺寸（关键新增方法）"""
        # 通过虚拟输入获取真实特征图尺寸
        dummy_input = torch.randn(1, 3, self.args.desired_height, self.args.desired_width)
        with torch.no_grad():
            features = self.model(dummy_input.to(self.device))
        return features.shape  # [B, C, H, W]

    def _show_annotation(self, original_frame, original_bbox, prep_bbox, feature_map, feature_bbox, frame_path, show_plot=False):
        """修复可视化布局错误的热力图生成函数"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            
            # 动态选择后端
            plt.switch_backend('TkAgg' if self.args.plot_results else 'Agg')
            
            # ================= 创建专业布局 =================
            fig = plt.figure(figsize=(24, 8), dpi=100)
            gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.2])  # 修复拼写错误

            # ================= 原始图像子图 =================
            ax0 = fig.add_subplot(gs[0])
            viz_orig = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            cv2.rectangle(viz_orig, 
                        (original_bbox[0], original_bbox[1]),
                        (original_bbox[0]+original_bbox[2], original_bbox[1]+original_bbox[3]),
                        (0,255,0), 2)
            ax0.imshow(viz_orig)
            ax0.set_title("Original Image\n(Ground Truth BBox)", fontsize=12, pad=10)

            # ================= 预处理图像子图 =================
            ax1 = fig.add_subplot(gs[1])
            prep_frame = cv2.resize(viz_orig, 
                                (self.args.desired_width, self.args.desired_height),
                                interpolation=cv2.INTER_AREA)
            cv2.rectangle(prep_frame,
                        (prep_bbox[0], prep_bbox[1]),
                        (prep_bbox[0]+prep_bbox[2], prep_bbox[1]+prep_bbox[3]),
                        (255,0,0), 2)
            ax1.imshow(prep_frame)
            ax1.set_title("Preprocessed Image\n(Feature Sampling Area)", fontsize=12, pad=10)

            # ================= 热力图子图 =================
            ax2 = fig.add_subplot(gs[2])
            
            # 安全提取目标特征
            y_start = max(0, min(feature_bbox[1], feature_map.shape[2]-1))
            y_end = min(feature_map.shape[2], y_start + feature_bbox[3])
            x_start = max(0, min(feature_bbox[0], feature_map.shape[3]-1))
            x_end = min(feature_map.shape[3], x_start + feature_bbox[2])
            
            target_roi = feature_map[0, :, y_start:y_end, x_start:x_end]
            target_feat = target_roi.mean(dim=(1,2)).unsqueeze(0)  # 保持四维结构
            
            # 计算相似度
            similarity = F.cosine_similarity(feature_map, target_feat[..., None, None])
            similarity_norm = (similarity - similarity.min()) / (similarity.max() - similarity.min() + 1e-6)
            
            # 生成热力图
            im = ax2.imshow(similarity_norm[0].cpu().numpy(), cmap='jet', vmin=0, vmax=1)
            
            # 添加专业颜色条
            cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label('Feature Activation Level', rotation=270, labelpad=20, fontsize=10)
            
            # 添加统计信息
            stats_text = (f"Max: {similarity_norm.max().item():.2f}\n"
                        f"Min: {similarity_norm.min().item():.2f}\n"
                        f"Mean: {similarity_norm.mean().item():.2f}")
            ax2.text(0.98, 0.02, stats_text,
                    transform=ax2.transAxes,
                    color='white',
                    fontsize=10,
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(facecolor='black', alpha=0.5))

            ax2.set_title("Feature Activation Heatmap\n(with Confidence Metrics)", fontsize=12, pad=10)

            # ================= 布局优化 =================
            plt.tight_layout(pad=3.0)
            plt.subplots_adjust(wspace=0.1)

            # ================= 保存与显示 =================
            save_path = Path(f"./vis_results/{frame_path.stem}.jpg")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=120)
            
            if show_plot:
                plt.show(block=True)  # 阻塞模式显示
            else:
                plt.close()
                        
            print_success(f"可视化结果保存至: {save_path}")

        except Exception as e:
            print_error(f"可视化异常: {str(e)}")
            if 'target_roi' in locals():
                print_warning(f"ROI尺寸: {target_roi.shape if target_roi else 'None'}")
            if 'similarity' in locals():
                print_warning(f"相似度矩阵尺寸: {similarity.shape if similarity else 'None'}")




# ----------------------- 辅助工具类 -----------------------
class SafePath:
    @staticmethod
    def long(path):
        """Handle Windows long path issues"""
        path = Path(path).resolve()
        if os.name == 'nt' and len(str(path)) > 260:
            return Path(f"\\\\?\\{path}")
        return path

# ----------------------- 主程序 -----------------------
def main(args):
    configure_visuals()
    
    try:
        # 初始化特征存储器
        feature_storage = FeatureStorage()
        
        print_header("Initializing System Components")
        device_mgr = DeviceManager(args.cpu)
        print_success(f"Hardware Initialized: Using {device_mgr.device.type.upper()}")

        print_header("Creating Output Structure")
        output_dir = SafePath.long(Path(args.queries_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        print_success(f"Output Directory Ready: {output_dir}")

        print_header("Loading AI Models")
        with tqdm(total=2, desc="Model Loading", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            feature_extractor = FeatureExtractor(args, device_mgr)
            pbar.update(1)
            pbar.update(1)
        print_success("DINO Model Loaded")

        print_header("Processing Image Sequence")
        if args.mode == 'auto':
            # 构建序列标注文件路径
            seq_name = Path(args.path_to_images).name  # 获取序列名称如uav0000076_00241_s
            anno_file = Path(args.annotations_dir) / f"{seq_name}.txt"
            
            if not anno_file.exists():
                raise FileNotFoundError(f"Annotation file missing: {anno_file}")

            # 加载全部标注数据
            with open(anno_file, 'r') as f:
                all_annotations = [line.strip().split(',') for line in f.readlines()]
            
            # 获取并排序图像文件
            seq_path = SafePath.long(Path(args.path_to_images))
            frame_files = sorted(seq_path.glob("img*.jpg"), 
                               key=lambda x: int(x.stem[3:]))  # 按数字序号排序
            
            # 随机选择帧索引（保持时序对应）
            total_frames = min(len(frame_files), len(all_annotations))
            selected_indices = random.sample(range(total_frames), min(args.sample_size, total_frames))
            
            annotator = AutoAnnotator(args, feature_storage)
            
            with tqdm(selected_indices, desc="Processing", unit="frame", 
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
                for idx in pbar:
                    frame_path = frame_files[idx]
                    try:
                        # 解析标注数据 [x, y, w, h, ...]
                        bbox = list(map(float, all_annotations[idx][:4]))
                        
                        # 处理单帧
                        annotator.process_frame(frame_path, bbox)
                        
                        # 实时保存
                        if pbar.n % 5 == 0:
                            feature_storage.save_features(output_dir)
                            
                    except Exception as e:
                        print_warning(f"Skipped {frame_path.name}: {str(e)}")
                        continue
            
            # 最终保存
            feature_storage.save_features(output_dir)
            print_success(f"Processed {len(selected_indices)} frames. Features saved to: {output_dir}")

    except Exception as e:
        print_error(f"Critical Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 启动主程序
    main(args)