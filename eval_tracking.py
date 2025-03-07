import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from PIL import Image
from scipy.spatial.distance import cdist

plt.style.use('seaborn-v0_8-bright')
sns.set_palette("husl")
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans'})

class AdvancedTrackingEvaluator:
    def __init__(self, pred_dir, gt_dir, attr_path, img_size=(1344, 756)):
        """
        pred_dir: 预测标注目录
        gt_dir: 真实标注目录（含.txt文件）
        attr_path: 属性文件路径
        img_size: 图像尺寸 (width, height)
        """
        self.pred_dir = pred_dir
        self.gt_dir = gt_dir
        self.attr_path = attr_path
        self.img_size = img_size
        self.attribute_labels = [
            'IV', 'SV', 'OCC', 'FM', 'MB', 'ROT',
            'BC', 'LR', 'OV', 'CM', 'ARC', 'VC'
        ]
        
        # 加载数据
        self.pred_annos = self._load_pred_annotations()
        self.gt_annos = self._load_gt_annotations()
        self.attributes = self._load_attributes()
        
        # 存储评估结果
        self.metrics = {
            'global': {},
            'challenge_specific': {}
        }

    def _load_pred_annotations(self):
        """加载预测标注（每帧一个文件）"""
        annos = {}
        for fname in tqdm(os.listdir(self.pred_dir), desc="Loading predictions"):
            if not fname.endswith('.txt'):
                continue
            frame_id = int(fname.split('.')[0])
            with open(os.path.join(self.pred_dir, fname)) as f:
                annos[frame_id] = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue  # 跳过空行
                
                    # 分割并过滤空字段
                    parts = [x for x in line.split(',') if x.strip() != '']
                
                    # 字段数量验证
                    if len(parts) != 10:
                        print(f"警告：文件 {fname} 第 {line_num} 行字段数错误（应有10个，实际{len(parts)}）")
                        continue  # 跳过无效行或根据需求处理
                
                    try:
                        parts = list(map(float, parts))
                    except ValueError as e:
                        print(f"错误：文件 {fname} 第 {line_num} 行包含无效数据 '{line}'")
                        raise e
                
                    annos[frame_id].append({
                        'bbox': parts[2:6],  # x,y,w,h
                        'score': parts[6],
                        'category': int(parts[7]),
                        'truncation': parts[8],
                        'occlusion': parts[9]
                    })
        return annos
    def _load_gt_annotations(self):
        """加载真实标注（每个序列一个文件）"""
        annos = {}
        with open(os.path.join(self.gt_dir, "uav0000014_00667_s.txt")) as f:
            for frame_id, line in tqdm(enumerate(f), desc="Loading ground truth"):
                line = line.strip()
                if not line:
                    continue  # 跳过空行
            
                parts = [x for x in line.split(',') if x.strip() != '']
                if len(parts) != 4:
                    print(f"警告：真实标注第 {frame_id+1} 行字段数错误（应有4个，实际{len(parts)}）")
                    continue
            
                try:
                    parts = list(map(float, parts))
                except ValueError as e:
                    print(f"错误：真实标注第 {frame_id+1} 行包含无效数据 '{line}'")
                    raise e
            
                annos[frame_id+1] = [{
                    'bbox': parts[:4]  # x,y,w,h
                }]
        return annos

    def _load_attributes(self):
        """加载属性信息"""
        with open(self.attr_path) as f:
            line = f.read().strip()
            parts = line.split(',')
            if len(parts) != 12:
                raise ValueError(f"属性文件 {self.attr_path} 格式错误，应有12个字段")
            return np.array(list(map(int, parts)))

    def _calculate_iou(self, box1, box2):
        """计算交并比"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi = max(x1, x2)
        yi = max(y1, y2)
        xu = min(x1+w1, x2+w2)
        yu = min(y1+h1, y2+h2)
        
        inter_area = max(0, xu - xi) * max(0, yu - yi)
        union_area = w1*h1 + w2*h2 - inter_area
        return inter_area / union_area if union_area else 0

    def _evaluate_frame(self, frame_id):
        """评估单个帧"""
        gt_boxes = [anno['bbox'] for anno in self.gt_annos.get(frame_id, [])]
        pred_boxes = [anno['bbox'] for anno in self.pred_annos.get(frame_id, [])]
        scores = [anno['score'] for anno in self.pred_annos.get(frame_id, [])]
        
        # 按置信度排序
        sorted_indices = np.argsort(scores)[::-1]
        pred_boxes = [pred_boxes[i] for i in sorted_indices]
        
        # 匹配结果
        matches = []
        matched_gt = set()
        for i, pbox in enumerate(pred_boxes):
            best_iou = 0.0
            best_j = -1
            for j, gbox in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = self._calculate_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j != -1:
                matched_gt.add(best_j)
                matches.append((i, best_j, best_iou))
        
        return {
            'frame_id': frame_id,
            'matches': matches,
            'num_gt': len(gt_boxes),
            'num_pred': len(pred_boxes)
        }

    def evaluate(self, iou_thresholds=[0.5, 0.75], max_detections=[1, 10, 100, 500]):
        """执行综合评估"""
        all_results = []
        attr_mask = self.attributes.astype(bool)
        
        # 逐帧评估
        for frame_id in tqdm(self.gt_annos, desc="Evaluating frames"):
            result = self._evaluate_frame(frame_id)
            result['attributes'] = attr_mask[frame_id-1] if frame_id <= len(self.attributes) else []
            all_results.append(result)
        
        # 全局指标计算
        self._calculate_global_metrics(all_results, iou_thresholds, max_detections)
        
        # 挑战条件分析
        self._analyze_challenge_conditions(all_results)

    def _calculate_global_metrics(self, results, iou_thresholds, max_detections):
        """计算全局指标"""
        ap_data = {thresh: [] for thresh in iou_thresholds}
        ar_data = {det: [] for det in max_detections}
        
        for res in results:
            for i, j, iou in res['matches']:
                for thresh in iou_thresholds:
                    ap_data[thresh].append(int(iou >= thresh))
            
            for det in max_detections:
                valid_matches = [m for m in res['matches'] if m[0] < det]
                ar_data[det].append(len(valid_matches)/res['num_gt'] if res['num_gt'] else 0)
        
        # 计算AP
        self.metrics['global']['AP'] = {
            f'AP{int(t*100)}': np.mean(ap_data[t]) for t in iou_thresholds
        }
        
        # 计算AR
        self.metrics['global']['AR'] = {
            f'AR{d}': np.mean(ar_data[d]) for d in max_detections
        }

    def _analyze_challenge_conditions(self, results):
        """挑战条件分析"""
        challenge_stats = {label: {'AP50': [], 'AR100': []} for label in self.attribute_labels}
    
        # 所有帧共享同一组属性（整个视频的属性）
        frame_attrs = self.attributes  # 形状应为 (12,)
    
        for res in results:
            # 无需根据帧号索引属性
            for i, label in enumerate(self.attribute_labels):
                if frame_attrs[i]:  # 直接使用全局属性
                    ap50 = len([m for m in res['matches'] if m[2] >= 0.5]) / len(res['matches']) if res['matches'] else 0
                    ar100 = len([m for m in res['matches'] if m[0] < 100]) / res['num_gt'] if res['num_gt'] else 0
                    challenge_stats[label]['AP50'].append(ap50)
                    challenge_stats[label]['AR100'].append(ar100)
    
        # 计算平均指标
        self.metrics['challenge_specific'] = {
            label: {
                'AP50': np.mean(stats['AP50']) if stats['AP50'] else 0,
                'AR100': np.mean(stats['AR100']) if stats['AR100'] else 0
            }
            for label, stats in challenge_stats.items()
        }

    def visualize_results(self, output_dir):
        """生成可视化报告"""
        os.makedirs(output_dir, exist_ok=True)
    
        # 创建复合图表布局
        fig = plt.figure(figsize=(20, 25))
        gs = GridSpec(4, 2, figure=fig)
    
        # 全局指标表格
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_global_metrics_table(ax1)
    
        # AP/AR对比图
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_ap_ar_comparison(ax2)
    
        # 挑战条件分析
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_challenge_analysis(ax3, output_dir)  # 添加 output_dir 参数
    
        # 保存结果
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comprehensive_analysis.png'), dpi=300)
        plt.close()

    def _plot_global_metrics_table(self, ax):
        """绘制全局指标表格"""
        data = []
        for ap in self.metrics['global']['AP']:
            data.append([ap, self.metrics['global']['AP'][ap]])
        for ar in self.metrics['global']['AR']:
            data.append([ar, self.metrics['global']['AR'][ar]])
        
        df = pd.DataFrame(data, columns=['Metric', 'Value'])
        ax.axis('off')
        pd.plotting.table(
            ax, 
            df.round(3),
            loc='center',
            colWidths=[0.3, 0.3],
            cellLoc='center'
        )
        ax.set_title('Global Performance Metrics', fontsize=12, pad=20)

    def _plot_ap_ar_comparison(self, ax):
        """绘制AP/AR对比图"""
        # AP数据
        ap_data = self.metrics['global']['AP']
        x_ap = [k.replace('AP', '') for k in ap_data.keys()]
        y_ap = list(ap_data.values())
        
        # AR数据
        ar_data = self.metrics['global']['AR']
        x_ar = [k.replace('AR', '') for k in ar_data.keys()]
        y_ar = list(ar_data.values())
        
        # 双轴图
        ax2 = ax.twinx()
        
        # AP柱状图
        bars = ax.bar(x_ap, y_ap, color=sns.color_palette()[0], alpha=0.7, label='AP')
        ax.bar_label(bars, fmt='%.2f', padding=3)
        
        # AR折线图
        line = ax2.plot(x_ar, y_ar, marker='o', color=sns.color_palette()[1], linewidth=2, label='AR')
        ax2.set_ylim(0, 1)
        
        # 样式设置
        ax.set_xlabel('Threshold / Max Detections')
        ax.set_ylabel('AP')
        ax2.set_ylabel('AR')
        ax.set_title('Precision-Recall Analysis')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

    def _plot_challenge_analysis(self, ax, output_dir):
        """挑战条件分析雷达图"""
        # 准备数据
        labels = np.array(self.attribute_labels)
        stats = np.array([[v['AP50'], v['AR100']] for v in self.metrics['challenge_specific'].values()])
    
        # 处理分母为零的情况
        min_vals = stats.min(axis=0)
        max_vals = stats.max(axis=0)
        denominator = max_vals - min_vals
    
        # 避免除以零：分母为零时设为1，并在标准化后置零
        denominator = np.where(denominator == 0, 1, denominator)
        stats_normalized = (stats - min_vals) / denominator
    
        # 将实际分母为零的列置零
        zero_denominator_mask = (max_vals - min_vals) == 0
        stats_normalized[:, zero_denominator_mask] = 0.0
    
        # 更新数据用于绘图
        stats = stats_normalized
    
        # 创建雷达图
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
    
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(polar=True)
    
        # 绘制AP50
        values = stats[:,0].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label='AP50')
        ax.fill(angles, values, alpha=0.25)
    
        # 绘制AR100
        values = stats[:,1].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label='AR100')
        ax.fill(angles, values, alpha=0.25)
    
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title('Challenge-Specific Performance', pad=20)
        ax.legend(loc='upper right')
        plt.savefig(os.path.join(output_dir, 'challenge_radar.png'), dpi=300)
        plt.close()

# 使用示例
if __name__ == "__main__":
    evaluator = AdvancedTrackingEvaluator(
        pred_dir="C:/Users/simplehearted/Desktop/FollowAnything_HIT/results/VisDrone_Results/Annotations/uav0000014_00667_s",
        gt_dir=r"D:\VisDrone2019-SOT-train_part1\VisDrone2019-SOT-train\annotations",
        attr_path=r"D:\VisDrone2019-SOT-train_part1\VisDrone2019-SOT-train\attributes\uav0000014_00667_s_attr.txt"
    )
    
    evaluator.evaluate()
    evaluator.visualize_results(r"C:\Users\simplehearted\Desktop\FollowAnything_HIT\results\VisDrone_Results")
    