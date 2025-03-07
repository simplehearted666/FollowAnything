'''生成掩膜文件'''
import cv2
import os
import numpy as np
from tqdm import tqdm
import logging

def setup_logger():
    """配置多级日志记录器"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 文件日志（存储DEBUG级别以上）
    file_handler = logging.FileHandler("conversion_fixed.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # 控制台日志（仅显示INFO级别以上）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def parse_annotation_line(line, line_number):
    """解析四字段标注行"""
    parts = line.strip().split(',')
    
    # 验证字段数量
    if len(parts) != 4:
        raise ValueError(f"需要4个字段，实际得到{len(parts)}个")
    
    try:
        x = int(parts[0])
        y = int(parts[1])
        w = int(parts[2])
        h = int(parts[3])
    except ValueError as e:
        raise ValueError(f"数值转换失败: {e}")
    
    return x, y, w, h

def validate_bbox(x, y, w, h, img_w, img_h):
    """增强型边界框验证"""
    # 检查非负值
    if any(v < 0 for v in [x, y, w, h]):
        return False
    
    # 检查有效性
    if w <= 0 or h <= 0:
        return False
    
    # 检查越界
    if (x + w) > img_w or (y + h) > img_h:
        return False
    
    return True

def process_sequence(anno_path, img_dir, mask_dir):
    """处理单个序列"""
    success_count = 0
    created_dir = False
    
    try:
        # 读取并预处理标注文件
        with open(anno_path, 'r') as f:
            raw_lines = [ln.strip() for ln in f.readlines()]
        
        # 过滤空行和注释行（以#开头）
        valid_lines = [(i+1, ln) for i, ln in enumerate(raw_lines) 
                      if ln and not ln.startswith('#')]
        
        # 获取有序图像文件列表
        img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith('.jpg') and f.startswith('img')
        ], key=lambda x: int(x[3:-4]))
        
        # 主处理循环
        with tqdm(zip(img_files, valid_lines), 
                 total=len(img_files),
                 desc=os.path.basename(anno_path),
                 leave=False,
                 unit='frame') as pbar:
            
            for img_file, (line_num, line) in pbar:
                # 生成帧索引
                frame_idx = img_file[3:-4].zfill(8)
                
                try:
                    # 解析标注行
                    x, y, w, h = parse_annotation_line(line, line_num)
                except ValueError as e:
                    logging.warning(f"行{line_num}无效: {line} | 原因: {e}")
                    continue
                
                # 处理图像
                img_path = os.path.join(img_dir, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        raise IOError("图片读取失败")
                    img_h, img_w = img.shape[:2]
                except Exception as e:
                    logging.error(f"图片处理错误: {img_path} | {e}")
                    continue
                
                # 验证边界框
                if not validate_bbox(x, y, w, h, img_w, img_h):
                    logging.debug(f"行{line_num}无效边界框: {x},{y},{w},{h} (图像尺寸 {img_w}x{img_h})")
                    continue
                
                # 创建掩码
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                mask[y:y+h, x:x+w] = 255
                
                # 延迟创建目录
                if not created_dir:
                    os.makedirs(mask_dir, exist_ok=True)
                    created_dir = True
                
                # 保存掩码
                mask_path = os.path.join(mask_dir, f"{frame_idx}.png")
                if cv2.imwrite(mask_path, mask):
                    success_count += 1
                else:
                    logging.error(f"掩码保存失败: {mask_path}")
    
    except Exception as e:
        logging.error(f"序列处理异常: {anno_path} | {e}")
        return success_count
    
    finally:
        # 清理空目录
        if created_dir and (success_count == 0):
            try:
                os.rmdir(mask_dir)
                logging.info(f"已清理空目录: {mask_dir}")
            except OSError:
                pass
    
    return success_count

def batch_convert(base_dir):
    """批量转换主流程"""
    # 路径校验
    required_dirs = ['annotations', 'sequences']
    for d in required_dirs:
        path = os.path.join(base_dir, d)
        if not os.path.exists(path):
            raise FileNotFoundError(f"必需目录不存在: {path}")
    
    anno_root = os.path.join(base_dir, 'annotations')
    img_root = os.path.join(base_dir, 'sequences')
    mask_root = os.path.join(base_dir, 'Annotations')
    
    # 获取有效标注文件
    anno_files = sorted([
        f for f in os.listdir(anno_root)
        if f.endswith('.txt') and not f.startswith('.')
    ])
    
    # 主进度条
    with tqdm(anno_files, desc='总进度', unit='seq', 
             bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}") as main_pbar:
        total_stats = {'success': 0, 'failed': 0}
        
        for anno_file in main_pbar:
            seq_name = os.path.splitext(anno_file)[0]
            main_pbar.set_postfix_str(f"当前序列: {seq_name[:15]}")
            
            anno_path = os.path.join(anno_root, anno_file)
            img_dir = os.path.join(img_root, seq_name)
            mask_dir = os.path.join(mask_root, seq_name)
            
            # 检查图片目录
            if not os.path.isdir(img_dir):
                logging.error(f"图片目录缺失: {img_dir}")
                total_stats['failed'] += 1
                continue
            
            # 执行转换
            success_frames = process_sequence(anno_path, img_dir, mask_dir)
            
            # 统计结果
            if success_frames > 0:
                total_stats['success'] += 1
                logging.info(f"成功处理 {success_frames} 帧: {seq_name}")
            else:
                total_stats['failed'] += 1
                logging.warning(f"序列无有效数据: {seq_name}")
            
            main_pbar.set_postfix_str(f"有效序列: {total_stats['success']}")
    
    # 最终统计
    logging.info("\n转换完成，最终统计:")
    logging.info(f"成功处理序列: {total_stats['success']}")
    logging.info(f"失败/无效序列: {total_stats['failed']}")

if __name__ == "__main__":
    setup_logger()
    base_dir = r"D:\VisDrone2019-SOT-val\VisDrone2019-SOT-val"
    
    try:
        logging.info("启动数据转换流程")
        batch_convert(base_dir)
        logging.info("所有处理已完成")
    except Exception as e:
        logging.critical(f"程序异常终止: {str(e)}", exc_info=True)
