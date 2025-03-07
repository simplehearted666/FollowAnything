import os
import json
import cv2
from tqdm import tqdm

def build_metadata(data_root, output_path):
    meta = {"videos": {}, "attributes": {}, "file_mapping": {}}
    
    # 获取有效序列列表
    seq_dirs = []
    for d in os.listdir(os.path.join(data_root, 'annotations')):
        anno_dir = os.path.join(data_root, 'annotations', d)
        img_dir = os.path.join(data_root, 'sequences', d)
        if os.path.isdir(anno_dir) and os.path.isdir(img_dir):
            seq_dirs.append(d)
    
    for seq in tqdm(seq_dirs, desc="Processing sequences"):
        # ==== 1. 文件路径验证 ====
        img_dir = os.path.join(data_root, 'sequences', seq)
        anno_dir = os.path.join(data_root, 'annotations', seq)
        attr_path = os.path.join(data_root, 'attributes', f"{seq}_attr.txt")
        
        if not os.path.exists(attr_path):
            print(f"\n警告：缺失属性文件 {seq}_attr.txt")
            continue
        
        # ==== 2. 动态获取分辨率 ====
        first_img = os.path.join(img_dir, "img0000001.jpg")
        if not os.path.exists(first_img):
            print(f"\n警告：缺失首帧图像 {first_img}")
            continue
        h, w = cv2.imread(first_img).shape[:2]
        
        # ==== 3. 构建文件映射 ====
        frame_pairs = []
        try:
            mask_files = sorted([f for f in os.listdir(anno_dir) if f.endswith(".png")])
            for mask_file in mask_files:
                frame_num = mask_file.split('.')[0].zfill(8)  # 保持8位数字
                img_file = f"img{int(frame_num):07d}.jpg"  # 转换为7位数字
                frame_pairs.append((img_file, mask_file))
        except Exception as e:
            print(f"\n处理序列 {seq} 时出错: {str(e)}")
            continue
        
        # ==== 4. 写入元数据 ====
        # Videos
        meta["videos"][seq] = {
            "resolution": [w, h],
            "frame_pairs": frame_pairs,
            "objects": {"0": {"category": "generic", "frames": [p[1] for p in frame_pairs]}}
        }
        
        # File Mapping
        meta["file_mapping"][seq] = {
            "image_pattern": "img{:07d}.jpg",
            "mask_pattern": "{:08d}.png"
        }
        
        # Attributes
        with open(attr_path, 'r') as f:
            attrs = list(map(int, f.readline().strip().split(',')))
            meta["attributes"][seq] = {
                "IV": attrs[0], "SV": attrs[1], "OCC": attrs[2],
                "FM": attrs[3], "MB": attrs[4], "ROT": attrs[5],
                "BC": attrs[6], "LR": attrs[7], "OV": attrs[8],
                "CM": attrs[9], "ARC": attrs[10], "VC": attrs[11]
            }
    
    # ==== 5. 保存前验证 ====
    print(f"\n生成统计：")
    print(f"- 有效视频序列: {len(meta['videos'])}")
    print(f"- 属性记录数: {len(meta['attributes'])}")
    print(f"- 文件映射数: {len(meta['file_mapping'])}")
    
    with open(os.path.join(output_path, 'val.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    data_root = r"C:\Users\simplehearted\Desktop\FollowAnything_HIT\Segment-and-Track-Anything\aot\datasets\VisDrone_SOT\val"  # 替换实际路径
    output_path = os.path.join(data_root, "meta")
    os.makedirs(output_path, exist_ok=True)
    build_metadata(data_root, output_path)