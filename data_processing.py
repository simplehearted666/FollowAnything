'''
VisDrone2019数据集转换为COCO数据集格式
并进行数据增强

'''
import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
from collections import defaultdict


#--------------------配置参数------------------------
class Config:
    #数据集参数
    DATA_ROOT = r"D:\VisDrone2019-DET-train"
    COCO_ANN_DIR = "annotations"
    IMAGE_DIR = "images"

    #训练参数
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    IMG_SIZE = 1024
    NUM_WORKERS = 4

    #模型参数
    #优化参数
    
    #保存路径
    SAVE_DIR = "checkpoints"


#--------------------数据转换------------------------
class VisDroneToCOCO:
    '''
    将VisDrone原始标注转换为COCO格式的转换器

    功能：
    1.解析VisDrone的txt格式标注
    2.转换为标准的COCO JSON格式
    3.保持原始坐标系统和类别定义

    参数：
    visdrone_root:VisDrone数据集根目录
    output_dir:COCO格式输出目录

    '''
    def __init__(self):
        #数据分类映射
        self.categories = [
            {"id": 0, "name": "ignored"},
            {"id": 1, "name": "pedestrian"},
            {"id": 2, "name": "people"},
            {"id": 3, "name": "bicycle"},
            {"id": 4, "name": "car"},
            {"id": 5, "name": "van"},
            {"id": 6, "name": "truck"},
            {"id": 7, "name": "tricycle"},
            {"id": 8, "name": "awning-tricycle"},
            {"id": 9, "name": "bus"},
            {"id": 10, "name": "motor"},
            {"id": 11, "name": "others"}

        ]

    def convert(self, split):
        """执行转换主函数
        
        处理流程：
        1. 遍历指定split的所有标注文件
        2. 解析每个标注文件中的目标信息
        3. 构建COCO格式的JSON结构
        4. 保存到指定路径
        """


        coco_data = {
            "images":[],
            "annotations":[],
            "categories": self.categories
        }#创建COCO数据字典

        ann_folder = os.path.join(Config.DATA_ROOT, Config.COCO_ANN_DIR, split)  
        img_folder = os.path.join(Config.DATA_ROOT, Config.IMAGE_DIR, split)

        image_id = 0
        annotation_id = 0#初始化

        for img_file in tqdm(sorted(os.listdir(img_folder)), desc = f"Converting {split}"):
            if not img_file.endswith(".jpg"):
                continue

            img_path = os.path.join(img_folder, img_file)#将当前图片移动到新文件夹
            img = cv2.imread(img_path)
            h, w = img.shape[:2]#获取图片高度和宽度信息

            coco_data["images"].append(
                {
                    "id":image_id,
                    "file_name": img_file,
                    "width": w,
                    "height": h
                }
            )

            txt_path = os.path.join(ann_folder, img_file.replace(".jpg", ".txt"))

            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    for line in f:
                        try:
                            parts = list(map(float, line.strip().split(",")))
                        except ValueError:
                            print(f"无效数据行：{line}")
                        if len(parts) < 8:
                            continue
                        
                        '''
                        <左上角x>:预测边界框的左上角的x坐标;

                        <左上角y>:预测对象边界框的左上角的y坐标;

                        <宽度>:预测对象包围框的宽度(以像素为单位);

                        <高度>:预测对象包围框的像素高度;

                        <得分>:检测结果文件中的分数表明了包围一个对象实例的预测边界框的置信度。“GROUNDTRUTH”文件中的分数设置为1或0。1表示计算时考虑包围盒,0表示忽略包围盒;

                        <类别>：对象类别表示标注对象的类型，(即忽略区域(0)，行人(1)，人(2)，自行车(3)，轿车(4)，货车(5)，卡车(6)，三轮车(7)，遮阳三轮车(8)，公共汽车(9)，马达(10)，其他(11)));

                        <截断>：检测结果文件中的分数应该设置为常数-1。GROUNDTRUTH文件中的分数表示物体部分出现在帧外的程度(即，没有截断= 0(截断率0%)，部分截断= 1(截断率1%~50%));

                        <遮挡>：检测结果文件中的分数应该设置为常数-1。GROUNDTRUTH文件中的分数表示物体被遮挡的比例(即没有遮挡=0(遮挡率0%)，部分遮挡=1(遮挡率1%~50%)，重度遮挡= 2(遮挡率50%~100%))。
                        '''

                        x, y, bw, bh, score, cat_id, trunc, occ = parts
                        cat_id = int(cat_id)

                        if bw < 1 or bh < 1:
                            continue

                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": cat_id,
                            "bbox": [x, y, bw, bh],
                            "area": bw * bh,
                            "iscrowd": 0,#coco格式规范，0表示单个可明确区分的实例
                            "truncated": trunc,
                            "occluded": occ
                        })
                        annotation_id += 1#读取下一行
                

            image_id += 1#读取下一张图
        
        output_path = os.path.join(Config.DATA_ROOT, Config.COCO_ANN_DIR, f"{split}_coco.json")
        with open(output_path, "w") as f:
            json.dump(coco_data, f, indent = 2)#将coco_data字典序列化为 JSON 格式并写入文件

        print(f"Converted{split}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")




#--------------------数据增强------------------------
class DroneAugmentor:
    def __init__(self, img_size):
        #训练集数据增强
        self.train_transform = A.Compose([
            A.LongestMaxSize(img_size),#将图像的最长边调整为 img_size
            
            #如果图像尺寸小于 img_size，则进行填充。
            #border_mode=cv2.BORDER_REPLICATE 表示使用复制边缘像素的方式填充，
            #position=A.PadIfNeeded.PositionType.TOP_LEFT 表示从左上角开始填充。
            A.PadIfNeeded(
                img_size, img_size,
                border_mode = cv2.BORDER_REPLICATE,
                position = A.PadIfNeeded.PositionType.TOP_LEFT
            ),
            A.HorizontalFlip(p = 0.5),#以 0.5 的概率对图像进行水平翻转
            A.VerticalFlip(p = 0.2),#以 0.2 的概率对图像进行垂直翻转
            A.RandomRotate90(p = 0.3),#以 0.3 的概率将图像随机旋转 90 度
            A.RandomSizedBBoxSafeCrop(img_size, img_size, p = 0.5),#以 0.5 的概率对图像进行随机裁剪，同时保证裁剪后的图像包含目标边界框
        
            #从给定的变换列表中随机选择一个进行操作，选择的概率为 p=0.5。
            # 包括 A.MotionBlur（运动模糊）、A.GaussianBlur（高斯模糊）和 A.GaussNoise（高斯噪声）
            A.OneOf([
                A.MotionBlur(p = 0.2),
                A.GaussianBlur(p = 0.3),
                A.GaussNoise(p = 0.5)
            ], p = 0.5),

            #以 0.5 的概率对图像的亮度、对比度、饱和度和色调进行随机调整
            A.ColorJitter(
                brightness = 0.2,
                contrast = 0.2,
                saturation = 0.2,
                hue = 0.1,
                p = 0.5
            ),

            #对RGB的三个通道进行归一化处理
            A.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            ),

            ToTensorV2()
            ],
        
            #指定边界框的格式为 coco，
            # 设置最小可见性为 0.25，
            # 标签字段为 category_ids
            bbox_params = A.BboxParams(
                fomat = "coco",
                min_visibility = 0.25,
                label_fields = ["category_ids"]
            )
        )

        
        #验证集数据预处理
        self.val_transform = A.Compose([
            A.LongestMaxSize(img_size),
            A.PadIfNeeded(
                img_size, img_size,
                border_mode = cv2.BORDER_CONSTANT,
                value = 0
            ),#使用黑色填充
            A.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            ),
            ToTensorV2()
            ],
            bbox_params = A.BboxParams(
                format = "coco",
                label_fields = ["category_ids"]
            )
        )



#---------------------自定义数据集类------------------------
class VisDroneDataset(Dataset):
    def __init__(self, split, transform = None):
        self.split = split
        self.transform = transform
        self.data = self._load_coco_annotations()#加载coco格式的标注数据
    
    def _load_coco_annotations(self):
        ann_path = os.path.join(Config.DATA_ROOT, Config.COCO_ANN_DIR, f"{self.split}_coco.json")
        with open(ann_path) as f:
            data = json.load(f)#加载JSON文件
        
        images = {img["id"]: img for img in data["images"]} #创建图像ID到图像信息的映射
        #images格式：
        #{
        # 0：{'id': 0, 'file_name': '0000001_00000_d_0000001.jpg', 'width': 1920, 'height': 1080},
        # 1: {'id': 1, 'file_name': '0000001_00000_d_0000002.jpg', 'width': 1920, 'height': 1080},
        #......
        #}
        annotations = defaultdict(list)
        for ann in data["annotations"]:
            annotations[ann["image_id"]].append(ann)#按图像ID将标注分组
        #annotations格式：
        #{
        # 0: [{'id': 0, 'image_id': 0, 'category_id': 1, 'bbox': [0.0, 0.0, 0.0, 0.0], 'area': 0.0, 'iscrowd': 0, 'truncated': 0.0, 'occ': 0.0},......],
        # 1: [{},{}, ......],
        #......
        #}
        return {"images": images, "annotations": annotations}
    
    def __len__(self):
        return len(self.data["images"])
    
    def __getitem__(self, idx):
        img_info = list(self.data["images"].values())[idx]
        #img_info格式：
        #{'id': 0, 'file_name': '0000001_00000_d_0000001.jpg', 'width': 1920, 'height': 1080}
        img_path = os.path.join(Config.DATA_ROOT, Config.IMAGE_DIR, self.split, img_info["file_name"])#构建图像路径

        #读取图像（OpenCV BGR格式转RGB）
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #获取该图像对应标注列表
        anns = self.data["annotations"][img_info["id"]]
        boxes = [ann["bbox"] for ann in anns]#提取该图所有边界框坐标
        category_ids = [ann["category_id"] + 1 for ann in anns]#类别ID（+1是因为coco类别ID从1开始，保留0作为背景类别）

        #数据增强
        if self.transform:
            transformed = self.transform(
                image = image, 
                bboxes = boxes,
                category_ids = category_ids
            )#生成数据增强实例
            image = transformed["image"]
            boxes = transformed["bboxes"]
            category_ids = transformed["category_ids"]
        
        #构建目标字典
        target = {
            "boxes": torch.as_tensor(boxes, dtype = torch.float32),
            "labels": torch.as_tensor(category_ids, dtype = torch.int64),
            "image_id": torch.tensor([img_info["id"]])
        }
        #target格式：
        #{
        # 'boxes': tensor([[   ], [   ], ......]),
        # 'labels': tensor([  ,  , ......]),
        # 'image_id': tensor([  ])
        #}
        #image格式：
        #tensor([3, H, W])
        return image, target



#---------------------主函数---------------------------
def main():
    #初始化系统
    os.makedirs(Config.SAVE_DIR, exist_ok = True)

    #转换COCO格式
    converter = VisDroneToCOCO()
    for split in ["train", "val"]:
        ann_path = os.path.join(Config.DATA_ROOT, Config.COCO_ANN_DIR, f"{split}_coco.json")
        if not os.path.exists(ann_path):
            converter.convert(split)


    #准备数据增强
    augmentor = DroneAugmentor(Config.IMG_SIZE)

    #创建数据集
    train_dataset = VisDroneDataset("train", augmentor.train_transform)
    val_dataset = VisDroneDataset("val", augmentor.val_transform)

    #创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size = Config.BATCH_SIZE,#每个批次的样本数量
        shuffle = True,#在每个epoch开始时打乱数据顺序
        num_workers = Config.NUM_WORKERS,#用于数据加载的子进程数
        collate_fn = lambda x: tuple(zip(*x)),
        #pin_memory = True, #启用锁页内存
        drop_last = True
    )

    # 原始批次数据:
    #    batch = [
    #       (img1_tensor, {"boxes": box1, "labels": label1}),
    #       (img2_tensor, {"boxes": box2, "labels": label2})
    #    ]

    # 经过 collate_fn 处理后：
    #    (
    #        [img1_tensor, img2_tensor],  # 图像列表
    #        [
    #            {"boxes": box1, "labels": label1}, 
    #            {"boxes": box2, "labels": label2}
    #        ]  # 标注字典列表
    #    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size = Config.BATCH_SIZE,
        num_workers = Config.NUM_WORKERS,
        collate_fn = lambda x: tuple(zip(*x))
    )


    #------------------后续示例代码-----------------------

    #初始化模型
    #model  = get_model()

    #训练准备
    #trainer = DroneTrainer(model, train_loader, val_loader)

    #训练循环
    #for epoch in range(Config.NUM_EPOCHS):
    #   train_loss = trainer.train_epoch（epoch)
    #   val_map = trainer.validate()
    #   print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
    #   print(f"Train Loss: {train_loss:.4f} | Val mAP:{val_map:.4f}")
    #   trainer.save_checkpoint(epoch, val_map)
    #print("Training completed!")


if __name__ == "__main__":
    main()