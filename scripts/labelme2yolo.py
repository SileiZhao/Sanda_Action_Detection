import os
import json
import random
import shutil
from pathlib import Path

# 定义路径
labelme_dir = "/home/zhaosilei/Datasets/Benz/sanda_det_labelme"
output_dir = "/home/zhaosilei/Datasets/Benz/sanda_det_yolo"
images_dir = os.path.join(output_dir, "images")
labels_dir = os.path.join(output_dir, "labels")
train_images_dir = os.path.join(images_dir, "train")
val_images_dir = os.path.join(images_dir, "val")
train_labels_dir = os.path.join(labels_dir, "train")
val_labels_dir = os.path.join(labels_dir, "val")

# 创建目录
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# 定义类别映射
class_mapping = {"red_side": 0, "blue_side": 1}

# 定义训练集和验证集的比例
train_ratio = 0.8

# 获取所有LabelMe标注文件
labelme_files = [f for f in os.listdir(labelme_dir) if f.endswith(".json")]
random.shuffle(labelme_files)  # 打乱顺序

# 划分训练集和验证集
num_train = int(len(labelme_files) * train_ratio)
train_files = labelme_files[:num_train]
val_files = labelme_files[num_train:]

# 用于保存图片路径
train_txt_path = os.path.join(output_dir, "train.txt")
val_txt_path = os.path.join(output_dir, "val.txt")


def convert_labelme_to_yolo(labelme_path, output_label_path, output_image_path):
    """
    将LabelMe格式的标注转换为YOLO格式
    :param labelme_path: LabelMe标注文件的路径
    :param output_label_path: 输出YOLO标注文件的路径
    :param output_image_path: 输出图片的路径
    """
    with open(labelme_path, "r") as f:
        data = json.load(f)

    # 从标注文件名推断图片路径
    image_name = Path(labelme_path).stem + ".jpg"
    image_path = os.path.join(os.path.dirname(labelme_path), image_name)

    image_width = data["imageWidth"]
    image_height = data["imageHeight"]

    if not os.path.exists(image_path):
        print(f"警告：图片 {image_path} 不存在，跳过该标注文件。")
        return

    # 复制图片到目标目录
    shutil.copy(image_path, output_image_path)

    # 生成YOLO格式的标注
    with open(output_label_path, "w") as f:
        for shape in data["shapes"]:
            label = shape["label"]
            if label not in class_mapping:
                continue

            # 获取类别ID
            class_id = class_mapping[label]

            # 获取检测框坐标
            points = shape["points"]
            x1, y1 = points[0]
            x2, y2 = points[1]

            # 转换为YOLO格式（中心点坐标和宽高，归一化）
            x_center = (x1 + x2) / 2 / image_width
            y_center = (y1 + y2) / 2 / image_height
            width = abs(x2 - x1) / image_width
            height = abs(y2 - y1) / image_height

            # 写入YOLO格式的标注文件
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


with open(train_txt_path, "w") as train_txt:
    for file in train_files:
        labelme_path = os.path.join(labelme_dir, file)
        image_name = Path(file).stem + ".jpg"
        label_name = Path(file).stem + ".txt"

        # 生成输出路径
        output_image_path = os.path.join(train_images_dir, image_name)
        output_label_path = os.path.join(train_labels_dir, label_name)

        # 转换标注格式
        convert_labelme_to_yolo(labelme_path, output_label_path, output_image_path)

        # 将图片路径写入train.txt
        train_txt.write(f"{output_image_path}\n")


with open(val_txt_path, "w") as val_txt:
    for file in val_files:
        labelme_path = os.path.join(labelme_dir, file)
        image_name = Path(file).stem + ".jpg"
        label_name = Path(file).stem + ".txt"

        # 生成输出路径
        output_image_path = os.path.join(val_images_dir, image_name)
        output_label_path = os.path.join(val_labels_dir, label_name)

        # 转换标注格式
        convert_labelme_to_yolo(labelme_path, output_label_path, output_image_path)

        # 将图片路径写入val.txt
        val_txt.write(f"{output_image_path}\n")

print("转换完成！")
