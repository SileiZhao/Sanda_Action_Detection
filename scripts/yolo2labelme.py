import os
import cv2
import numpy as np
from ultralytics import YOLO
from labelme import LabelFile

# 加载YOLOv8模型
model = YOLO("yolov8n.pt")  # 使用YOLOv8模型

# 定义图片目录
image_dir = "/home/zhaosilei/Datasets/Benz/sanda_det"

# 定义输出目录
output_dir = "/home/zhaosilei/Datasets/Benz/sanda_det_labelme"
os.makedirs(output_dir, exist_ok=True)

# 定义颜色范围（HSV格式）
red_lower1 = np.array([0, 50, 50])  # 红色的HSV范围
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 50, 50])
red_upper2 = np.array([180, 255, 255])

blue_lower = np.array([100, 50, 50])  # 蓝色的HSV范围
blue_upper = np.array([130, 255, 255])

def detect_color(image, bbox):
    """
    检测检测框内人物的衣服颜色。
    :param image: 原始图像
    :param bbox: 检测框坐标 (x1, y1, x2, y2)
    :return: "red_side" 或 "blue_side"
    """
    x1, y1, x2, y2 = map(int, bbox)
    roi = image[y1:y2, x1:x2]  # 提取检测框区域
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # 转换为HSV颜色空间

    # 检测红色
    red_mask1 = cv2.inRange(hsv_roi, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv_roi, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_pixels = cv2.countNonZero(red_mask)

    # 检测蓝色
    blue_mask = cv2.inRange(hsv_roi, blue_lower, blue_upper)
    blue_pixels = cv2.countNonZero(blue_mask)

    # 判断颜色
    if red_pixels > blue_pixels:
        return "red_side"
    else:
        return "blue_side"

# 遍历图片目录
for image_name in os.listdir(image_dir):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # 使用YOLOv8进行检测
    results = model(image)

    # 初始化LabelMe格式的标注数据
    shapes = []
    for result in results:
        for box in result.boxes:
            # 获取检测框的坐标和类别
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])

            # 过滤掉非人的检测框
            if cls_id != 0:  # YOLOv8中人的类别ID通常是0
                continue

            # 计算检测框的面积
            area = (x2 - x1) * (y2 - y1)

            # 过滤掉面积小于120000的检测框
            if area < 120000:
                continue

            # 检测衣服颜色
            label = detect_color(image, (x1, y1, x2, y2))

            # 将检测框转换为LabelMe格式
            shape = {
                "label": label,
                "points": [[x1, y1], [x2, y2]],
                "shape_type": "rectangle",
                "flags": {},
                "group_id": None,
                "description": "",
                "mask": None,
            }
            shapes.append(shape)

    # 创建LabelMe格式的JSON文件
    label_file = LabelFile()
    label_file.save(
        os.path.join(output_dir, image_name.replace(".jpg", ".json")),
        shapes,
        image_path,
        height,
        width
    )

    print(f"Processed {image_name} and saved annotations.")

print("All images processed and annotations saved.")