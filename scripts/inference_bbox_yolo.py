"""
YOLO 推理检测框
"""

import os
import cv2
import json
import tqdm
from ultralytics import YOLO
from pathlib import Path

model = YOLO("/Sanda_action_recognation/runs/detect/train/weights/best.pt")

class_mapping = {0: "red_side", 1: "blue_side"}
image_dir = "/Users/zhaosilei/Downloads/sanda_det"
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


def yolo_to_labelme(image_path, results, output_json_path):
    """
    将YOLO推理结果转换为LabelMe格式的标注文件。
    :param image_path: 图片路径
    :param results: YOLO推理结果
    :param output_json_path: 输出LabelMe标注文件的路径
    """
    # 读取图片尺寸
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    shapes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            label = class_mapping.get(cls_id, "unknown")

            shape = {
                "label": label,
                "points": [[x1, y1], [x2, y2]],
                "shape_type": "rectangle",
                "flags": {},
                "group_id": None,
                "description": "",
                "mask": None,
                "confidence": conf,
            }
            shapes.append(shape)

    labelme_data = {
        "version": "5.1.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }

    with open(output_json_path, "w") as f:
        json.dump(labelme_data, f, indent=2)


for image_file in tqdm.tqdm(image_files):
    image_path = os.path.join(image_dir, image_file)
    output_json_path = os.path.join(image_dir, Path(image_file).stem + ".json")

    results = model(image_path)

    yolo_to_labelme(image_path, results, output_json_path)
    print(f"已生成标注文件：{output_json_path}")

print("推理完成！")
