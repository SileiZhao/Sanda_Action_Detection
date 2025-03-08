"""
该脚本用于生成bbox和keypoints标注
"""
from ultralytics import YOLO
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from mmpose.apis import inference_topdown
from mmpose.structures import merge_data_samples
from mmpose.apis import init_model as init_pose_estimator

# 定义路径
video_folder = '/Users/zhaosilei/PycharmProjects/Benz/Sanda_Action_Detection/data'
det_checkpoint = '/Users/zhaosilei/PycharmProjects/Benz/Sanda_Action_Detection/exp/runs/detect/train/weights/best.pt'
pose_config = '/Users/zhaosilei/PycharmProjects/Benz/mmpose-main/configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb256-420e_body8-256x192.py'
pose_checkpoint = '/Users/zhaosilei/PycharmProjects/Benz/mmpose-main/checkoints/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth'

output_folder = '/Users/zhaosilei/PycharmProjects/Benz/Sanda_Action_Detection/output'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 初始化 YOLO 目标检测器
detector = YOLO(det_checkpoint)

# 初始化姿态估计器
pose_estimator = init_pose_estimator(pose_config, pose_checkpoint, device='cpu')


def process_video(video_path, csv_path):
    # 读取原始 CSV 文件
    original_df = pd.read_csv(csv_path)

    # 创建一个新的 DataFrame 用于存储结果
    columns = ['video_name', 'frame_number', 'label', 'athlete_id',
               'x1', 'y1', 'x2', 'y2']  # bbox 拆分为 4 列
    # 添加关键点列（17 个关键点，每个关键点包含 x、y、score）
    for i in range(1, 18):
        columns.extend([f'p{i}x', f'p{i}y', f'p{i}s'])
    new_df = pd.DataFrame(columns=columns)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 从原始 CSV 文件中读取当前帧的动作标签
        frame_label = original_df[original_df['frame_number'] == frame_idx]['label'].values
        frame_label = frame_label[0] if len(frame_label) > 0 else 0  # 如果找不到标签，默认为 0

        # 目标检测
        det_results = detector(frame)  # 使用 YOLO 进行目标检测
        red_boxes = []
        blue_boxes = []

        for result in det_results:
            for box in result.boxes:
                # 获取检测框的类别和坐标
                cls_id = int(box.cls)  # 类别 ID
                bbox = box.xyxy.cpu().numpy()[0]  # 检测框坐标 [x1, y1, x2, y2]
                conf = box.conf.cpu().numpy()[0]  # 置信度

                # 过滤出 red_side 和 blue_side 的检测框
                if cls_id == 0:  # red_side
                    red_boxes.append({'bbox': bbox, 'conf': conf})
                elif cls_id == 1:  # blue_side
                    blue_boxes.append({'bbox': bbox, 'conf': conf})

        # 每帧最多保留一个 red_side 和一个 blue_side，选择置信度最高的
        if len(red_boxes) > 1:
            red_boxes = [max(red_boxes, key=lambda x: x['conf'])]
        if len(blue_boxes) > 1:
            blue_boxes = [max(blue_boxes, key=lambda x: x['conf'])]

        # 如果没有检测到目标，留空列表
        red_box = red_boxes[0]['bbox'] if red_boxes else np.array([])
        blue_box = blue_boxes[0]['bbox'] if blue_boxes else np.array([])

        # 姿态估计
        red_pose_results = inference_topdown(pose_estimator, frame, np.array([red_box])) if red_box.size > 0 else []
        blue_pose_results = inference_topdown(pose_estimator, frame, np.array([blue_box])) if blue_box.size > 0 else []

        # 合并姿态估计结果
        red_pose_instances = merge_data_samples(red_pose_results).pred_instances if red_pose_results else []
        blue_pose_instances = merge_data_samples(blue_pose_results).pred_instances if blue_pose_results else []

        # 将结果写入新的 DataFrame
        if red_box.size > 0:
            # 拆分 bbox
            x1, y1, x2, y2 = red_box.tolist()
            # 获取关键点和关键点分数
            keypoints = red_pose_instances.keypoints.tolist()[0] if red_pose_instances else []
            keypoint_scores = red_pose_instances.keypoint_scores.tolist()[0] if red_pose_instances else []
            # 合并关键点和关键点分数
            keypoint_data = []
            for kp, score in zip(keypoints, keypoint_scores):
                keypoint_data.extend([kp[0], kp[1], score])
            # 写入新行
            new_row = [os.path.basename(video_path), frame_idx, frame_label, 0, x1, y1, x2, y2] + keypoint_data
            new_df.loc[len(new_df)] = new_row

        if blue_box.size > 0:
            # 拆分 bbox
            x1, y1, x2, y2 = blue_box.tolist()
            # 获取关键点和关键点分数
            keypoints = blue_pose_instances.keypoints.tolist()[0] if blue_pose_instances else []
            keypoint_scores = blue_pose_instances.keypoint_scores.tolist()[0] if blue_pose_instances else []
            # 合并关键点和关键点分数
            keypoint_data = []
            for kp, score in zip(keypoints, keypoint_scores):
                keypoint_data.extend([kp[0], kp[1], score])
            # 写入新行
            new_row = [os.path.basename(video_path), frame_idx, frame_label, 1, x1, y1, x2, y2] + keypoint_data
            new_df.loc[len(new_df)] = new_row

        frame_idx += 1

    # 保存新的 DataFrame 到 CSV 文件，覆盖原始文件
    new_df.to_csv(os.path.join(output_folder, os.path.basename(csv_path)), index=False)
    cap.release()


# 主函数
def main():
    # 遍历视频文件夹中的所有视频文件
    for root, dirs, files in os.walk(video_folder):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                csv_path = os.path.join(root, file.replace('.mp4', '.csv'))

                if os.path.exists(csv_path):
                    print(f'Processing {video_path}...')
                    process_video(video_path, csv_path)
                    print(f'Finished processing {video_path}')


if __name__ == '__main__':
    main()
