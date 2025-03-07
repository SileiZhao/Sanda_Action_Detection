import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize


def custom_collate_fn(batch):
    """
    自定义 collate_fn，处理不同样本中 N 不一致的情况。
    """
    collated_batch = {}

    # 处理图像数据
    collated_batch["image"] = torch.stack([item["image"] for item in batch], dim=0)

    # 处理 boxes、poses、labels 等字段
    for key in ["boxes", "box_player_ids", "poses", "pose_player_ids", "labels", "label_player_ids"]:
        # 将每个样本的字段存储为列表
        collated_batch[key] = [item[key] for item in batch]

    return collated_batch


class SandaActionDataset(Dataset):
    def __init__(self, root_dir, transform=None, frame_stride=1, clip_length=32):
        """
        初始化数据集。

        参数:
            root_dir (str): 数据集根目录。
            transform (callable, optional): 图像预处理变换。
            frame_stride (int): 帧采样步长（默认为1，即使用所有帧）。
            clip_length (int): 视频片段的帧数。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.frame_stride = frame_stride
        self.clip_length = clip_length

        # 加载所有视频的元数据
        self.metadata = []
        videos_dir = os.path.join(root_dir, "videos")
        rawframes_dir = os.path.join(root_dir, "rawframes")
        annotations_dir = os.path.join(root_dir, "annotations")

        # 遍历视频文件
        for video_name in os.listdir(videos_dir):
            video_id = os.path.splitext(video_name)[0]
            frames_dir = os.path.join(rawframes_dir, video_id)
            annotation_path = os.path.join(annotations_dir, f"{video_id}.csv")

            # 加载标注文件
            if not os.path.exists(annotation_path):
                continue  # 如果标注文件不存在，跳过该视频
            annotations = pd.read_csv(annotation_path)

            # 按帧存储元数据
            frame_files = sorted(os.listdir(frames_dir))
            for i in range(0, len(frame_files) - self.clip_length + 1, self.frame_stride):
                frame_ids = [int(os.path.splitext(frame_files[j])[0].split("_")[-1]) for j in
                             range(i, i + self.clip_length)]
                frame_annotations = annotations[annotations["frame_number"].isin(frame_ids)]
                if not frame_annotations.empty:
                    self.metadata.append({
                        "video_id": video_id,
                        "frame_paths": [os.path.join(frames_dir, frame_files[j]) for j in
                                        range(i, i + self.clip_length)],
                        "annotations": frame_annotations.to_dict("records"),
                    })

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 加载当前视频片段的元数据
        metadata = self.metadata[idx]
        frame_paths = metadata["frame_paths"]
        frame_annotations = metadata["annotations"]

        # 加载多帧图像
        frames = []
        for frame_path in frame_paths:
            image = read_image(frame_path)  # 形状: (3, H, W), 类型: uint8
            image = image.float() / 255.0  # 转换为浮点型并归一化到 [0, 1]
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        frames = torch.stack(frames, dim=1)  # 形状: (3, clip_length, H, W)

        # 处理检测框、姿态关键点和标签
        boxes = []
        box_player_ids = []
        poses = []
        pose_player_ids = []
        labels = []
        label_player_ids = []

        for ann in frame_annotations:
            # 检测框
            boxes.append([ann["x1"], ann["y1"], ann["x2"], ann["y2"]])
            box_player_ids.append(ann["athlete_id"])

            # 姿态关键点
            keypoints = []
            for i in range(1, 18):
                keypoints.append([ann[f"p{i}x"], ann[f"p{i}y"], ann[f"p{i}s"]])
            poses.append(keypoints)
            pose_player_ids.append(ann["athlete_id"])

            # 动作标签
            labels.append(ann["label"])
            label_player_ids.append(ann["athlete_id"])

        # 转换为张量
        boxes = torch.tensor(boxes, dtype=torch.float32)  # 形状: (N, 4)
        box_player_ids = torch.tensor(box_player_ids, dtype=torch.int64)  # 形状: (N,)
        poses = torch.tensor(poses, dtype=torch.float32)  # 形状: (N, 17, 3)
        pose_player_ids = torch.tensor(pose_player_ids, dtype=torch.int64)  # 形状: (N,)
        labels = torch.tensor(labels, dtype=torch.int64)  # 形状: (N,)
        label_player_ids = torch.tensor(label_player_ids, dtype=torch.int64)  # 形状: (N,)

        # 返回数据
        return {
            "image": frames,  # 图像张量: (3, clip_length, H, W)
            "boxes": boxes,  # 检测框
            "box_player_ids": box_player_ids,  # 检测框对应的运动员 ID
            "poses": poses,  # 姿态关键点
            "pose_player_ids": pose_player_ids,  # 姿态对应的运动员 ID
            "labels": labels,  # 动作标签
            "label_player_ids": label_player_ids,  # 标签对应的运动员 ID
        }
