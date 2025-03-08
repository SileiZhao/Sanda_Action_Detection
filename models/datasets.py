import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize


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

        # 初始化红蓝双方的目标检测框和姿态关键点
        num_frames = len(frame_paths)
        boxes = torch.zeros((num_frames, 2, 4), dtype=torch.float32)  # 形状: (clip_length, 2, 4)
        poses = torch.zeros((num_frames, 2, 17, 3), dtype=torch.float32)  # 形状: (clip_length, 2, 17, 3)
        labels = torch.zeros((num_frames,), dtype=torch.int64)  # 形状: (clip_length,)
        box_player_ids = torch.zeros((num_frames, 2), dtype=torch.int64)  # 形状: (clip_length, 2)
        pose_player_ids = torch.zeros((num_frames, 2), dtype=torch.int64)  # 形状: (clip_length, 2)
        label_player_ids = torch.zeros((num_frames, 2), dtype=torch.int64)  # 形状: (clip_length, 2)

        # 填充红蓝双方的信息
        for ann in frame_annotations:
            frame_idx = ann["frame_number"] - int(os.path.splitext(frame_paths[0])[0].split("_")[-1])
            athlete_id = ann["athlete_id"]

            # 确保 athlete_id 是 0（红方）或 1（蓝方）
            if athlete_id not in [0, 1]:
                continue

            # 填充目标检测框
            boxes[frame_idx, athlete_id] = torch.tensor([ann["x1"], ann["y1"], ann["x2"], ann["y2"]])

            # 填充姿态关键点
            keypoints = []
            for i in range(1, 18):
                keypoints.append([ann[f"p{i}x"], ann[f"p{i}y"], ann[f"p{i}s"]])
            poses[frame_idx, athlete_id] = torch.tensor(keypoints)

            # 填充动作标签（由于红蓝方标签一致，只保留一个）
            labels[frame_idx] = ann["label"]

            # 填充运动员 ID
            box_player_ids[frame_idx, athlete_id] = athlete_id
            pose_player_ids[frame_idx, athlete_id] = athlete_id
            label_player_ids[frame_idx, athlete_id] = athlete_id

        # 返回数据
        return {
            "image": frames,  # 图像张量: (3, clip_length, H, W)
            "boxes": boxes,  # 检测框: (clip_length, 2, 4)
            "box_player_ids": box_player_ids,  # 检测框对应的运动员 ID: (clip_length, 2)
            "poses": poses,  # 姿态关键点: (clip_length, 2, 17, 3)
            "pose_player_ids": pose_player_ids,  # 姿态对应的运动员 ID: (clip_length, 2)
            "labels": labels,  # 动作标签: (clip_length,)
            "label_player_ids": label_player_ids,  # 标签对应的运动员 ID: (clip_length, 2)
        }


if __name__ == '__main__':
    root_dir = '../mini_data'

    transform = Compose([
        Resize((256, 256)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = SandaActionDataset(root_dir=root_dir, transform=transform)

    sample = dataset[0]
    print(sample)
    print(sample["image"].shape)
    print(sample["boxes"].shape)
    print(sample["box_player_ids"].shape)
    print(sample["poses"].shape)
    print(sample["pose_player_ids"].shape)
    print(sample["labels"].shape)
    print(sample["label_player_ids"].shape)

