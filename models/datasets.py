import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize


class SandaActionDataset(Dataset):
    def __init__(self, root_dir, transform=None, frame_stride=1):
        """
        初始化数据集。

        参数:
            root_dir (str): 数据集根目录。
            transform (callable, optional): 图像预处理变换。
            frame_stride (int): 帧采样步长（默认为1，即使用所有帧）。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.frame_stride = frame_stride

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
            for i, frame_file in enumerate(frame_files):
                if i % self.frame_stride == 0:  # 按步长采样帧
                    frame_id = int(os.path.splitext(frame_file)[0].split("_")[-1])
                    frame_annotations = annotations[annotations["frame_number"] == frame_id]
                    if not frame_annotations.empty:
                        self.metadata.append({
                            "video_id": video_id,
                            "frame_path": os.path.join(frames_dir, frame_file),
                            "annotations": frame_annotations.to_dict("records"),
                        })

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 加载当前帧的元数据
        metadata = self.metadata[idx]
        frame_path = metadata["frame_path"]
        frame_annotations = metadata["annotations"]

        # 加载图像
        image = read_image(frame_path)  # 形状: (3, H, W)
        image = image.float() / 255.0  # 归一化到 [0, 1]

        if self.transform:
            image = self.transform(image)

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
            "image": image,  # 图像张量
            "boxes": boxes,  # 检测框
            "box_player_ids": box_player_ids,  # 检测框对应的运动员 ID
            "poses": poses,  # 姿态关键点
            "pose_player_ids": pose_player_ids,  # 姿态对应的运动员 ID
            "labels": labels,  # 动作标签
            "label_player_ids": label_player_ids,  # 标签对应的运动员 ID
        }


if __name__ == '__main__':

    # 图像预处理
    transform = Compose([
        Resize((256, 256)),  # 调整图像大小
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ])

    # 创建数据集
    dataset = SandaActionDataset(root_dir="../data", transform=transform, frame_stride=1)

    # 测试数据集
    sample = dataset[0]
    print("Image shape:", sample["image"].shape)  # 图像形状: (3, 256, 256)
    print("Boxes shape:", sample["boxes"].shape)  # 检测框形状: (N, 4)
    print("Box player IDs:", sample["box_player_ids"])  # 检测框对应的运动员 ID
    print("Poses shape:", sample["poses"].shape)  # 姿态关键点形状: (N, 17, 3)
    print("Pose player IDs:", sample["pose_player_ids"])  # 姿态对应的运动员 ID
    print("Labels:", sample["labels"])  # 动作标签
    print("Label player IDs:", sample["label_player_ids"])  # 标签对应的运动员 ID
