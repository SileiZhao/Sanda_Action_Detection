import os
import torch


class Config:
    def __init__(self):
        # 数据集路径
        self.DATA_ROOT = "data"

        # 训练参数
        self.BATCH_SIZE = 16
        self.NUM_EPOCHS = 50
        self.LEARNING_RATE = 0.001
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.IMAGE_SIZE = (256, 256)
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.VIDEO_CLIP_LENGTH = 32
        self.NUM_CLASSES = 2
        self.CHECKPOINT_DIR = "checkpoints"
        self.TEST_INTERVAL = 5

        # ResNetRoIHead 参数
        self.ROI_HEAD_PARAMS = {
            "dim_in": [256, 2048, 2048],  # 三个路径的通道数
            "num_classes": self.NUM_CLASSES,
            "pool_size": [[32, 2, 2], [32, 2, 2], [8, 2, 2]],  # 时间池化大小
            "resolution": [[8, 8], [8, 8], [8, 8]],  # ROIAlign 输出大小
            "scale_factor": [16, 16, 16],  # 空间缩放因子
            "dropout_rate": 0.5,
            "act_func": "softmax",
            "aligned": True,
        }


# 确保检查点目录存在
if not os.path.exists(Config().CHECKPOINT_DIR):
    os.makedirs(Config().CHECKPOINT_DIR)
