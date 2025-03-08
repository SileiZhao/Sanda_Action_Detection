# utils/config.py
import torch


class Config:
    # 数据集路径
    DATA_ROOT = "./mini_data"

    # 超参数
    BATCH_SIZE = 1
    VIDEO_CLIP_LENGTH = 32
    NUM_EPOCHS = 120
    LEARNING_RATE = 0.001
    CHECKPOINT_DIR = "checkpoints"
    TEST_INTERVAL = 10  # 每 10 轮进行一次测试
    NUM_CLASSES = 2

    # 图像预处理
    IMAGE_SIZE = (256, 256)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # 设备
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
