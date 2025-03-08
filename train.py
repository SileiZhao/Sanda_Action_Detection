import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Resize, Normalize
from models.multimodal_slowfast import SlowFast
from models.heads import ResNetRoIHead
from models.datasets import SandaActionDataset
from utils.config import Config
from utils.train_utils import train, test


def main():
    # 加载配置
    config = Config()

    # 图像预处理
    transform = Compose([
        Resize(config.IMAGE_SIZE),  # 调整图像大小
        Normalize(mean=config.MEAN, std=config.STD),  # 归一化
    ])

    # 创建数据集
    dataset = SandaActionDataset(root_dir=config.DATA_ROOT, transform=transform, clip_length=config.VIDEO_CLIP_LENGTH)

    # 划分数据集为训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    # 初始化 SlowFast 和 ResNetRoIHead
    model = SlowFast(class_num=config.NUM_CLASSES).to(config.DEVICE)
    roi_head = ResNetRoIHead(**config.ROI_HEAD_PARAMS).to(config.DEVICE)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.Adam(
        list(model.parameters()) + list(roi_head.parameters()),
        lr=config.LEARNING_RATE,
    )

    # 训练循环
    best_acc = 0.0
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{config.NUM_EPOCHS}")

        # 训练
        train_loss, train_acc = train(model, roi_head, train_loader, criterion, optimizer, config.DEVICE)

        # 打印训练结果
        print(f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%")

        # 每 TEST_INTERVAL 轮进行一次测试
        if epoch % config.TEST_INTERVAL == 0:
            test_loss, test_acc = test(model, roi_head, test_loader, criterion, config.DEVICE)

            # 打印测试结果
            print(f"Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.2f}%")

            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "roi_head_state_dict": roi_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }, os.path.join(config.CHECKPOINT_DIR, f"best_model.pth"))
                print(f"Best model saved at epoch {epoch} with accuracy {best_acc:.2f}%")

    print("Training complete!")


if __name__ == "__main__":
    main()
