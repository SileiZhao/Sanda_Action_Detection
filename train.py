import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Resize, Normalize
from models.multimodal_slowfast import SlowFast  # 假设模型定义在 models/multimodal_slowfast.py
from models.datasets import SandaActionDataset  # 假设数据集定义在 models/datasets.py
from utils.config import Config  # 配置文件
from utils.train_utils import train, test  # 训练和测试函数


def main():
    # 加载配置
    config = Config()

    # 检查点目录
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)

    # 图像预处理
    transform = Compose([
        Resize(config.IMAGE_SIZE),  # 调整图像大小
        Normalize(mean=config.MEAN, std=config.STD),  # 归一化
    ])

    # 创建数据集
    dataset = SandaActionDataset(root_dir=config.DATA_ROOT, transform=transform, clip_length=config.VIDEO_CLIP_LENGTH)

    # 划分数据集为训练集和测试集
    train_size = int(0.5 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    # 初始化模型
    model = SlowFast(class_num=Config.NUM_CLASSES).to(config.DEVICE)  # 假设有 10 个动作类别

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 每 30 轮学习率下降 10%

    # 训练循环
    best_acc = 0.0
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{config.NUM_EPOCHS}")

        # 训练
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, config.DEVICE)
        scheduler.step()

        # 打印训练结果
        print(f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%")

        # 每 TEST_INTERVAL 轮进行一次测试
        if epoch % config.TEST_INTERVAL == 0:
            test_loss, test_acc = test(model, test_loader, criterion, config.DEVICE)

            # 打印测试结果
            print(f"Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.2f}%")

            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, f"best_model.pth"))
                print(f"Best model saved at epoch {epoch} with accuracy {best_acc:.2f}%")

        # 保存检查点
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
        }, os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth"))

    print("Training complete!")


if __name__ == "__main__":
    main()

