# utils/train_utils.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(model, train_loader, criterion, optimizer, device):
    """
    训练函数。
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, data in enumerate(tqdm(train_loader)):
        inputs = data["image"].to(device)  # 图像输入: (batch_size, 3, clip_length, H, W)
        poses = data["poses"].to(device)  # 姿态关键点: (batch_size, clip_length, 2, 17, 3)
        labels = data["labels"].to(device)  # 动作标签: (batch_size, clip_length, 2)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs, poses)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 统计损失和准确率
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # 计算平均损失和准确率
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc


def test(model, test_loader, criterion, device):
    """
    测试函数。
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs = data["image"].to(device)  # 图像输入: (batch_size, 3, clip_length, H, W)
            poses = data["poses"].to(device)  # 姿态关键点: (batch_size, clip_length, 2, 17, 3)
            labels = data["labels"].to(device)  # 动作标签: (batch_size, clip_length, 2)

            # 前向传播
            outputs = model(inputs, poses)
            loss = criterion(outputs, labels)

            # 统计损失和准确率
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # 计算平均损失和准确率
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc
