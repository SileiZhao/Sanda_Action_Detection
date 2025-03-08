import torch
import torch.nn as nn
from tqdm import tqdm
import time
from datetime import timedelta


def merge_bboxes(bboxes, device):
    """
    将红蓝双方的 bboxes 合并为一个大的 bbox，并转换为 (K, 5) 格式。
    Args:
        bboxes (Tensor): 形状为 (batch_size, clip_length, 2, 4)。
    Returns:
        merged_bboxes (Tensor): 形状为 (K, 5)，其中 K = batch_size * clip_length。
    """
    batch_size, clip_length, _, _ = bboxes.shape

    # 取红蓝双方的 bboxes 的最小 x1, y1 和最大 x2, y2
    x1 = torch.min(bboxes[:, :, :, 0], dim=2).values  # (batch_size, clip_length)
    y1 = torch.min(bboxes[:, :, :, 1], dim=2).values  # (batch_size, clip_length)
    x2 = torch.max(bboxes[:, :, :, 2], dim=2).values  # (batch_size, clip_length)
    y2 = torch.max(bboxes[:, :, :, 3], dim=2).values  # (batch_size, clip_length)

    # 合并为 (batch_size, clip_length, 4)
    merged_bboxes = torch.stack([x1, y1, x2, y2], dim=2)  # (batch_size, clip_length, 4)

    # 转换为 (K, 5) 格式，其中 K = batch_size * clip_length
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, clip_length).reshape(-1)  # (K,)
    merged_bboxes = merged_bboxes.reshape(-1, 4).to(device)  # (K, 4)
    merged_bboxes = torch.cat([batch_indices.unsqueeze(1).to(device), merged_bboxes], dim=1)  # (K, 5)

    return merged_bboxes


def get_sequence_label(labels):
    """
    根据 clip_length 中的标签值，确定整个序列的动作标签。
    Args:
        labels (Tensor): 形状为 (batch_size, clip_length)。
    Returns:
        sequence_labels (Tensor): 形状为 (batch_size,)。
    """
    # 统计每个序列中标签 1 的数量
    num_ones = torch.sum(labels == 1, dim=1)  # (batch_size,)
    # 如果标签 1 的数量大于 clip_length 的一半，则序列标签为 1，否则为 0
    sequence_labels = (num_ones > labels.size(1) // 2).long()
    return sequence_labels


def train(model, roi_head, train_loader, criterion, optimizer, device):
    model.train()
    roi_head.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()  # 记录训练开始时间

    for batch_idx, data in enumerate(tqdm(train_loader)):
        # 记录数据加载时间
        data_time = time.time() - start_time

        # 将数据移动到设备
        inputs = data["image"].to(device)  # (batch_size, 3, clip_length, H, W)
        poses = data["poses"].to(device)  # (batch_size, clip_length, 2, 17, 3)
        bboxes = data["boxes"].to(device)  # (batch_size, clip_length, 2, 4)
        labels = data["labels"].to(device)  # (batch_size, clip_length)

        # 合并 bboxes 并转换为 (K, 5) 格式
        bboxes = merge_bboxes(bboxes, device)  # (K, 5)

        # 获取序列标签
        labels = get_sequence_label(labels)  # (batch_size,)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        features = model(inputs, poses)  # 获取三个路径的特征
        outputs = roi_head(features, bboxes)  # 通过检测头获取 ROI 特征

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()

        # 计算梯度范数
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item()
        grad_norm = grad_norm ** 0.5

        optimizer.step()

        # 统计损失和准确率
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 计算每次迭代的时间
        iter_time = time.time() - start_time

        # 计算预计剩余时间
        eta = (len(train_loader) - batch_idx - 1) * iter_time
        eta = str(timedelta(seconds=int(eta)))

        # 获取当前学习率
        lr = optimizer.param_groups[0]["lr"]

        # 获取 GPU 内存占用
        if torch.cuda.is_available():
            memory = torch.cuda.memory_allocated() / 1024 ** 2  # 转换为 MB
        else:
            memory = 0

        # 计算当前准确率
        acc = 100. * correct / total

        # 打印训练日志
        if (batch_idx + 1) % 10 == 0:  # 每 10 个迭代打印一次日志
            print(
                f"{time.strftime('%Y/%m/%d %H:%M:%S')} - INFO - "
                f"Epoch(train) [{epoch}][{batch_idx + 1}/{len(train_loader)}]  "
                f"lr: {lr:.4e}  eta: {eta}  time: {iter_time:.4f}  "
                f"data_time: {data_time:.4f}  memory: {memory:.0f}  "
                f"grad_norm: {grad_norm:.4f}  loss: {loss.item():.4f}  acc: {acc:.4f}%"
            )

        # 重置计时器
        start_time = time.time()

    # 计算平均损失和准确率
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc


def test(model, roi_head, test_loader, criterion, device):
    model.eval()
    roi_head.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()  # 记录测试开始时间

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            # 记录数据加载时间
            data_time = time.time() - start_time

            # 将数据移动到设备
            inputs = data["image"].to(device)  # (batch_size, 3, clip_length, H, W)
            poses = data["poses"].to(device)  # (batch_size, clip_length, 2, 17, 3)
            bboxes = data["boxes"].to(device)  # (batch_size, clip_length, 2, 4)
            labels = data["labels"].to(device)  # (batch_size, clip_length)

            # 合并 bboxes 并转换为 (K, 5) 格式
            bboxes = merge_bboxes(bboxes, device)  # (K, 5)

            # 获取序列标签
            labels = get_sequence_label(labels)  # (batch_size,)

            # 前向传播
            features = model(inputs, poses)  # 获取三个路径的特征
            outputs = roi_head(features, bboxes)  # 通过检测头获取 ROI 特征

            # 计算损失
            loss = criterion(outputs, labels)

            # 统计损失和准确率
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 计算每次迭代的时间
            iter_time = time.time() - start_time

            # 计算预计剩余时间
            eta = (len(test_loader) - batch_idx - 1) * iter_time
            eta = str(timedelta(seconds=int(eta)))

            # 获取 GPU 内存占用
            if torch.cuda.is_available():
                memory = torch.cuda.memory_allocated() / 1024 ** 2  # 转换为 MB
            else:
                memory = 0

            # 计算当前准确率
            acc = 100. * correct / total

            # 打印测试日志
            if (batch_idx + 1) % 10 == 0:  # 每 10 个迭代打印一次日志
                print(
                    f"{time.strftime('%Y/%m/%d %H:%M:%S')} - INFO - "
                    f"Epoch(test) [{epoch}][{batch_idx + 1}/{len(test_loader)}]  "
                    f"eta: {eta}  time: {iter_time:.4f}  "
                    f"data_time: {data_time:.4f}  memory: {memory:.0f}  "
                    f"loss: {loss.item():.4f}  acc: {acc:.2f}%"
                )

            # 重置计时器
            start_time = time.time()

    # 计算平均损失和准确率
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc
